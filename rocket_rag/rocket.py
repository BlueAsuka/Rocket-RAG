"""
ROCKET Implementation from scratch 
The implementation is modified from the following link:
https://github.com/dida-do/xrocket/blob/main/xrocket/
For a more explaination of the project, please refer to the following link:
https://medium.com/dida-machine-learning/explainable-time-series-classification-with-x-rocket-3087b912a08d
"""

import math
import loguru
import torch
from torch import nn


class RocketConv(nn.Module):
    """Convolutional layer for ROCKET transformation of timeseries.

    This layer serves to transform an observed input time series into a 
    set of time series of the same length based on 1D convolutional activations.
    I.e., a forward pass transforms a tensor of shape (Batch * Channels * Timeobs)
    into a tensor of shape (Batch * Kernels * Channels * Timeobs).

    This implementation mostly conforms to the descriptions in:
    Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
    "Minirocket: A very fast (almost) deterministic transform for time series classification."
    Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.

    In contrast to the paper, all convolutional activations are padded such that they
    match the length of the input sequence instead of randomly applying paddings.

    Attributes:
        in_channels: The number of channels going into the convolutions.
        weight: Tensor of shape (Kernels * 1 * Kernel_length)
        dilation: The dilation value used by the layer.
        padding: The padding widths corresponding to the dilation.
        num_kernels: The number of distinct kernels in the module.
        out_channels: The number of ouput channels.
        patterns: List of the patterns of the convolutional kernels.
        kernel_length: Number of paramters in each kernel.
    """
    
    def __init__(self, in_channels: int, dilation: int, kernel_length: int = 9, alpha: float = -1.0, beta: float = 2.0, pad: bool = True) -> None:
        """Set up attributes including kernels, dilations, and padding values for the layer.

        Args:
            in_channels: Number of channels in each timeseries.
            max_kernel_span: Number of time-observations in a typical timeseries.
            kernel_length: Number of paramters in each kernel, default = 9.
            alpha: Parameter value occuring six times per kernel, default = -1.0.
            beta: Paramter value occuring three times per kernel, default = 2.0.
            max_dilations: Maximum number of distinct dilation values, default = 32.
            pad: Indicates if zero padding should be applied to inputs.
        """
        super().__init__()
        self.in_channels = in_channels
        self.dilation = dilation
        self.kernel_length = kernel_length
        self.pad = pad
        self._initialize_weights(alpha=alpha, beta=beta)
        
    def _initialize_weights(self, alpha: float, beta: float) -> None:
        """Create kernel weights following the scheme in the original paper.

        Kernels with kernel_length=9 are created as combinations of values -1 and 2,
        where the value 2 occurs three times in each kernel.
        """
        beta_indices = torch.combinations(
            torch.arange(self.kernel_length), self.kernel_length // 3
        ).unsqueeze(1)
        weights = torch.full(
            size=[len(beta_indices), 1, self.kernel_length], fill_value=alpha
        ).scatter(2, beta_indices, beta)
        self.weight = nn.Parameter(data=weights, requires_grad=False)
        
    @property
    def padding(self) -> int:
        """The padding length for the set dilation value.

        If padding is set, padding length is set such that each output sequence
        has the same length as the input.
        """
        if self.pad:
            return ((self.kernel_length - 1) * self.dilation) // 2
        else:
            return 0

    @property
    def num_kernels(self) -> int:
        """The number of distinct kernels in the module."""
        return len(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass to calculate channel-wise convolution actications.

        This step is equivalent to depthwise separable convolutions for each channel.
        Outputs are always calculated using zero padding to keep the
        Timeobs dimesion constant (This deviates from the original authors).

        Args:
            x: Tensor of shape (Batch * Channels * Timeobs)

        Returns:
            out: Tensor of shape (Batch * Kernels * Channels * Timeobs)
        """
        # repeat kernels for each channel
        kernels = self.weight.repeat(self.in_channels, 1, 1)

        # calculate convolution activations
        x = nn.functional.conv1d(
            x,
            kernels,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.in_channels,
        )
        x = x.reshape(
            x.shape[0],
            self.num_kernels,
            self.in_channels,
            -1,
        )

        return x
    
    @property
    def patterns(self) -> list:
        """List of the patterns of the convolutional kernels used to extract features."""
        patterns = self.weight.squeeze().tolist()
        return patterns

    @property
    def out_channels(self) -> int:
        """The number of ouput channels corresponds to the number of input channels."""
        return self.in_channels


class ChannelMix(nn.Module):
    """Channel mixing layer for ROCKET transformation of timeseries.

    This layer serves to select and interact the input channels to create uni-
    and multivariate feature sequences.
    I.e., a forward pass transforms a tensor of shape
    (Batch * Kernels * Channels * Timeobs) into a tensor of shape
    (Batch * Kernels * Combinations * Timeobs).

    This implementation is based on the descriptions in:
    Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
    "Minirocket: A very fast (almost) deterministic transform for time series classification."
    Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.

    In contrast to the paper, all channel combinations will be considered, not only
    a randomly selected subset as in the original authors' implementation.

    Attributes:
        in_channels: The number of channels in the data.
        in_kernels: The number of distinct kernels passed to the module.
        order: The maximum number of channels to be interacted.
        method: The channel mixing method, either 'additive' or 'multiplicative'.
        combinations: List of channel combinations being interacted.
        weight: List of parameters for each combination.
        num_combinations: The number of channel combinations considered in the module.
    """

    def __init__(
        self,
        in_channels: int,
        in_kernels: int,
        order: int = 1,
        method: str = "additive",
    ) -> None:
        """Set up attributes including combinations and combination weights for the layer.

        Args:
            in_channels: The number of channels going into the module.
            in_kernels: The number of kernel outputs going into the module.
            order: The maximum number of channels to be interacted.
            method: Keyword to indicate the channel mixing method, default='additive'.
        """
        super().__init__()
        self.in_channels = in_channels
        self.order = order
        self.method = method
        self._initialize_weights(in_kernels=in_kernels)

    def _initialize_weights(self, in_kernels: int) -> None:
        """Set up the channel combinations as weights.

        The created weight tensor will have dimensions
        (Kernels * Combinations * Channels).

        Args:
            in_kernels: The number of kernel outputs going into the module.
        """
        weight = torch.Tensor([])
        for order in range(self.order):
            combinations = torch.combinations(torch.arange(self.in_channels), order + 1)
            channel_map = torch.zeros(len(combinations), self.in_channels).scatter(
                1, combinations, 1
            )
            weight = torch.cat([weight, channel_map])
        weight = weight.unsqueeze(0).repeat(in_kernels, 1, 1)
        self.weight = nn.Parameter(data=weight, requires_grad=False)

    @property
    def num_combinations(self) -> int:
        """The number of channel combinations to be considered."""
        return self.weight.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass to calculate comnbination-wise activations.

        All channel combinations up to the initialized order will be included
        (This deviates from the original authors who use only a random selection).
        Channel activations can be interacted additively or multiplicatively to
        also capture negative correlations (The original authors suggest an
        additive implementation, which is the default here and runs faster).

        Args:
            x: Tensor of shape (Batch * Kernels * Channels * Timeobs)

        Returns:
            out: Tensor of shape (Batch * Kernels * Combinations * Timeobs)
        """
        if self.method == "additive":
            x = self.weight.matmul(x)
        elif self.method == "multiplicative":
            x = x[:, :, None, :, :].mul(self.weight[None, :, :, :, None])
            x = x.where(x != 0, torch.ones(1, device=x.device)).prod(dim=-2)
        else:
            raise ValueError(f"method '{self.method}' not available")

        return x

    @property
    def combinations(self) -> list:
        """A list of the channel weightings considered."""
        return self.weight.reshape(-1, self.in_channels).tolist()
    

class PPVThresholds(nn.Module):
    """Threshold layer for ROCKET transformation of timeseries.

    This layer serves to apply proportion of positive values pooling based on a
    set of threshold values from the convolutional outputs.
    I.e., a forward pass transforms a tensor of shape
    (Batch * Kernels * Combinations * Timeobs) into a tensor of shape
    (Batch * Kernels * Combinations * Thresholds).
    LAyer needs to be fitted to define the values of the thresholds.

    This implementation conforms to the descriptions in:
    Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
    "Minirocket: A very fast (almost) deterministic transform for time series classification."
    Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.

    Attributes:
        num_thresholds: The number of thresholds per channel.
        is_fitted: Indicates that thresholds are fitted to a data example.
        bias: Tensor of shape (Kernels * Dilations * Combinations * Quantiles)
    """

    def __init__(
        self,
        num_thresholds: int,
    ) -> None:
        """Set up attributes including quantile values for the layer.

        Args:
            num_thresholds: The number of thresholds to be considered per channel.
        """
        super().__init__()
        self.num_thresholds = num_thresholds
        self.is_fitted: bool = False

    def _select_quantiles(
        self,
        num_thresholds: int,
        uniform: bool = False,
    ) -> torch.Tensor:
        """Automatically selects the quantile values to initialize threshold values.

        Following the original authors' code, the module uses a "low-discrepancy
        sequence to assign quantiles to kernel/dilation combinations", source:
        https://github.com/angus924/minirocket/blob/main/code/minirocket_multivariate.py

        Alternatively, quantiles can be choosen to be uniformly spaced in [0, 1].

        Args:
            num_thresholds: The number of thresholds to be considered per channel.
            uniform: Indicates if quantiles should be uniformly spaced, default=False.
        """
        if uniform:
            # uniformly spaced quantiles
            quantiles = torch.linspace(0, 1, num_thresholds + 2)[1:-1]
        else:
            # low-discrepancy sequence to assign quantiles 
            phi = (math.sqrt(5) + 1) / 2
            quantiles = torch.Tensor(
                [((i + 1) * phi) % 1 for i in range(num_thresholds)]
            )

        return quantiles

    def fit(self, x: torch.Tensor, quantiles: list = None) -> None:
        """Obtain quantile values from the first available example to use as thresholds.

        Accepts either a single example or a batch as an input.

        Args:
            x: Tensor of shape (Batch * Kernels * Combinations * Timeobs)
            quantiles (optional): A list of values between 0 and 1 to indicate the quantiles
                at which to set the thresholds.
        """
        # get quantiles
        if quantiles is None:
            quantiles = self._select_quantiles(
                num_thresholds=self.num_thresholds, uniform=False
            )
        else:
            if type(quantiles) != list or not all(0 <= q <= 1 for q in quantiles):
                raise ValueError(
                    "quantiles needs to be a list of values between 0 and 1"
                )

        # flatten if input is a batch
        if len(x.shape) == 4:
            x = x.movedim(0, -1).flatten(start_dim=-2)

        # extract threshold values from activation quantiles
        thresholds = x.quantile(q=quantiles.to(x.device), dim=-1).movedim(
            source=0, destination=-1
        )

        # set attributes
        self.bias = nn.Parameter(data=thresholds, requires_grad=False)
        self.is_fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass to calculate pooled features.

        Module weights will be fit to the first example if not yet fitted.

        Args:
            x: Tensor of shape (Batch * Kernels * Combinations * Timeobs)

        Returns:
            out: Tensor of shape (Batch * Kernels * Combinations * Thresholds)
        """
        # fit if module is not yet fitted
        if not self.is_fitted:
            self.fit(x)
            loguru.logger.warning("automatically fit biases to first training example")

        # add missing dims
        thresholds = self.bias.unsqueeze(-1)
        x = x.unsqueeze(-2)

        # apply percentage of values pooling with thresholds
        x = x.gt(thresholds).sum(dim=-1).div(x.size(-1))
        return x

    @property
    def thresholds(self) -> list:
        """A list of the threshold values considered."""
        return self.bias.flatten().tolist()
    

class DilationBlock(nn.Module):
    """MiniRocket block for transformation of timeseries at a single dilation value.

    This layer serves to perform the encoding of an input timeseries with a fixed
    dilation value.
    A DilationBlock consists of the following three sublayers:
     - RocketConv
     - ChannelMix
     - PPVThresholds
    A forward pass transforms a tensor of shape (Batch * Channels * Timeobs)
    into a tensor of shape (Batch * (Features/Dilation)).

    This implementation is based on the descriptions in:
    Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
    "Minirocket: A very fast (almost) deterministic transform for time series classification."
    Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.

    The block structure deviates from the original paper and sublayers have differences as
    explained in the respective implementations.

    Attributes:
        in_channels: Number of channels in each timeseries.
        dilation: The dilation value to apply to the convolutional kernels.
        num_thresholds: The number of thresholds per channel combination.
        combination_order: The maximum number of channels to be interacted.
        combination_method: The channel mixing method, either 'additive' or 'multiplicative'.
        kernel_length: Number of paramters in each kernel, default = 9.
        num_kernels: The number of kernels considered in the module.
        num_combinations: The number of channel combinations considered in the module.
        feature_names: (pattern, dilation, channels, threshold) tuples to identify features.
        is_fitted: Indicates that thresholds are fitted to a data example.
    """

    def __init__(
        self,
        in_channels: int,
        dilation: int,
        num_thresholds: int = 1,
        combination_order: int = 1,
        combination_method: str = "additive",
        kernel_length: int = 9,
    ):
        """Set up attributes including quantile values for the layer.

        Args:
            in_channels: Number of channels in each timeseries.
            dilation: The dilation value to apply to the convolutional kernels.
            num_thresholds: The number of thresholds per channel combination.
            combination_order: The maximum number of channels to be interacted.
            combination_method: Keyword for the channel mixing method, default='additive'.
            kernel_length: Number of paramters in each kernel, default = 9.
        """
        super().__init__()

        # set up constituent layers
        self.conv = RocketConv(
            in_channels=in_channels,
            dilation=dilation,
            kernel_length=kernel_length,
        )
        self.mix = ChannelMix(
            in_channels=self.conv.out_channels,
            in_kernels=self.conv.num_kernels,
            order=combination_order,
            method=combination_method,
        )
        self.thresholds = PPVThresholds(
            num_thresholds=num_thresholds,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass to calculate a feature vector.

        Pooling thresholds will be fit to the first example if not yet fitted.

        Args:
            x: Tensor of shape (Batch * Channels * Timeobs)

        Returns:
            x: Tensor of shape (Batch * (Features/Dilations))
        """
        x = self.conv(x)
        x = self.mix(x)
        x = self.thresholds(x)
        x = torch.flatten(x, start_dim=1)
        return x

    @property
    def in_channels(self) -> int:
        """The number of incoming channels."""
        return self.conv.in_channels

    @property
    def dilation(self) -> int:
        """The value to dilute the kernels with over the time dimension."""
        return self.conv.dilation

    @property
    def combination_order(self) -> int:
        """The highest number of channels to combine in a feature."""
        return self.mix.order

    @property
    def num_kernels(self) -> int:
        """The number of kernels in the convolutional block."""
        return self.conv.num_kernels

    @property
    def num_combinations(self) -> int:
        """The total number of channel combinations."""
        return self.mix.num_combinations

    @property
    def num_thresholds(self) -> int:
        """The number of thresholds to apply to each channel combinations."""
        return self.thresholds.num_thresholds

    def fit(self, x: torch.Tensor) -> None:
        """Obtain pooling threshold values from an input.

        Accepts either a single example or a batch as an input.

        Args:
            x: Tensor of shape (Channels * Timeobs) or
                Tensor of shape (Batch * Channels * Timeobs)
        """
        x = self.conv(x)
        x = self.mix(x)
        self.thresholds.fit(x)

    @property
    def is_fitted(self) -> bool:
        """Indicates if module biases were fitted to data."""
        return self.thresholds.is_fitted

    @property
    def feature_names(self) -> list[tuple]:
        """(pattern, dilation, channels, threshold) tuples to identify features."""
        assert self.is_fitted, "module needs to be fitted for thresholds to be named"
        feature_names = [
            (
                str(pattern),
                self.dilation,
                str(channels),
                f"{threshold:.4f}",
            )
            for pattern, channels, threshold in zip(
                self.conv.patterns * self.num_combinations * self.num_thresholds,
                self.mix.combinations * self.num_thresholds,
                self.thresholds.thresholds,
            )
        ]
        return feature_names


class XRocket(nn.Module):
    """Explainable ROCKET module for timeseries embeddings.

    Serves to encode a (multivariate) timeseries into a fixed-length feature vector.
    I.e., a forward pass transforms a tensor of shape (Batch * Channels * Timeobs)
    into a tensor of shape (Batch * Features).
    The implementation is such that the origin of each feature can be traced.

    This implementation is based on the descriptions in:
    Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
    "Minirocket: A very fast (almost) deterministic transform for time series classification."
    Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.

    The implemented block structure deviates from the original paper but the calculations are
    almost identical. Please refer to the sublayer implementations for details.

    Attributes:
        in_channels: Number of channels in each timeseries.
        max_kernel_span: Number of time-observations in a typical timeseries.
        combination_order: The maximum number of channels to be combined.
        combination_method: The channel mixing method, either 'additive' or 'multiplicative'.
        feature_dims: The number of values in each feature dimension.
        num_features: The total number of feature embeddings.
        is_fitted: Indicates of the module has been fitted to data.
        feature_names: List of feature name tuples (pattern, dilation, channels, threshold).
    """

    def __init__(
        self,
        in_channels: int,
        max_kernel_span: int,
        combination_order: int = 1,
        combination_method: str = "additive",
        feature_cap: int = 10_000,
        kernel_length: int = 9,
        max_dilations: int = 32,
    ):
        """Set up attributes for all sub-layers.

        Args:
            in_channels: The number of channels in the data.
            max_kernel_span: Number of time-observations in a typical timeseries.
            combination_order: The maximum number of channels to be interacted.
            combination_method: Keyword for the channel mixing method, default='additive'.
            feature_cap: Maximum number of features to be considered.
            kernel_length: The length of the 1D convolutional kernels.
            max_dilations: The maximum number of distinct dilation values.
        """
        super().__init__()
        self.dilations = self._deduce_dilation_values(
            max_kernel_span=max_kernel_span,
            kernel_length=kernel_length,
            max_dilations=max_dilations,
        )
        num_kernels = len(
            torch.combinations(torch.arange(kernel_length), kernel_length // 3)
        )
        num_combinations = sum(
            [
                len(torch.combinations(torch.arange(in_channels), order + 1))
                for order in range(combination_order)
            ]
        )
        num_mix_channels = self.num_dilations * num_kernels * num_combinations
        if feature_cap < num_mix_channels:
            raise ValueError(
                (
                    f"input combinations ({num_mix_channels}) "
                    f"greater than feature cap ({feature_cap})."
                )
            )
        num_thresholds = feature_cap // num_mix_channels

        # set up rocket blocks
        self.blocks = nn.ModuleList()
        for dilation in self.dilations:
            self.blocks.append(
                DilationBlock(
                    in_channels=in_channels,
                    dilation=dilation,
                    num_thresholds=num_thresholds,
                    combination_order=combination_order,
                    combination_method=combination_method,
                )
            )

    def _deduce_dilation_values(
        self,
        max_kernel_span: int,
        kernel_length: int,
        max_dilations: int,
    ) -> None:
        """Create dilation values following the scheme in the original paper.

        Dilation values are chosen according to the number of observations
        with the formula in the paper.
        """
        max_exponent = math.log((max_kernel_span - 1) / (kernel_length - 1), 2)
        integers = (
            (2 ** torch.linspace(0, max_exponent, max_dilations)).to(dtype=int).tolist()
        )
        return list(set(integers))

    @property
    def num_dilations(self) -> int:
        """The number of distinct dilation values for each kernel."""
        return len(self.dilations)

    @property
    def num_kernels(self) -> int:
        """The number of convolutional kernels per dilation."""
        return self.blocks[0].num_kernels

    @property
    def num_combinations(self) -> int:
        """The number of channel combinations per dilation."""
        return self.blocks[0].num_combinations

    @property
    def num_thresholds(self) -> int:
        """The number of pooling thresholds per channel combination per dilation."""
        return self.blocks[0].num_thresholds

    @property
    def feature_dims(self) -> dict:
        """A dictionary with the number of each feature attribute."""
        feature_dims = {
            "num_kernels": self.num_kernels,
            "num_dilations": self.num_dilations,
            "num_combinations": self.num_combinations,
            "num_thresholds": self.num_thresholds,
        }
        return feature_dims

    @property
    def num_features(self) -> int:
        """The total number of feature encodings used by the layer."""
        num_features = (
            self.num_kernels
            * self.num_dilations
            * self.num_combinations
            * self.num_thresholds
        )
        return num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass to calculate timeseries feature encodings.

        Args:
            x: Tensor of shape (Batch * Channels * Timeobs)

        Returns:
            out: Tensor of shape (Batch * Features)
        """
        out = torch.cat([block(x) for block in self.blocks], dim=-1)
        return out

    def fit(self, x: torch.Tensor) -> None:
        """Obtain parameter valies from the first available example.

        Accepts either a single example or a batch as an input.

        Args:
            x: Tensor of shape (Channels * Timeobs) or
                Tensor of shape (Batch * Channels * Timeobs)
        """
        for block in self.blocks:
            block.fit(x)

    @property
    def is_fitted(self) -> bool:
        """Indicates if module biases were fitted to data."""
        return self.blocks[0].is_fitted

    @property
    def feature_names(self) -> list:
        """(pattern, dilation, channels, threshold) tuples to identify features."""
        assert self.is_fitted, "module needs to be fitted for thresholds to be named"
        feature_names = []
        for block in self.blocks:
            feature_names += block.feature_names
        return feature_names

    @property
    def in_channels(self) -> int:
        """Number of channels in each timeseries."""
        return self.blocks[0].in_channels

    @property
    def combination_order(self) -> int:
        """The maximum number of channels to be combined."""
        return self.blocks[0].mix.order

    @property
    def device(self) -> torch.device:
        """The device the module is loaded on."""
        return next(self.parameters()).device
