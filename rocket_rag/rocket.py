"""
ROCKET Implementation from scratch 
The implementation is modified from the following link:
https://github.com/dida-do/xrocket/blob/main/xrocket/
For a more explaination of the project, please refer to the following link:
https://medium.com/dida-machine-learning/explainable-time-series-classification-with-x-rocket-3087b912a08d
"""

import torch
from torch import nn


class Rocket(nn.Module):
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
        
        