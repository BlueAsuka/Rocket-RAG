import os
import json
import loguru
import pandas as pd
import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from typing import Tuple
from pathlib import Path

# from rocket import Rocket
from tsai.imports import default_device
from tsai.models import MINIROCKET_Pytorch

cfg_path = os.path.join(
            os.path.abspath(Path(os.path.dirname(__file__)).parent.absolute()),
            "config/configs.json"
            )
cfg = json.load(open(cfg_path))


def smoothing(ts_df: pd.DataFrame, field: str, method: str=cfg['DEFAULT_SMOOTHING_METHOD']) -> np.ndarray:
    """
    Smooth the time series 

    Args:
        ts_df: a pandas dataframe containing the time series
        field: a string of the field in the dataframe for smoothing
        method: the string of the smoothing method, only support four methods:
                Exponential Moving Average (EMA),
                Moving Average (MA),
                Gaussian Smoothing,
                Savitzky-Golay Filter
    
    Return:
        The numpy array of the time series after smoothing
    """
    
    if field not in ts_df.columns:
        loguru.logger.error(f"{field} is not included in the dataframe.")
    
    if method == 'ewma':
        smoothed = ts_df[field].ewm(span=cfg['EWA_SPAN']).mean().values
    elif method == 'ma':
        smoothed = ts_df[field].rolling(window=cfg['MA_WINDOW_SIZE']).mean().values
    elif method == 'gaussian':
        smoothed = gaussian_filter1d(ts_df[field].values, sigma=cfg['GAUSSIAN_SIGMA'])
    elif method == 'savgol_filter':
        smoothed = savgol_filter(ts_df[field].values, window_length=cfg['SAVGOL_WINDOW_SIZE'], polyorder=cfg['SAVGOL_POLYORDER'])
    else:
        loguru.logger.warning(f"Smoothing method {method} is not supported, use {cfg['DEFAULT_SMOOTHING_METHOD']} instead.")
        smoothed = smoothing(ts_df, field, warning=True, method=cfg['DEFAULT_SMOOTHING_METHOD'])
    
    return np.array(smoothed)


def fft(ts: np.ndarray, field: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Fast Fourier Transform on the time series

    Args:
        ts_df: a pandas dataframe containing the time series
        field: a string of the field in the dataframe for FFT
        
    Return:
        The numpy array of the FFT result
    """
    
    if not isinstance(ts, np.ndarray):
        loguru.logger.error(f"{field} is not a numpy array.")
        ts = np.array(ts)
    
    fft_values = np.fft.fft(ts)
    fft_freqs = np.fft.fftfreq(len(ts))
    
    positive_freq_idx = np.where(fft_freqs >= 0)
    return fft_values[positive_freq_idx], fft_freqs[positive_freq_idx]


def rocket_transform(ts: np.ndarray) -> np.ndarray:
    """
    Get the time series transformation using the ROCKET method
    
    Args:
        ts: a numpy array of the time series in the shape of (seq_len,)
    
    Return:
        The numpy array of the time series after transformation
    """
    
    if not isinstance(ts, np.ndarray):
        loguru.logger.error(f"{ts} is not a numpy array.")
        ts = np.array(ts)
        
    if len(ts.shape) != 1:
        loguru.logger.error(f"{ts} is not a 1D array.")
        return np.array([])
    
    # Add the batch and in_channels dimensions to the time series array
    # The shape of the input is (batch_size, in_channels, seq_len)
    # Also convert the data type to float32
    ts_array = np.expand_dims(np.array(ts, dtype=np.float32), axis=(0, 1))
    assert len(ts_array.shape) == 3
    assert ts_array.dtype == np.float32
    
    mrf = MINIROCKET_Pytorch.MiniRocketFeatures(c_in=ts_array.shape[1], seq_len=ts_array.shape[-1])
    mrf.fit(ts_array)
    rocket_feature = MINIROCKET_Pytorch.get_minirocket_features(ts_array, mrf)
    return rocket_feature.squeeze()


def rocket_batch_transform(ts: np.ndarray) -> np.ndarray:
    """
    Get the time series transformation using the ROCKET method

    Args:

    Return: 
    """
    
    # TODO: Use torch to do the batch transformation
    pass


if __name__ == "__main__":
    pass
