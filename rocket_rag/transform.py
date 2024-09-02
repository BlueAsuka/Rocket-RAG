import loguru
import pandas as pd
import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from typing import Tuple


DEFAULT_SMOOTHING_METHOD = 'savgol_filter'

MA_WINDOW_SIZE = 15
EWA_SPAN = 10
GAUSSIAN_SIGMA = 2
SAVGOL_WINDOW_SIZE = 15
SAVGOL_POLYORDER = 2


def smoothing(ts_df: pd.DataFrame, field: str, method: str=DEFAULT_SMOOTHING_METHOD) -> np.ndarray:
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
        smoothed = ts_df[field].ewm(span=EWA_SPAN).mean().values
    elif method == 'ma':
        smoothed = ts_df[field].rolling(window=MA_WINDOW_SIZE).mean().values
    elif method == 'gaussian':
        smoothed = gaussian_filter1d(ts_df[field].values, sigma=GAUSSIAN_SIGMA)
    elif method == 'savgol_filter':
        smoothed = savgol_filter(ts_df[field].values, window_length=SAVGOL_WINDOW_SIZE, polyorder=SAVGOL_POLYORDER)
    else:
        loguru.logger.warning(f"Smoothing method {method} is not supported, use {DEFAULT_SMOOTHING_METHOD} instead.")
        smoothed = smoothing(ts_df, field, warning=True, method=DEFAULT_SMOOTHING_METHOD)
    
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


def rocket_transform():
    pass
