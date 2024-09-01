"""
Helper functions
"""

import os
import loguru
import pandas as pd
import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from typing import List, Union, Dict
from pyts.transformation import ROCKET


RAW_DATA_DIR = '../data/raw/'
INSTANCES_DIR = '../data/instances/'
INFERENCE_DIR = '../data/inference/'
STATES = ['normal', 
          'backlash1', 'backlash2',
          'lackLubrication1', 'lackLubrication2',
          'spalling1', 'spalling2', 'spalling3', 'spalling4',
          'spalling5', 'spalling6', 'spalling7', 'spalling8']
LOADS= ['20kg', '40kg', '-40kg']

DEFAULT_SMOOTHING_METHOD = 'savgol_filter'

MA_WINDOW_SIZE = 15
EWA_SPAN = 10
GAUSSIAN_SIGMA = 2
SAVGOL_WINDOW_SIZE = 15
SAVGOL_POLYORDER = 2


def read_time_series_csv(filepath: str, verbo: bool=False) -> pd.DataFrame:

    """
    Read time series csv file using pandas

    Args:
        filepath: a string of the file path
        verbo: a boolean to determine whether to show the logging infomation 
    
    Return:
        A pandas Dataframe  
    """

    try:
        df = pd.read_csv(filepath)
        if verbo:
            loguru.logger.info(f"Read {filepath} SUCCESSFULLY.")
        return df
    except FileNotFoundError as e:
        raise e

def extract_data_from_df(df: pd.DataFrame, field: str, verbo: bool=False) -> List[float]:
    """
    Extract data from a specific field in an dataframe

    Args:
        df: the dataframe for data extraction
        field: a string of the field in the dataframe for extraction
        verbo: a boolean to determine whether to show the logging infomation 
    
    Return:
        A list of float in the field of the dataframe
    """

    if field not in df.columns:
        loguru.logger.error(f"{field} is not included in the dataframe.")
        return []
    
    if verbo:
        loguru.logger.info(f"Extract {field} column from the dataframe.")
    return df[field].tolist()

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

def read_csv_and_smooth(ts_filename: str, field: str) -> np.ndarray:
    """
    Read the .csv file, extract field data from the provided dataframe and moveing average smoothing
    
    Args:
        ts_filename: the string of the csv filename
        field: the string of the field in the pd.dataframe for extraction
        verbo: the boolean to determine whether show the logging information
    
    Return:
        The numpy array of the time series
    """
    # loguru.logger.debug(f"Reading files...")
    df = read_time_series_csv(ts_filename, verbo=False)
    # temp_arr = extract_data_from_df(df, field)
    temp_arr = smoothing(df, field)
    return temp_arr

def fit_transform(ts_filename: List[str], field: str, verbo: bool) -> np.ndarray:
    """
    Fit the workflow of the extracting data from a csv file and transform it by ROCKET

    Args:
        ts_filename: the string of the csv filename
        field: the string of the field in the pd.dataframe for extraction
        verbo: the boolean to determine whether show the logging information
    
    Return:
        The numpy array of the ROCKET features of the time series
    """
    if verbo:
        loguru.logger.debug(f'Extract the time series data points')
    if not isinstance(ts_filename, list):
        loguru.logger.warning(f'The single filename string is warpped by [] for valid input.')
        ts_filename = [ts_filename]

    ts = np.array([read_csv_and_smooth(ts_filename=tf, field=field) for tf in ts_filename])
    
    if verbo:
        loguru.logger.debug(f'Rocket transforming...')
    rocket = ROCKET(n_kernels=10000, kernel_sizes=([9]), random_state=42)
    rocket_features = rocket.fit_transform(ts)
    return rocket_features

def parse_files(main_directory: str) -> Dict[str, List[str]]:
    """
    extract all files from a given directory
    
    Args:
        main_directory: the string of the main directory
    
    Return:
        Return a dictionary {load1: [], load2: [], load3: []}
    """
    load_state_dict = {load: [] for load in LOADS}
    for load in LOADS:
        for state in STATES:
            directory = os.path.join(main_directory, load, state)
            files = [os.path.join(directory, f) for f in os.listdir(directory) 
                     if os.path.isfile(os.path.join(directory, f))]
            load_state_dict[load].extend(files)
    return load_state_dict 

def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: the string of the text to be truncated
        max_length: the integer of the maximum length
    
    Return:
        The truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
