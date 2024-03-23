"""
Helper functions
"""

import os
import loguru
import pandas as pd
import numpy as np

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
SMOOTHING_PE_WINDOW_SIZE = 20
SMOOTHING_CUR_WINDOW_SIZE = 15


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

def moving_average_smoothing(ts: Union[List[float], np.ndarray], 
                             window_size: int, 
                             tolist: bool, 
                             warning: bool=False) -> np.ndarray:
    """
    Smooth the time series using moving average approach

    Args:
        time_series: an numpy array containing the time series
        window_size: the window size for averaging to smooth the time series 
        tolist: whether return the array in list or warp it in np.ndarray
        warning: the boolean to determine whether show the warning info
    
    Return:
        The numpy array of the time series after smoothing
    """
    if not isinstance(ts, np.ndarray):
        if warning:
            loguru.logger.warning(f"The input requires np.ndarray, but {type(ts)} is provided.")
            loguru.logger.warning(f"Automatically cast {type(ts)} to np.ndarray, further type checking is recommanded.")
        ts = np.array(ts)

    S = np.zeros(ts.shape[0])
    for i in range(ts.shape[0]):
        if i < window_size:
            S[i] = np.mean(ts[:i+1])
        else:
            S[i] = np.mean(ts[i-window_size:i+1])
    return S.tolist() if tolist else S

def fit(ts_filename: str, 
        field: str, 
        smooth: bool, 
        smooth_ws: int, 
        tolist: bool, 
        verbo: bool=False) -> np.ndarray:
    """
    Fit the workflow including read csv file, extract data from field and moveing average smoothing
    
    Args:
        ts_filename: the string of the csv filename
        field: the string of the field in the pd.dataframe for extraction
        smooth: the boolean to determine whether smooth the time series
        smooth_ws: the window size for averaging to smooth the time series 
        tolist: whether return the array in list or warp it in np.ndarray
        verbo: the boolean to determine whether show the logging information
    
    Return:
        The numpy array of the time series
    """
    # loguru.logger.debug(f"Reading files...")
    df = read_time_series_csv(ts_filename, verbo=False)
    temp_arr = extract_data_from_df(df, field)
    if smooth:
        temp_arr = moving_average_smoothing(temp_arr, window_size=smooth_ws, tolist=tolist)
    if verbo:
        loguru.logger.info(f"time series extracted SUCCESSFULLY.")
    return temp_arr

def fit_transform(ts_filename: List[str],
                  field: str,
                  smooth: bool,
                  smooth_ws: int,
                  tolist: bool,
                  verbo: bool) -> np.ndarray:
    """
    Fit the workflow of the extracting data from a csv file and transform it by ROCKET

    Args:
        ts_filename: the string of the csv filename
        field: the string of the field in the pd.dataframe for extraction
        smooth: the boolean to determine whether smooth the time series
        smooth_ws: the window size for averaging to smooth the time series 
        tolist: whether return the array in list or warp it in np.ndarray
        verbo: the boolean to determine whether show the logging information
    
    Return:
        The numpy array of the ROCKET features of the time series
    """
    if verbo:
        loguru.logger.debug(f'Extract the time series data points')
    if not isinstance(ts_filename, list):
        loguru.logger.error(f'The input ts_filename must be a list, but got {type(ts_filename)}, \
                                The single filename can be warpped by [] for valid input.')

    ts = np.array([fit(ts_filename=tf,
                         field=field,
                         smooth=smooth,
                         smooth_ws=smooth_ws,
                         tolist=tolist,
                         verbo=verbo) for tf in ts_filename])
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
