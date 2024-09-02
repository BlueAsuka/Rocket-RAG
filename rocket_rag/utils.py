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