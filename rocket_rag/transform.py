import re
import os
import json
import loguru
import pandas as pd
import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List
from pathlib import Path
from tsai.models import MINIROCKET_Pytorch
from openai import OpenAI


class TimeSeriesTransform:
    """Time series transform class"""
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.default_smoothing_method = cfg['DEFAULT_SMOOTHING_METHOD']
    
    def smoothing(self, ts_df: pd.DataFrame, field: str, method: str=None) -> np.ndarray:
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
        
        method = self.default_smoothing_method if method is None else method
        
        if method == 'ewma':
            smoothed = ts_df[field].ewm(span=self.cfg['EWA_SPAN']).mean().values
        elif method == 'ma':
            smoothed = ts_df[field].rolling(window=self.cfg['MA_WINDOW_SIZE']).mean().values
        elif method == 'gaussian':
            smoothed = gaussian_filter1d(ts_df[field].values, sigma=self.cfg['GAUSSIAN_SIGMA'])
        elif method == 'savgol_filter':
            smoothed = savgol_filter(ts_df[field].values, window_length=self.cfg['SAVGOL_WINDOW_SIZE'], polyorder=self.cfg['SAVGOL_POLYORDER'])
        else:
            loguru.logger.warning(f"Smoothing method {method} is not supported, use {self.cfg['DEFAULT_SMOOTHING_METHOD']} instead.")
            smoothed = self.smoothing(ts_df, field, method=self.cfg['DEFAULT_SMOOTHING_METHOD'])
        
        return np.array(smoothed)


    def get_fft(self, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Fast Fourier Transform on the time series

        Args:
            ts_df: a pandas dataframe containing the time series
            
        Return:
            The numpy array of the FFT result
        """
        
        if not isinstance(ts, np.ndarray):
            ts = np.array(ts)
        
        fft_values = np.fft.fft(ts)
        fft_freqs = np.fft.fftfreq(len(ts))
        
        positive_freq_idx = np.where(fft_freqs >= 0)
        return fft_values[positive_freq_idx], fft_freqs[positive_freq_idx]


    def get_rocket(self, ts: np.ndarray) -> np.ndarray:
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
        
        mrf = MINIROCKET_Pytorch.MiniRocketFeatures(c_in=ts_array.shape[1], seq_len=ts_array.shape[-1], random_state=self.cfg['RANDOM_STATE'])
        mrf.fit(ts_array)
        rocket_feature = MINIROCKET_Pytorch.get_minirocket_features(ts_array, mrf)
        return rocket_feature.squeeze()


    def get_ApEn(self, ts: np.ndarray) -> float:
        """
        Get the approximate entropy of the time series

        Args:
            ts_df: a pandas dataframe containing the time series
            field: a string of the field in the dataframe for ApEn

        Return:
            The approximate entropy of the time series
        """
        pass


    def rocket_batch_transform(self, ts: np.ndarray) -> np.ndarray:
        """
        Get the time series transformation using the ROCKET method

        Args:

        Return: 
        """
        
        # TODO: Use torch to do the batch transformation
        pass


class TextTransform:
    """Build the semantic chunker for text splitting """
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
    def split_text(self, text: str) -> List[str]:
        """Split the given text based on the semantic structure of the text
        
        Args:
            text (str): The text to be split

        Returns:
            list[str]: The list of chunks
        """
        
        split_sents = self._split_sentences(text)
        combin_sents = self._combine_sentences(split_sents)
        sents_embeds = self.get_embedding_from_openai(combin_sents)
        sents_dists = self._calculate_cosine_similarity(sents_embeds)
        
        breakpoint_percentile_threshold = 80
        breakpoint_distance_threshold = np.percentile(sents_dists, breakpoint_percentile_threshold)
        
        indices_above_thresh = [i for i, distance in enumerate(sents_dists) if distance > breakpoint_distance_threshold]
        
        chunks = []
        start_index = 0
        for index in indices_above_thresh:
            chunk = ' '.join(split_sents[start_index:index+1])
            chunks.append(chunk)
            start_index = index + 1
        
        if start_index < len(split_sents):
            chunk = ' '.join(split_sents[start_index:])
            chunks.append(chunk)
        
        return chunks
    
    def get_embedding_from_openai(self, texts: str) -> List[float]:
        """Get the embedding from OpenAI API"""
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        try:
            response = client.embeddings.create(
                input=texts,
                model=self.cfg['EMB_MODEL']
            )
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings
        except Exception as e:
            print("An error occurred:", e)
            return np.array([])  # Return an empty array in case of an error

    def _split_sentences(self, text: str) -> List[str]:
        """Split the given text into sentences based on punctuation followed by whitespace.
        
        Args:
            text (str): The text to be split into sentences

        Returns:
            list[str]: The list of sentences
        """
        
        return  re.split(r'(?<=[.?!])\s+', text)

    def _combine_sentences(self, sentences: List[str]) -> List[str]:
        """ Create a buffer by combining each sentence with its previous and next sentence to provide a wider context. 
        For example, combine sentences 1,2,3 and 2,3,4 before computing the cosine simiality.
        
        Args:
            sentences (list[str]): The list of sentences to be combined
            
        Returns:
            list[str]: The list of combined sentences
        """
        
        combined_sentences = []
        for i in range(len(sentences)):
            combined_sentence = sentences[i]
            if i > 0:
                combined_sentence = sentences[i-1] + ' ' + combined_sentence
            if i < len(sentences) - 1:
                combined_sentence += ' ' + sentences[i+1]
            combined_sentences.append(combined_sentence)
        return combined_sentences

    def _calculate_cosine_similarity(self, embeddings: np.ndarray) -> List[float]:
        """ Calculate the cosine distance (1 - cosine similarity) between consecutive embeddings.
        
        Args:
            embeddings (np.ndarray): The embeddings to be compared
            
        Returns:
            list[float]: The list of cosine distances
        """
        
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - similarity
            distances.append(distance)
        return distances


class ImageTransform:
    pass


if __name__ == "__main__":
    cfg_path = os.path.join(
            os.path.abspath(Path(os.path.dirname(__file__)).parent.absolute()),
            "config/configs.json"
            )
    cfg = json.load(open(cfg_path))
    
    loguru.logger.debug(f"Testing the time series transformation")
    ts_trans = TimeSeriesTransform(cfg=cfg)
    loguru.logger.debug(f"Testing the text transformation")
    txt_trans = TextTransform(cfg=cfg)
    # loguru.logger.debug(f"Testing the image transformation")
    # img_trans = ImageTransform(cfg=cfg)
    loguru.logger.debug(f"DONE")
