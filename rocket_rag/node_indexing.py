"""
Node indexing for vector store
The node indexer will include the BoWs transformation and vector embedding 
After constructing nodes containing dense and sparse embeddings, nodes will be stored in the vector store for later retrievel  
"""

import re
import os
import pickle
import loguru
import numpy as np

from tqdm.auto import tqdm
from typing import List, Dict

import sys
sys.path.append('../')
from pyts.transformation import ROCKET 
from pydantic import BaseModel, Field
from typing import List
from abc import abstractmethod
from transform import *
from rocket_rag.node import *


class BaseNodeIndexer():
    """Base node indexer object.
    
    Generic abstract interface for node indexing.
    
    """
    
    nodes: List[Node] = Field(
        default_factory=list,
        description='List of nodes'
    )

    @classmethod
    def load_nodes(self, filename: str) -> List[Node]:
        """
        Load all nodes from a given file (Read the pickle file)

        Args:
            filepath: the filepath to load the node list
        
        Return:
            A node list in the file
        """
    
        loguru.logger.debug(f'Loading all nodes...')
        try:
            with open(filename, 'rb') as f:
                nodes_pkl = pickle.load(f)
            loguru.logger.debug(f'All nodes are loaded.')
            return nodes_pkl
        except FileNotFoundError as e:
            loguru.logger.error(f"Error loading nodes: {e}")
            return []
    
    @classmethod
    def save_nodes(self, nodes: List[Node], filename: str) -> None:
        """
        Save all nodes to a given file (Write the pickle file)

        Args:
            nodes: the node list to be saved
            filename: the filepath to save the node list

        Return:
            None
        """

        if not os.path.exists(filename):
            loguru.logger.debug(f'Creating a new file at {filename}...')
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        loguru.logger.debug(f'Saving all nodes...')
        try:
            with open(filename, 'wb') as f:
                pickle.dump(nodes, f)
            loguru.logger.debug(f'All nodes are stored.')
        except Exception as e:
            raise FileNotFoundError(e)
    
    @classmethod
    def concatenate_nodes(self, node_names: List[Node], filename: str=None) -> None:
        """
        Concatenate node lists 
        Load node lists via filenames and then concatenate all list together follew by saving

        Args:
            node_names: A list of file names of nodes
            filename: The file name to save the concatenated nodes
        
        Return:
            Save the new list to the given file name
        """

        concatenated_nodes = []
        for nf in node_names:
            node_pkl = self.load_nodes(nf)
            concatenated_nodes.extend(node_pkl)
        loguru.logger.debug(f'Concatenated nodes DONE!')

        filename = '../' if filename is None else filename
        self.save_nodes(concatenated_nodes, filename) 
        loguru.logger.debug(f'Concatenated nodes are saved to {filename}')


class TextNodeIndexer(BaseNodeIndexer):
    pass


class ImageNodeIndexer(BaseNodeIndexer):
    pass


class TimeSeriesNodeIndexer(BaseNodeIndexer):
    
    def __init__(self, nodes_filename=None) -> None:
        super().__init__()
        self.nodes_filename = nodes_filename
    
    def indexing(self, 
                 ts: List[np.ndarray], 
                 ids: List[str],
                 meta_info: Dict[str, str]=None) -> List[Node]:
        """
        Node indexing after Bag-of-Words (BoWs) transformation
        A list of BoWs will be embedded using sparse and dense text embeddings
        The result will be represented in nodes and stored for later retrievel

        Args:
            ts: the numpy array of time series data points
            ids: A list of string for marking and naming nodes, this can be a a list of filename
            meta_info: A dictionary {str, str} to demonstrate some info of the node such as the load information

        Return:
            A list of nodes including BoWs string, dense and sparse embeddings
        """
        
        loguru.logger.debug(f'Indexing nodes...')
        assert len(ts) == len(ids), 'The number of time series data and ids should be the same.'
        
        nodes = []
        for i in tqdm(range(len(ts))):
            nodes.append(Node(id_=ids[i],
                              rocket=get_rocket(ts[i]),
                              fft=get_fft(ts[i]),
                              ApEn=get_ApEn(ts[i]),
                              extra_info=meta_info))
        loguru.logger.debug(f'Nodes indexing DONE.')
        
        self.nodes = nodes
        
        # Save the nodes
        self.save_nodes(self.nodes, self.nodes_filename)


if __name__ == '__main__':
    loguru.logger.debug(f'Testing on nodes indexing...')
    
    INSTANCES_DIR = '../data/instances/'
    INFERENCE_DIR = '../data/inference/'
    STATES = ['normal', 
              'backlash1', 'backlash2',
              'lackLubrication1', 'lackLubrication2',
              'spalling1', 'spalling2', 'spalling3', 'spalling4', 'spalling5', 'spalling6', 'spalling7', 'spalling8']
    LOADS= ['20kg', '40kg', '-40kg']
    
    load_num = 20
    load = '20kg'
    ids = [os.listdir(os.path.join(INSTANCES_DIR, load, state)) for state in STATES]
    ids = [filename for sublist in ids for filename in sublist]
    
    ts = []
    for f in ids:
        state = re.match(fr'(.*)_{load_num}', f).group(1)
        temp_ts_df = pd.read_csv(os.path.join(INSTANCES_DIR, load, state, f))
        ts.append(smoothing(ts_df=temp_ts_df, field='current'))
    
    ts_node_indexer = TimeSeriesNodeIndexer(nodes_filename=f'../store/ts_indexing/current_nodes_{load}.pkl')
    ts_node_indexer.indexing(ts=ts, ids=ids, meta_info={'load': load})

    loguru.logger.debug(f'Test Successfully.')
