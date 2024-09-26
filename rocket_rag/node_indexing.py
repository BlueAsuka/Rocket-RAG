"""
Node indexing for vector store
The node indexer will include the BoWs transformation and vector embedding 
After constructing nodes containing dense and sparse embeddings, nodes will be stored in the vector store for later retrievel  
"""

import os
import sys
import pickle
import loguru
import numpy as np

from tqdm.auto import tqdm
from typing import List, Dict
from abc import abstractmethod

sys.path.append('../')
from rocket_rag.node import *
from rocket_rag.transform import *


class BaseNodeIndexer():
    """Base node indexer object.
    
    Generic abstract interface for node indexing.
    
    """
    
    nodes: List[Node] = Field(
        default_factory=list,
        description='List of nodes'
    )
    filename: str = Field(
        default=None,
        description='The file name to save the nodes'
    )

    def __init__(self, nodes: List[Node]=None, filename: str=None) -> None:
        self.nodes = nodes
        self.filename = filename

    @abstractmethod
    def index(self) -> None:
        """Index nodes."""
        pass

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
    def save_nodes(self, nodes: List[Node]=None, filename: str=None) -> None:
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
            with open(filename, 'wb') as file:
                pickle.dump(nodes, file)
            loguru.logger.debug(f'All nodes are stored.')
        except Exception as e:
            raise FileNotFoundError(e)
    
    @classmethod
    def concatenate_nodes(self, nodes_name: List[Node], save_filename: str=None) -> None:
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
        for nf in nodes_name:
            node_pkl = self.load_nodes(nf)
            concatenated_nodes.extend(node_pkl)
        loguru.logger.debug(f'Concatenated nodes DONE!')

        filename = '../' if save_filename is None else save_filename
        self.save_nodes(concatenated_nodes, filename) 
        loguru.logger.debug(f'Concatenated nodes are saved to {filename}')


class TextNodeIndexer(BaseNodeIndexer):
    
    def index(self,
              txt: List[str], 
              txt_transform: TextTransform,
              doc_ids: List[int],
              meta_info: Dict[str, str]=None) -> List[Node]:
        """
        Node indexing after Bag-of-Words (BoWs) transformation
        A list of dense text embeddings
        The result will be represented in nodes and stored for later retrievel

        Args:
            txt: the list of text data points
            txt_transform: The text transform class to get embedding of the text
            ids: The list of ids for each text chunk, can be page numbers or paragraph numbers
            meta_info: A dictionary {str, str} to demonstrate some info of the node such as the load information

        Return:
            A list of nodes including strings and dense embeddings
        """

        assert len(txt) == len(doc_ids), 'The number of text data and ids should be the same.'

        loguru.logger.debug(f'Text Embedding...')
        embeds = []
        for i in tqdm(range(len(txt))):
            embeds.append(txt_transform.get_embedding_from_openai(txt[i]).squeeze().tolist())
        assert len(txt) == len(embeds), 'The number of text data and embeddings should be the same.'
        
        loguru.logger.debug(f'Text Node Indexing...')
        nodes = []
        for i in tqdm(range(len(txt))):
            id = str(doc_ids[i]) if not isinstance(doc_ids[i], str) else doc_ids[i]
            nodes.append(TextNode(id_=id,
                                  text=txt[i], 
                                  embedding=embeds[i], 
                                  extra_info=meta_info))
        loguru.logger.debug(f'Text Node Indexing DONE!')
        
        self.nodes = nodes
        
        # Save the nodes
        # self.save_nodes(self.nodes, self.filename)
        
        return nodes

class ImageNodeIndexer(BaseNodeIndexer):
    pass


class TimeSeriesNodeIndexer(BaseNodeIndexer):
    
    def index(self, 
              ts: List[np.ndarray], 
              ts_transform: TimeSeriesTransform,
              labels: List[str],
              meta_info: Dict[str, str]=None) -> List[Node]:
        """
        Node indexing after Bag-of-Words (BoWs) transformation
        A list of BoWs will be embedded using sparse and dense text embeddings
        The result will be represented in nodes and stored for later retrievel

        Args:
            ts: the numpy array of time series data points
            ts_transform: The time series transform class to get different attributes of the time series
            labels: A list of string for marking and naming nodes, used for id of the node
            meta_info: A dictionary {str, str} to demonstrate some info of the node such as the load information

        Return:
            A list of nodes including BoWs string, dense and sparse embeddings
        """
        
        assert len(ts) == len(labels), 'The number of time series data and labels should be the same.'

        loguru.logger.debug(f'Time Series Nodes Indexing...')
        
        nodes = []
        for i in tqdm(range(len(ts))):
            nodes.append(TimeSeriesNode(id_=labels[i],
                                        rocket=ts_transform.get_rocket(ts[i]),
                                        fft=ts_transform.get_fft(ts[i]),
                                        ApEn=ts_transform.get_ApEn(ts[i]),
                                        extra_info=meta_info))
        loguru.logger.debug(f'Time Series Nodes Indexing DONE.')
        
        self.nodes = nodes
        
        # Save the nodes
        # self.save_nodes(self.nodes, self.filename)
        
        return self.nodes


if __name__ == '__main__':
    # cfg_path = os.path.join(
    #     os.path.abspath(Path(os.path.dirname(__file__)).parent.absolute()),
    #     "config/configs.json"
    #     )
    # cfg = json.load(open(cfg_path))
    
    # INSTANCES_DIR = '../data/instances/'
    # INFERENCE_DIR = '../data/inference/'
    # STATES = ['normal', 
    #           'backlash1', 'backlash2',
    #           'lackLubrication1', 'lackLubrication2',
    #           'spalling1', 'spalling2', 'spalling3', 'spalling4', 'spalling5', 'spalling6', 'spalling7', 'spalling8']
    # LOADS= ['20kg', '40kg', '-40kg']
    
    # loguru.logger.debug(f'Testing on time series nodes indexing...')
    
    # load_num = 20
    # load = '20kg'
    # ids = [os.listdir(os.path.join(INSTANCES_DIR, load, state)) for state in STATES]
    # ids = [filename for sublist in ids for filename in sublist]
    
    # ts_transform = TimeSeriesTransform(cfg=cfg)
    
    # ts = []
    # for f in ids:
    #     state = re.match(fr'(.*)_{load_num}', f).group(1)
    #     temp_ts_df = pd.read_csv(os.path.join(INSTANCES_DIR, load, state, f))
    #     ts.append(ts_transform.smoothing(ts_df=temp_ts_df, field='current'))
    
    # ts_node_indexer = TimeSeriesNodeIndexer(ts_transform=ts_transform, 
    #                                         nodes_filename=f'../store/ts_indexing/current_nodes_{load}.pkl')
    # ts_node_indexer.indexing(ts=ts, ids=ids, meta_info={'load': load})

    # loguru.logger.debug(f'Test Successfully.')
    pass
