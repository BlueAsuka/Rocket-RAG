"""
Node indexing for vector store
The node indexer will include the BoWs transformation and vector embedding 
After constructing nodes containing dense and sparse embeddings, nodes will be stored in the vector store for later retrievel  
"""

import os
import pickle
import loguru
import numpy as np

from tqdm.auto import tqdm
from typing import List, Dict

import sys
sys.path.append('../')
from rocket_rag.node import Node
from pyts.transformation import ROCKET 


class BaseIndexer():
    pass


class TextNodeIndexer(BaseIndexer):
    pass


class TimeSeriesNodeIndexer(BaseIndexer):
    
    def indexing(self, 
                 rocket: ROCKET,
                 ts: np.ndarray, 
                 batch_size: int,
                 ids: List[str],
                 meta_info: Dict[str, str]=None) -> List[Node]:
        """
        Node indexing after Bag-of-Words (BoWs) transformation
        A list of BoWs will be embedded using sparse and dense text embeddings
        The result will be represented in nodes and stored for later retrievel

        Args:
            encoders: the encoder object for sparse and dense encoding 
            ts: the numpy array of time series data points
            batch_size: The number for indexing at once
            ids: A list of string for marking and naming nodes, this can be a a list of filename
            meta_info: A dictionary {str, str} to demonstrate some info of the node such as the load information

        Return:
            A list of nodes including BoWs string, dense and sparse embeddings
        """
        
        loguru.logger.debug(f'Indexing nodes...')
        nodes = []
        for i in tqdm(range(0, len(ids), batch_size)):
            i_end = min(i + batch_size, len(ts))
            batch = ts[i:i_end]
            ids_batch = ids[i:i_end]

            # ROCKET transformation
            rocket_features = rocket.fit_transform(batch)

            for i in range(len(batch)):
                nodes.append(Node(id_=ids_batch[i][:-len('.csv')],
                                  rocket_feature=rocket_features[i].tolist(),
                                  extra_info=meta_info))
        loguru.logger.info(f'Nodes indexing DONE.')

        return nodes

    def save_node_indexing(self, nodes: list[Node], filepath: str) -> None:
        """
        Save nodes to a given filepath

        Args:
            nodes: A list of nodes constrcuted after nodes indexing
            filepath: the filepath to save the node list
        
        Return:
            None
        """

        if not os.path.exists(filepath):
            loguru.logger.debug(f'Creating a new file at {filepath}...')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        loguru.logger.debug(f'Saving all nodes...')
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(nodes, f)
            loguru.logger.info(f'All nodes are stored.')
        except Exception as e:
            raise FileNotFoundError(e)

    def load_node_indexing(self, filepath: str) -> list[Node]:
        """
        Load all nodes from a given file (Read the pickle file)

        Args:
            filepath: the filepath to load the node list
        
        Return:
            A node list in the file
        """
    
        loguru.logger.debug(f'Loading all nodes...')
        try:
            with open(filepath, 'rb') as f:
                nodes_pkl = pickle.load(f)
            loguru.logger.info(f'All nodes are loaded.')
            return nodes_pkl
        except FileNotFoundError as e:
            loguru.logger.error(f"Error loading nodes: {e}")

    def concatenate_nodes(self, node_names: list[Node], save_path: str=None) -> None:
        """
        Concatenate node lists 
        Load node lists via filenames and then concatenate all list together follew by saving

        Args:
            node_names: A list of file names of nodes
            save_path: the path to save the new node list
        
        Return:
            Save the new list to the given file name
        """

        concatenated_nodes = []
        for nf in node_names:
            node_pkl = self.load_node_indexing(nf)
            concatenated_nodes.extend(node_pkl)
        loguru.logger.debug(f'Concatenated nodes DONE!')

        save_path = '../' if save_path is None else save_path
        self.save_node_indexing(concatenated_nodes, save_path) 
        loguru.logger.info(f'Concatenated nodes are saved to {save_path}')


if __name__ == '__main__':
    # loguru.logger.debug(f'Testing on nodes indexing...')

    # load = '-40kg'
    # load_state_dict = {load: [os.listdir(os.path.join(INSTANCES_DIR, load, state)) for state in STATES] for load in LOADS}
    # ids = [item for sublist in load_state_dict[load] for item in sublist]

    # files_dict = parse_files(main_directory=INSTANCES_DIR)
    # ts_files = files_dict[load]
    
    # ts = np.array([fit(ts_filename=f,
    #                    field='current',
    #                    smooth=True,
    #                    smooth_ws=15,
    #                    tolist=False) for f in ts_files])
    
    # node_indexer = TextNodeIndexer()
    # rocket = ROCKET(n_kernels=10000, kernel_sizes=([9]), random_state=42)
    # nodes = node_indexer.indexing(rocket=rocket,      
    #                               ts=ts,
    #                               batch_size=40,
    #                               ids=ids,
    #                               meta_info={'load': load})
    
    # print(np.array(nodes[5].get_rocket_feature()))

    # node_indexer.save_node_indexing(nodes, f'../store/nodes_{load}.pkl')

    # node_indexer.load_node_indexing(f'../store/nodes_{load}.pkl')

    # print()
    # if_files_dict = parse_files(main_directory=INFERENCE_DIR)
    # if_ts_files = if_files_dict[load]
    # np.random.seed(42)
    # rand_idx = np.random.randint(0, len(if_ts_files))
    # if_ts_filename = if_ts_files[rand_idx]
    # if_rocket_feature = fit_transform([if_ts_filename],
    #                                   field='current',
    #                                   smooth=True,
    #                                   smooth_ws=15,
    #                                   tolist=False,
    #                                   verbo=False)
    # print(f'ROCKET transformation on randomly selected inference sample:')
    # print(np.array(if_rocket_feature))

    # loguru.logger.info(f'Test Successfully.')

    pass