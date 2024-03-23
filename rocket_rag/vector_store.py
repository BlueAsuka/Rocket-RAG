"""
The data structure for storing vectors constructed using Bag-of-Words and text embedding models
This code is referred to the code release on the following link:
https://docs.llamaindex.ai/en/latest/examples/low_level/vector_store.html
"""

import os
import random
import loguru
import numpy as np

from typing import List, Any, Dict, Tuple
from tqdm.auto import tqdm

import sys

from yaml import load_all
sys.path.append("..")
from rocket_rag.utils import *
from rocket_rag.node import Node
from rocket_rag.node_indexing import NodeIndexer

from pyts.transformation import ROCKET


class BaseVectorStore():
    """Simple custom Vector Store.

    Stores documents in a simple in-memory dict.

    """

    stores_text: bool = True

    def get(self, text_id: str) -> List[float]:
        """Get vector for a text ID."""
        pass

    def add(self, nodes: List[Node]) -> List[str]:
        """Add nodes to index"""
        pass

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes using node_id."""
        pass

    def query(self, query: str, **kwargs: Any):
        """Get nodes for response"""
        pass    


class VectorStore(BaseVectorStore):
    """An updated version of above SimpleVectorStore"""

    stores_text: bool = True

    def __init__(self) -> None:
        """Init params"""
        self.node_dict: Dict[str, Node] = {}
        self.nodes = self.node_dict.values()
        super().__init__()

    def get(self, text_id: str) -> List[float]:
        """Get vector for a text ID."""
        return self.node_dict[text_id]
    
    def add(self, nodes: List[Node]) -> List[str]:
        """Add nodes to index"""
        for node in nodes:
            self.node_dict[node.node_id] = node
    
    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes using node_id"""
        del self.node_dict[node_id]

    def knn_query(self, 
              query: Union[List[Any], np.ndarray],
            #   docs: List[List[float]],
            #   doc_ids: List[str],
              k: int=5,
              verbo: bool=False) -> List[Tuple[float, str]]:
        """
        Retrievel by getting the top-k embeddings for the given query based on euclidean distance.

        Args:
            query: the rocket feature of the query (a time series)
            k: the top k similiar for documents retrieveling
        
        Return:
            A list of tuples including ids of retrieveled document and corresponding similarity scores
        """

        if isinstance(query, list):
            query = np.array(query)
        
        if verbo:
            loguru.logger.debug(f'Looking up all docs...')
        if len(self.nodes) == 0:
            loguru.logger.error(f'No docs in the vector store! Please add doc.')
            return ([], [])
        rocket_features = np.array([node.get_rocket_feature() for node in self.nodes])
        doc_ids = np.array([node.id_ for node in self.nodes])

        if verbo:
            loguru.logger.debug(f'Calculating sample similarity based on euclidean distance...')
        euclideans = [
            [(np.linalg.norm(rocket_features[i] - q), doc_ids[i])
                for i in range(len(rocket_features))]
                for q in query
            ]
        
        # TODO: Support for difference distance metrics such as cosine similarity
        # cosine = []

        if verbo:
            loguru.logger.debug(f'Sorting the euclidean distances...')
        topk_retrieve_res = []
        for e in euclideans:
            e_topk = sorted(e, key=lambda x: x[0])[:k]
            topk_retrieve_res.append(e_topk)
        topk_retrieve_res = np.array(topk_retrieve_res)
        
        if verbo:
            loguru.logger.debug(f'Parsing the topk retrieving results...')
        similarities = [s for s, _ in topk_retrieve_res.squeeze()]
        result_ids = [r for _, r in topk_retrieve_res.squeeze()]

        return (similarities, result_ids)
    
    def ridge_prediction(self, query, verbo=False):
        """
        Apply ridge classifier with sklearn package for the retrieving results
        the retrieving result is a prediction of the given qurey.
        """
        pass
        # TODO: Implement the ridge classifier for the prediction.


if __name__ == "__main__":
    loguru.logger.debug(f'Testing on vector store module...')
    load = '20kg'
    node_indexer = NodeIndexer()
    nodes = node_indexer.load_node_indexing(f'../store/nodes_{load}.pkl')

    loguru.logger.debug(f'Initializing vector store...')
    vector_store = VectorStore()
    vector_store.add(nodes)
    loguru.logger.info(f'Loaded nodes into the vector store.')

    if_files_dict = parse_files(main_directory=INFERENCE_DIR)
    if_ts_files = if_files_dict[load]
    # np.random.seed(42)
    rand_idx = np.random.randint(0, len(if_ts_files))
    if_ts_filename = if_ts_files[rand_idx]

    # Warp the filename string as a list and do the ROCKET transformation
    if_rocket_feature = fit_transform([if_ts_filename],
                                      field='current',
                                      smooth=True,
                                      smooth_ws=15,
                                      tolist=False,
                                      verbo=True)

    loguru.logger.debug(f'Retrieveling...')
    s, ids = vector_store.knn_query(if_rocket_feature, k=5, verbo=True)

    print(f'Retrievel following results for the given time series {if_ts_filename}:')
    for i in range(len(s)):
        print(f'File: {ids[i]}, score: {s[i]}')    

    loguru.logger.info('Testing on vector store module DONE!')
