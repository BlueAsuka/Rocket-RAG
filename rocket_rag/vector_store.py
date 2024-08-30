"""
The data structure for storing vectors constructed using Bag-of-Words and text embedding models
This code is referred to the code release on the following link:
https://docs.llamaindex.ai/en/latest/examples/low_level/vector_store.html
"""

import os
import sys
import random
import loguru
import numpy as np

from typing import List, Any, Dict, Tuple
from tqdm.auto import tqdm

sys.path.append("..")
from rocket_rag.utils import *
from rocket_rag.node import Node
from rocket_rag.node_indexing import NodeIndexer

from pyts.transformation import ROCKET
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier


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

class TextVectorStore(BaseVectorStore):
    pass


class TimeSeriesVectorStore(BaseVectorStore):
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
    
    def get_rocket_features(self):
        """Extract rocket features from the nodes"""
        if len(self.nodes) == 0:
            loguru.logger.error(f'No docs in the vector store! Please add doc.')
        return np.array([node.get_rocket_feature() for node in self.nodes])
    
    def get_doc_ids(self):
        """Extract doc ids from the nodes"""
        if len(self.nodes) == 0:
            loguru.logger.error(f'No docs in the vector store! Please add doc.')
        return np.array([node.id_ for node in self.nodes])
    
    def add(self, nodes: List[Node]) -> List[str]:
        """Add nodes to index"""
        for node in nodes:
            self.node_dict[node.node_id] = node
    
    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes using node_id"""
        del self.node_dict[node_id]

    def knn_query(self, 
                  query: Union[List[Any], np.ndarray],
                  k: int=1,
                  metric: str='euclidean',
                  verbo: bool=False) -> List[Tuple[float, str]]:
        """
        Retrievel by getting the top-k embeddings for the given query based on euclidean distance.

        Args:
            query: the rocket feature of the query (a time series)
            k: the top k similiar for documents retrieveling, default to 1
            metric: the distance metric to use, default to euclidean
            verbo: the boolean flag to show the debugging information
        
        Return:
            A list of tuples including ids of retrieveled document ids and corresponding distances
        """

        if isinstance(query, list):
            query = np.array(query)
        
        if verbo:
            loguru.logger.debug(f'Looking up all docs...')
        rocket_features = self.get_rocket_features()
        doc_ids = self.get_doc_ids()

        # ======================================
        # Handcraft KNN classifier
        # ======================================
        # if verbo:
        #     loguru.logger.debug(f'Calculating sample similarity based on euclidean distance...')
        # euclideans = [[(np.linalg.norm(rocket_features[i] - q), doc_ids[i])
        #                 for i in range(len(rocket_features))]
        #                 for q in query]

        # if verbo:
        #     loguru.logger.debug(f'Sorting the euclidean distances...')
        # topk_retrieve_res = []
        # for e in euclideans:
        #     e_topk = sorted(e, key=lambda x: x[0])[:k]
        #     topk_retrieve_res.append(e_topk)
        # topk_retrieve_res = np.array(topk_retrieve_res)
        
        # if verbo:
        #     loguru.logger.debug(f'Parsing the topk retrieving results...')
        # similarities = [s for s, _ in topk_retrieve_res.squeeze()]
        # result_ids = [r for _, r in topk_retrieve_res.squeeze()]
        
        ids_to_doc = {i: doc_ids[i] for i in range(len(doc_ids))}
        
        if verbo:
            loguru.logger.debug(f'Calculating sample similarity based on {metric} distance...')
        knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
        knn_model.fit(rocket_features, doc_ids)
        distances, ids = knn_model.kneighbors(query, n_neighbors=k, return_distance=True)
        result_ids = [ids_to_doc[i] for i in ids.tolist()[0]]

        return (result_ids, distances.squeeze().tolist())
    
    def ridge_query(self, 
                    query: Union[List[Any], np.ndarray],
                    alpha: float=None,
                    verbo=False):
        """
        Apply ridge classifier for retrieving results, the retrieving result is a prediction of the given qurey.
        Use ridge classifier can only return top-1 retrieval result.
        
        Args:
            query: the rocket feature of the query (a time series)
            alpha: the alpha parameter for ridge classifier
            verbo: the boolean flag to show the debugging information
            
        Return:
            A  tuple including id of retrieveled document and corresponding similarity scores of the ridge model
        """
        
        if isinstance(query, list):
            query = np.array(query)
        
        if verbo:
            loguru.logger.debug(f'Looking up all docs...')
        rocket_features = self.get_rocket_features()
        doc_ids = self.get_doc_ids()
        
        if verbo:
            loguru.logger.debug(f'Finding similar sample using ridge classifier...')
        alpha = 0.5 if alpha is None else alpha
        ridge_model = RidgeClassifier(alpha=alpha)
        ridge_model.fit(rocket_features, doc_ids)
        result_id = ridge_model.predict(query)
        score = ridge_model.score(rocket_features, doc_ids)
        
        return (result_id, score)
        

if __name__ == "__main__":
    loguru.logger.debug(f'Testing on vector store module...')
    load = '20kg'
    node_indexer = NodeIndexer()
    nodes = node_indexer.load_node_indexing(f'../store/nodes_{load}.pkl')

    loguru.logger.debug(f'Initializing vector store...')
    vector_store = TimeSeriesVectorStore()
    vector_store.add(nodes)
    loguru.logger.info(f'Loaded nodes into the vector store.')

    if_files_dict = parse_files(main_directory=INFERENCE_DIR)
    if_ts_files = if_files_dict[load]
    # np.random.seed(42)
    rand_idx = np.random.randint(0, len(if_ts_files))
    if_ts_filename = if_ts_files[rand_idx]

    # Warp the filename string as a list and do the ROCKET transformation
    if_rocket_feature = fit_transform(if_ts_filename,
                                      field='current',
                                      smooth=True,
                                      smooth_ws=15,
                                      tolist=False,
                                      verbo=True)

    loguru.logger.debug(f'Retrieveling...')

    print(f'Retrievel following results using 1-NN for the given time series {if_ts_filename}:')
    ids, dists = vector_store.knn_query(if_rocket_feature, k=1, verbo=True)
    print(f'File: {ids}, distances: {dists}')   
    
    print(f'Retrievel following results using 5-NN for the given time series {if_ts_filename}:')
    ids, dists = vector_store.knn_query(if_rocket_feature, k=5, verbo=True)
    print(f'File: {ids}, distances: {dists}')   
    
    print(f'Retrievel following results using Ridge for the given time series {if_ts_filename}:')
    result_id, score = vector_store.ridge_query(if_rocket_feature, alpha=1.0, verbo=True)
    print(f'File: {result_id}, score: {score}')

    loguru.logger.info('Testing on vector store module DONE!')
