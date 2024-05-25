import os
import sys
import json
import loguru

from openai import OpenAI, AsyncClient, OpenAIError
from typing import List, Dict

from node_indexing import NodeIndexer
from vector_store import VectorStore
from prompts import fault_diagnosis_prompt, multi_queries_gen_prompt
from utils import fit_transform


VS_DIR = '../store/'
CONFIG_DIR = '../config/'
CONFIG_FILE = 'config.json'
with open(os.path.join(CONFIG_DIR, CONFIG_FILE), 'r') as f:
    config = json.load(f)
    

class RagAgent:
    """ The RAG agent class. """

    def __init__(self, vs_dir:str):
        """ Initialize the RAG agent. """
        self.vs = VectorStore()
        self.ni = NodeIndexer()
        self.nodes = self.ni.load_node_indexing(vs_dir)
        self.vs.add(self.nodes)
        assert len(self.nodes) == 0, f'No docs in the vector store!'
        self.ts_rocket = None
        self.query_res = None
        
        self.fault_diagnosis_statement = None
        
        self._init_openai_client()

    def _init_openai_client(self):
        """ Initialize the OpenAI client. """
        
        if os.environ.get('OPENAI_API_KEY') or config["openai_api_key"]:
            self.openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', config["openai_api_key"]))
            self.async_openai_client = AsyncClient(api_key=os.environ.get('OPENAI_API_KEY', config["openai_api_key"]))
            loguru.logger.info("OpenAI API key found. Initialized OpenAI client.")
        else:
            raise ValueError("OpenAI API key not found. \
                             Please set the OPENAI_API_KEY environment variable in variable env or config file.")
    
    def get_rocket_feature(self, ts_file:str):
        """ Get the rocket feature from the given time series file. 
        
        Args:
            ts_file (str): The path to the time series file.
        """
        
        self.ts_rocket = fit_transform([ts_file],
                                       field='current',
                                       smooth=True,
                                       smooth_ws=config["cur_smooth_ws"],
                                       tolist=False,
                                       verbo=True)
        return self.ts_rocket
    
    def get_fault_prediction(self, mode:str='ridge'):
        """ Get the fault prediction from the given time series file. 
        
        Args:
            mode (str, optional): The mode to query the vector database for get the prediction. Defaults to 'ridge query'.
        """
        
        if mode not in config["query_mode"]:
            raise ValueError("Invalid mode. Please use 'ridge' or 'knn'.")
        
        if mode == 'ridge':
            self.query_res = self.vs.ridge_query(self.ts_rocket)
        elif mode == 'knn':
            self.query_res = self.vs.knn_query(self.ts_rocket)
        return self.query_res
    
    def generate_response(self, 
                          prompts:List[Dict[str, str]], 
                          temperature:float=0.7, 
                          ac:bool=False, 
                          json:bool=False,
                          stream:bool=False):
        """ Generate a response from the OpenAI API. 
        
        Args:
            prompt (str): The prompt to generate a response for.
            temperature (float, optional): The temperature to use for the generation. Defaults to 0.7.
            ac (bool, optional): Whether to use async mode or not. Defaults to False.
            json (bool, optional): Whether to return the response as JSON or not. Defaults to False.
            stream (bool, optional): Whether to stream the response or not. Defaults to False.
        
        Returns:
            OpenAI chat completion object: The generated response object.
        """
        
        client = self.openai_client if not ac else self.async_openai_client
        response_format = "json_object" if json else "text"
        
        try:
            response = client.chat.completions.create(
                model=config['gpt_model'],
                messages=prompts,
                response_format=response_format,
                temperature=temperature,
                stream=stream
            )
        except OpenAIError as e:
            raise e
        
        return response
    
    def generate_fault_diagnosis_statement(self) -> str:
        """ Generate a fault diagnosis statement from the given prompts.
        
        Args:
        
        Returns:
            str: The generated fault diagnosis statement.
        """
        
        if self.query_res is None:
            raise ValueError("Please run the get_fault_prediction method to get the query reuslt first.")
        
        ids, sco = self.query_res
        sys_prompt = fault_diagnosis_prompt.sys_prompt
        user_prompt = fault_diagnosis_prompt.user_prompt.format(res=str(ids[0], score=str(sco[0])))
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.generate_response(messages, temperature=0.1, ac=False, stream=False)
        
        self.fault_diagnosis_statement = response.choices[0].message.content
        return self.fault_diagnosis_statement
    
    def generate_multi_queries(self) -> str:
        """ Generate a fault diagnosis statement from the given prompts.
        
        Args:
        
        Returns:
            str: The generated fault diagnosis statement.
        """
        
        if self.fault_diagnosis_statement is None:
            raise ValueError("Please run the generate_fault_diagnosis_statement method to get the fault diagnosis statement first.")
        
        