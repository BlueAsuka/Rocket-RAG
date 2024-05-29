import os
import re
import sys
import json
import loguru
import asyncio
import datetime

from openai import OpenAI, AsyncClient, OpenAIError
from typing import List, Dict
from tqdm.auto import tqdm

from prompts import parse_output_prompt
from utils import fit_transform
from node_indexing import NodeIndexer
from vector_store import VectorStore
from tools import Tools
from prompts import (fault_diagnosis_prompt, 
                     multi_queries_gen_prompt, 
                     refinement_prompt,
                     tool_usage_prompt,
                     text_summarization_prompt,
                     parse_output_prompt)

VS_DIR = '../store/'
LOG_DIR = '../logs/'
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
        self.query_mode = ""
        self.ts_rocket = None
        self.query_res = None

        # The following attributes are used to store the intermedium results of the RAG agent.
        self.fault_diagnosis_json: Dict[str, str] = None
        self.refined_diagnosis_json: Dict[str, str] = None
        self.generated_queries: List[str] = None
        self.query_answers: List[str] = None
        self.text_summarization: List[str] = None

        self.memory = [] # The memory for the dialog history
        self.tools = Tools() # The tools for calling 
        
        self._init_openai_client() # Initialize the OpenAI client

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
        self.query_mode = mode
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
                tools=self.tools.info,
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
        
        idx, vals = self.query_res
        sys_prompt = fault_diagnosis_prompt.sys_prompt
        user_prompt = fault_diagnosis_prompt.ridge_prompt.format(res=idx, score=vals) if self.query_mode == 'ridge' else \
                      fault_diagnosis_prompt.knn_prompt.format(res=idx, distances=vals)
 
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        self.memory = messages
        
        response = self.generate_response(messages, temperature=0.1, ac=False, stream=False)
        resp_json_str = response.choices[0].message.content
        self.memory.append({"role": "assistant", "content": resp_json_str})
        self.fault_diagnosis_json = json.loads(resp_json_str)
        return self.fault_diagnosis_json
    
    def refine_fault_diagnosis_statement(self):
        """ Refine the fault diagnosis statement based on the user's feedback.
        
        Args:
        
        Returns:
            str: The refined fault diagnosis statement.
        """
        
        if self.fault_diagnosis_json is None:
            raise ValueError("Please run the generate_fault_diagnosis_statement method to get the fault diagnosis statement first.")
        
        sys_prompt = refinement_prompt.sys_promot
        user_prompt = refinement_prompt.user_prompt.format(ft=self.fault_diagnosis_json['fault_type'],
                                                           dl=self.fault_diagnosis_json['degradation_level'])
        
        refinement_prompt = [
            {"role": "system", "content": sys_prompt},  
            {"role": "user", "content": user_prompt}
        ]
        self.memory += refinement_prompt
            
        response = self.generate_response(self.memory, temperature=0.1, ac=False,  json=True,stream=False)
        refined_json_str = response.choices[0].message.content
        self.memory.append({"role": "assistant", "content": refined_json_str})
        self.refined_diagnosis_json = json.loads(refined_json_str)
        return self.refined_diagnosis_json
        
    def formalize_query(query: str):
        """Preprocess the query for the vector store query
        
        Remove some symbols including '-', '"', '.' and indexing numbers or patterns like 1. 2. 3. ...
        """
        query = query.strip().replace('"', '').replace('. ', '')
        pattern = re.compile(r'[-0-9]+|\d+\. ')
        result = pattern.sub('', query)
        return result.strip()
    
    def generate_multi_queries(self, num_queries: int=5, refined: bool=True) -> str:
        """ Generate a fault diagnosis statement from the given prompts.
        
        Args:
            num_queries (int, optional): The number of queries to generate. Defaults to 5.
            refined (bool, optional): Whether to use the refined fault diagnosis statement. Defaults to True.
        
        Returns:
            str: The generated fault diagnosis statement.
        """
        
        if self.fault_diagnosis_json is None or self.refined_diagnosis_json is None:
            raise ValueError("Please run the generate_fault_diagnosis_statement method to get the fault diagnosis statement first.")
        
        if not isinstance(self.fault_diagnosis_json, dict):
            raise ValueError("The fault diagnosis statement is not a JSON object.")
        
        if not isinstance(self.refined_diagnosis_json, dict):
            raise ValueError("The refined diagnosis statement is not a JSON object.")
        
        fault_type = self.refined_diagnosis_json['refinement'] if refined else self.fault_diagnosis_json['fault_type']
        fault_description = self.fault_diagnosis_json['description']
        sys_prompt = multi_queries_gen_prompt.sys_prompt
        user_prompt = multi_queries_gen_prompt.user_prompt.format(fault_type=fault_type, 
                                                                  fault_description=fault_description,
                                                                  num=str(num_queries))
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        self.memory += messages
        
        response = self.generate_response(messages, temperature=0.1, ac=False, stream=False)
        multi_queries_gen = response.choices[0].message.content
        self.generated_queries = [self.formalize_query(query) for query in multi_queries_gen.split('\n')]
        return self.generated_queries
    
    def call_google_search(self, query: List[str]):
        """Call the Google search API to get the search results.
        
        Args:
            query (a list of str): The list of query string.
            
        Returns:
            list: A list of search results.
        """

        num_query = len(query)
        for i, q in enumerate(query):
            loguru.logger.info(f'Calling Google search API for query {i+1}/{num_query}')

            call_google_messages = [
                {"role": "system", "content": tool_usage_prompt.call_google},
                {"role": "user", "content": tool_usage_prompt.call_google_user_input.format(q=q)}
            ]

            # Whether to store the call_google_messages and searching result in the memory is still under discussion.
            self.memory += call_google_messages

            call_google_response = self.generate_response(call_google_messages, 
                                                          temperature=0.1, 
                                                          ac=False, 
                                                          stream=False)
            call_google_messages.append({"role": "assistant", "content": call_google_response.choices[0].message.content})
            tool_calls = call_google_response.choices[0].message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    # Call the function 
                    function_name = tool_call.function.name
                    function_to_call = self.tools.available_tools[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(**function_args)

                    # Add the tool call result and feed back to the GPT
                    call_google_messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )

            # Get the final result after processing
            get_searching_response = self.generate_response(call_google_messages,
                                                            temperature=0.1,  
                                                            ac=False,
                                                            stream=False)
            self.query_answers.append(get_searching_response.choices[0].message.content)
        return self.query_answers
    
    async def gather_query_answers(self, num_children: int=3, verbose: bool=False):
        """ Gather the query answers from the Google search API. 
        
        Args:
            num_children (int): The number of children to be generated. Defaults to 3.
            verbose (bool): Whether to print the progress. Defaults to False.


        Returns:
            list: A list of query answers.
        """

        if not self.query_answers:
            raise ValueError('No answer is needed to be combined.')
        
        # Get the system prompt for the text summarization
        text_summarization_sys_prompt = text_summarization_prompt.sys_prompt

        # Parse the prompt message for each node including several answers 
        node_batch_prompts = []
        for idx in range(0, len(self.query_answers), num_children):
            # only looks at num_children (in default 3) answers at once
            node_batch = self.query_answers[idx: idx+num_children]
            node_batch_text = "\n\n".join([node for node in node_batch])
            # Parse the prompt for summerization with given answers
            text_summarization_user_prompt = text_summarization_prompt.user_prompt.format(text=node_batch_text)
            temp_prompt = [
                {"role": "system", "content": text_summarization_sys_prompt},
                {"role": "user", "content": text_summarization_user_prompt}
            ]
            node_batch_prompts.append(temp_prompt)
        
        # Use async mode to generate the summerization 
        tasks = [self.generate_response(prompt=p, temperature=0.3, ac=True) for p in node_batch_prompts]
        combined_responses = await asyncio.gather(*tasks)
        self.text_summarization = [r.choices[0].message.content for r in combined_responses]

        if len(self.text_summarization) == 1:
            loguru.logger.info("Combined all responses to one. Done")
            return self.text_summarization[0]
        else:
            loguru.logger.info(f"Combined into {len(self.text_summarization)} responses, keep combining")
        if verbose:
            loguru.logger.info(self.text_summarization)
        return await self.gather_query_answers()

    def parse_output_file(self, file_name: str):
        """ Parse the output file to form and store the report for decision support.

        Args:
            output_file (str): The output file path.
        """

        # Extract all necessary information from the output file
        fault_type = self.fault_diagnosis_json["fault_type"]
        fault_level = self.fault_diagnosis_json["degradation_level"]
        fault_description = self.fault_diagnosis_json["description"]
        fault_text = f"Fault Type Diagnosis: {fault_type} \n\n" + \
                          f"Degradation Level: {fault_level} \n\n" + \
                              f"Fault Description: {fault_description}"

        queries = [str(i+1) + ". " + q for i, q in enumerate(self.query_answers)]
        querys_text = "\n\n".join(queries)

        answers = [str(i+1) + ". " + a for i, a in enumerate(self.text_summarization)]
        answers_text = "\n\n".join(answers)

        rec_text = self.text_summarization[0]

        # Parse the output path
        dt = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(LOG_DIR, (file_name + dt + ".md"))

        parse_output_messages = [
            {"role": "system", "content": parse_output_prompt.sys_prompt},
            {"role": "user", "content": parse_output_prompt.user_prompt.format(fault_text=fault_text, 
                                                                               query_text=querys_text, 
                                                                               answer_text=answers_text, 
                                                                               rec_text=rec_text,
                                                                               file_path=file_path)},
        ]
        response = self.generate_response(prompts=parse_output_messages, temperature=0.1)

        # Check whether the file is parsed and stored successfully
        if os.path.exists(file_path):
            loguru.logger.info(f"File {file_name} is parsed and stored successfully.")
