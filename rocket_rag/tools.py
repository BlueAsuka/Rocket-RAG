"""
This is the Google Searching module used by GPT.
This module is used to perform Google searches and return the results.
1. function to perform Google search
2. json to describe function declaration
3. a dictionary to store the function name and its json
"""

import os
import json
from googleapiclient.discovery import build


CONFIG_DIR = "../config"
CONFIG_FILE = "config.json"
with open(os.path.join(CONFIG_DIR, CONFIG_FILE), "r") as f:
    config = json.load(f)

class Tools():
    """ This is the Tools class used by GPT. """

    def __init__(self) -> None:

        self.available_tools = None
        self.tools = [
        {
            "type": "function",
            "function": {
                "name": "call_google",
                "description": "Call the google chrome web browser to search online based on a given query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query string for searching online",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    def call_google(self, query: str, **kwargs):
        """ Call the google chrome for searching online 
        
        Args:
            query (str): The query string for searching online
            **kwargs: Additional keyword arguments to pass to the google search API
            
        Returns:
            str: The results using the google search API
        """
        
        service = build(serviceName="customsearch", 
                        version="v1", 
                        developerKey=config["google_api_key"],
                        static_discovery=False)
        res = service.cse().list(q=query, cx=config["google_cse_id"], **kwargs).execute()
        res_items = res["items"]
        res_snippets = [r['snippet'] for r in res_items]
        return str(res_snippets)
    
    def get_available_tools(self):
        """ Get the tools available for GPT """

        if len(self.tools) == 0:
            return {"": None}
        self.available_tools = {n:f for n, f in Tools.__dict__.items() if not n.startswith("_") and callable(f)}
        return self.available_tools
