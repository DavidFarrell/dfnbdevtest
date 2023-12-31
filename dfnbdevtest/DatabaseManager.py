# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/pinecone_database_manager.ipynb.

# %% auto 0
__all__ = ['DatabaseManager']

# %% ../nbs/pinecone_database_manager.ipynb 3
import json
from bs4 import BeautifulSoup
import re
#import pymilvus
#from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility

import pinecone
from datasets import load_dataset, Dataset
import unidecode

#for embeddings
import os
import openai
import time # to step back in case of rate limit
from tqdm.auto import tqdm # for progress bars
import tiktoken

from IPython.display import Markdown

'''


%pip install -qU \
  tiktoken==0.4.0 \
  openai==0.27.7 \
  langchain==0.0.179 \
  pinecone-client==2.2.1 \
  datasets==2.13.1 \
  cohere
'''

# %% ../nbs/pinecone_database_manager.ipynb 5
class DatabaseManager:
    def __init__(self, config):
        """
        Initialize the DatabaseManager with configuration settings.
        :param config: Configuration details for database connection and other settings.
        """
        self.config = config
        # Initialize database connection here



    def chunk_json(self, json_data):
        """
        Divide the JSON data into manageable chunks.
        :param json_data: The parsed JSON data.
        :return: List of chunks.
        """
        # Implement chunking logic here
        chunks = []
        return chunks

    def embed_chunks(self, chunks):
        """
        Create embeddings for each chunk of data.
        :param chunks: List of data chunks.
        :return: List of embedded chunks.
        """
        # Implement embedding logic here
        embedded_chunks = []
        return embedded_chunks

    def initialize_database(self):
        """
        Set up the Milvus database, including connection and schema.
        """
        # Implement database initialization here

    def store_in_milvus(self, embedded_chunks):
        """
        Store embedded chunks in the Milvus database.
        :param embedded_chunks: List of embedded chunks.
        """
        # Implement storage logic here

    def search_database(self, query, k):
        """
        Search the database for K nearest chunks based on the query embedding.
        :param query: Search query.
        :param k: Number of nearest chunks to find.
        :return: Search results.
        """
        # Implement search logic here

    def retrieve_data(self, search_results):
        """
        Fetch chunk data and metadata based on search results.
        :param search_results: Results from the database search.
        :return: Corresponding data and metadata.
        """
        # Implement data retrieval logic here

    def __del__(self):
        """
        Cleanup when an instance is destroyed, like closing database connections.
        """
        # Implement cleanup logic here
    
    def ingest_json(self, file_path):
        """
        Read and parse a JSON file.
        :param file_path: Path to the JSON file.
        :return: Parsed JSON data.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data


