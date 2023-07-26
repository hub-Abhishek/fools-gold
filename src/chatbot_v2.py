import os
import uuid
import chromadb
import requests, json
import pandas as pd
import replicate

from abc import ABC, abstractmethod
from chromadb.config import Settings
from src.bot_utils_v2 import embeddings_function# , CustomLLM
from src.doc_utils import process_file_get_chunks
from src.constants import prompt_temp, qa_template, system_message

class bot():
    def __init__(self, config, secrets, bot_config, local=False, logger=None) -> None:
        self.config = config
        self.secrets = secrets
        self.bot_config = bot_config
        self.local = local
        self.logger = logger

        self.embedding_function = None

        self.query_id = None
        self.save_chat_temporarily_to_db = False

    def initialize(self):
        self.get_mappings_db()
        self.get_embedding_func()
        self.get_db()
        # self.get_model()
        
    def get_embedding_func(self):
        self.embeddings_function = embeddings_function()


    def get_db(self):
        if self.embedding_function is None:
            self.embedding_function = self.get_embedding_func()
        
        settings = Settings(chroma_api_impl="rest",
                            # TODO: replace api with url
                            chroma_server_host=self.secrets['public_chroma_db']['api'],
                            chroma_server_http_port=self.secrets['public_chroma_db']['port_number'])
        self.chromadb_client = chromadb.Client(settings)
        
        # Testing if the connection is working
        self.chromadb_client.heartbeat()
        self.collection = self.chromadb_client.get_or_create_collection(self.config['collection_name'], embedding_function=self.embedding_function)

    def embed_documents_into_db(self, chunks):
        
        ids = [str(uuid.uuid1()) for _ in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        documents = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings_function.embed_documents(documents)

        self.collection.add(ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents)
        return ids
    
    # def get_model(self):
        # from src.bot_utils import CustomLLM
        # self.chat_model = CustomLLM(url=f"{self.secrets['model']['url']}")

    def chat(self, query, documents):
        self.logger.info("waiting for response from llm")
        documents = '\n'.join(list(set(documents)))
        message = qa_template.format(question=query, context=documents)
        response = replicate.run(self.config['model_replicate'],
                                 input = {
                                     "prompt": message,
                                     "system_prompt": system_message,
                                     "max_new_tokens": 500,
                                     "temperature:": 0.1,

                                     })
        return response
    
    def process_and_predict(self, query):
        
        # 1. get embeddings of query
        # 2. match embeddings with db
        # 3. get documents from db
        # 4. get response from llm
        
        query_embeddings = self.embeddings_function.embed_query(query)
        response = self.collection.query(query_embeddings, )
        response = self.chat(query=query, documents=response['documents'][0]).replace(self.config['EOL'], '')
        self.logger.info(response)
                # ids = self.embed_documents_into_db(chunks)
                # self.add_ids_to_db(ids, file)
        return response
        
        
    def process_file(self, uploaded_file):
        for file in uploaded_file:
            existing_info = self.collection.get(where={'file_name': file})
            if len(existing_info['ids']) == 0:
                chunks = process_file_get_chunks(file)
                ids = self.embed_documents_into_db(chunks)
                self.add_ids_to_db(ids, file)


    def add_ids_to_db(self, ids, file):
        db_new = pd.DataFrame(columns=['file_name', 'ids'])
        db_new['file_name'] = [file for _ in ids]
        db_new['ids'] = ids
        self.db = pd.concat([self.db, db_new]).reset_index(drop=True)
        self.db.to_csv(self.config['db_csv_folder'], index=False)

        
    def get_mappings_db(self):
        if os.path.exists(self.config['db_csv_folder']):
            self.db = pd.read_csv(self.config['db_csv_folder'])
        else:
            self.db = pd.DataFrame(columns=['file_name', 'ids'])


        
            