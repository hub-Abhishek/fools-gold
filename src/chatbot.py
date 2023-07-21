from abc import ABC, abstractmethod

import chromadb
from chromadb.config import Settings
import requests, json
import uuid

# import langchain
# from langchain.cache import InMemoryCache
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from langchain.docstore.document import Document

# from src.utils import print_message, get_aws_info, get_logger
# from src.text_preprocessing import get_chunks    
# from src.resource_manager import Embeddings, Models

class bot():
    def __init__(self, config, secrets, bot_config, local=False) -> None:
        self.config = config
        self.secrets = secrets
        self.bot_config = bot_config
        self.local = local

        self.embedding_function = None
        self.memory = None
        self.retriever = None
        self.qa = None

        self.query_id = None
        self.save_chat_temporarily_to_db = False

    def initialize(self):
        self.get_memory()
        self.get_embedding_func()
        self.get_db()
        self.get_retriever()
        self.get_model()
        self.get_qa()

    def get_memory(self):
        if self.memory:
            return self.memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # return self.memory

    def get_embedding_func(self):
        if self.local:
            embeddings_model_name = self.bot_config['model_name'] # 'google/flan-t5-base'
            self.embeddings_function = HuggingFaceInstructEmbeddings(
                            query_instruction="Represent the query for retrieval: ", 
                            model_name=embeddings_model_name
                            )
            # return embeddings_function
        else:
            from src.bot_utils import embeddings_function
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
        self.chromadb_client.get_or_create_collection('my_collection', embedding_function=self.embedding_function)

        self.collection = self.chromadb_client.get_collection('my_collection', embedding_function=self.embedding_function)

    def embed_documents_into_db(self, chunks):
        ids = [str(uuid.uuid1()) for _ in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        documents = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings_function.embed_documents(documents)

        self.collection.add(ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents)

    def get_retriever(self):
        if self.retriever:
            return self.retriever
        self.langchain_chromadbdb = Chroma(client=self.chromadb_client, embedding_function=self.embeddings_function, collection_name=self.config['collection_name'])
        self.retriever = self.langchain_chromadbdb.as_retriever()
        # return self.retriever
    
    def get_model(self):
        if self.local:
            from langchain.llms import HuggingFaceHub
            self.chat_model = HuggingFaceHub(repo_id="google/flan-t5-base", huggingfacehub_api_token=self.secrets['HUGGINGFACEHUB_API_TOKEN'])
            # return self.chat_model
        else:
            from src.bot_utils import CustomLLM
            self.chat_model = CustomLLM(url=f"{self.secrets['model']['url']}")
    
    def get_qa(self):
        from langchain.chains import RetrievalQA, ConversationalRetrievalChain
        self.qa = ConversationalRetrievalChain.from_llm(self.chat_model, self.retriever, memory=self.memory)
        # return self.qa
    
    def process_and_predict(self, query):
        if self.qa is None:
            self.get_qa()
        if self.save_chat_temporarily_to_db:
            pass
        else:
            self.response = self.qa({"question": query})
            print(self.response)
            return self.response['answer']





# chromadb.heartbeat()
# db = Chroma(client=chromadb, embedding_function=self.embeddings, collection_name=self.config['collection_name'])

# model_name = 'google/flan-t5-base'
# model =  HuggingFaceHub(
#                 repo_id=model_name)




# def build_qa(self):
#     if self.run_local:
#         print_message('Building the QA model...', st=self.st)
#         retriever = self.db.as_retriever()
#         qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
#         return qa

# def build_model_and_db(self):
#     if self.run_local:
    
#         self.embeddings = self.build_embeddings()        
#         self.llm = self.build_model()
#         self.db = self.build_db()
#         self.qa = self.build_qa()
#         return self.qa

# def get_query_ids(self):
#     if self.run_local:
#         with open('app_database/query_ids.txt','r') as f:
#             ids = f.readlines()
#             f.close()
#         ids = [id.strip() for id in ids]
#         return ids

# def delete_ids_from_db(self, ids):
#     if self.run_local:
#         self.db._collection.delete(ids=ids)
    

# def process_query(self, query):
    
#     if self.run_local:
#         self.query_ids = self.get_query_ids()
#         existing_queries = self.db._collection.get(ids=self.query_ids)['documents']
#         print_message(f'Existing queries: {existing_queries}', st=self.st)
#         if query in existing_queries:
#             print_message(f'Query already exists in the database. Returning the existing query id.', st=self.st)
#         else:
#             print_message(f'Query does not exist in the database. Adding the query to the database...', st=self.st)
#             self.delete_ids_from_db(ids=self.query_ids)
#             existing_queries = [Document(page_content='  '.join(existing_queries) + '  ' + query,  
#                                         metadata={'title': 'query(ies)', 'query_number': 1})] # TODO - add query number


#             chunks = get_chunks(existing_queries, new_files=False)
#             self.query_ids = self.db.add_documents(chunks)
#             self.db.persist()
#             print_message(f'Query added to the database. Query id: {self.query_ids}, Query: {existing_queries}', st=self.st)
#             with open('app_database/query_ids.txt','w') as f:
#                 f.write('\n'.join(self.query_ids))
