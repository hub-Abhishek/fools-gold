from abc import ABC, abstractmethod
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from src.utils import print_message
from src.text_preprocessing import get_chunks

class Chatbot(ABC):
    def __init__(self, chunks, model_repo, model_type, embeddings_model_name, model_name, st, new_files, model_kwargs, run_local) -> None:
        super().__init__()
        self.chunks = chunks
        self.model_repo = model_repo
        self.model_type = model_type
        self.embeddings_model_name = embeddings_model_name
        self.model_name = model_name
        self.st = st
        self.new_files = new_files
        self.model_kwargs = model_kwargs
        self.run_local = run_local
        self.query_ids = get_query_ids()


    def build_embeddings(self):
        return HuggingFaceInstructEmbeddings(
            query_instruction="Represent the query for retrieval: ", 
            model_name=self.embeddings_model_name
            )

    def build_model(self):
        return HuggingFaceHub(
            repo_id=self.model_name, model_kwargs=self.model_kwargs)

    # def get_db(self):
        
    #     self.db = Chroma.from_documents(self.chunks, self.embeddings, persist_directory='app_database/db', llm=self.llm)
    #     retriever = self.db.as_retriever()
    #     qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    #     return qa

    def build_db(self):
        
        db = Chroma.from_documents(self.chunks, self.embeddings, persist_directory='app_database/db', llm=self.llm)
        # retriever = self.db.as_retriever()
        # qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        return db
    
    def build_qa(self):
        retriever = self.db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        return qa

    def build_model_and_db(self):

        # model_name='hkunlp/instructor-large'
        if self.st:
            self.st.write('Building the embeddings...')
        else:
            print('Building the embeddings...')
        self.embeddings = self.build_embeddings()

        if self.st:
            self.st.write('Building the model...')
        else:
            print('Building the model...')
        self.llm = self.build_model()

        if self.st:
            self.st.write('Building the database...')
        else:
            print('Building the database...')
        # self.qa = self.get_db()
        self.db = self.build_db()

        if self.st:
            self.st.write('Building the QA model...')
        else:
            print('Building the QA model...')
        self.qa = self.build_qa() 

        return self.qa
    
    def get_query_ids(self):
        with open('app_database/query_ids.txt','r') as f:
            ids = f.readlines()
            f.close()
        ids = [id.strip() for id in ids]
        return ids
    
    def delete_ids_from_db(self, ids):
        self.db._collection.delete(ids=ids)
        
    
    def process_query(self, query):
        self.query_ids = self.get_query_ids()
        existing_queries = self.db._collection.query.get(ids=self.query_ids)['documents']
        if query in existing_queries:
            print_message(f'Query already exists in the database. Returning the existing query id.', st=self.st)
        else:
            print_message(f'Query does not exist in the database. Adding the query to the database...', st=self.st)
            self.delete_ids_from_db(ids=self.query_ids)
            existing_queries = Document(page_content='  '.join(existing_queries) + '  ' + query,  
                                        metadata={'title': 'query(ies)', 'query_number': 1}) # TODO - add query number


            chunks = get_chunks(existing_queries, new_files=False)
            self.query_ids = self.db.add_documents(chunks)
            with open('app_database/query_ids.txt','w') as f:
                f.write(self.query_id)
