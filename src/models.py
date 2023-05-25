from abc import ABC, abstractmethod
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA

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


    def get_embeddings(self):
        return HuggingFaceInstructEmbeddings(
            query_instruction="Represent the query for retrieval: ", 
            model_name=self.embeddings_model_name
            )

    def get_model(self):
        return HuggingFaceHub(
            repo_id=self.model_name, model_kwargs=self.model_kwargs)

    def get_db(self):
        
        self.db = Chroma.from_documents(self.chunks, self.embeddings, persist_directory='app_database/db', llm=self.llm)
        retriever = self.db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        return qa

    def build_model_and_db(self):

        # model_name='hkunlp/instructor-large'
        self.st.write('Building the embeddings...')
        self.embeddings = self.get_embeddings()

        self.st.write('Building the model...')
        self.llm = self.get_model()

        self.st.write('Building the database...')
        self.qa = self.get_db()

        return self.qa