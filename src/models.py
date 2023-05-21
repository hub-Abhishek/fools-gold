from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA


def get_embeddings(model_name):
    return HuggingFaceInstructEmbeddings(
        query_instruction="Represent the query for retrieval: ", 
        model_name=model_name
        )

def get_model(model_name):
    return HuggingFaceHub(
        repo_id=model_name, model_kwargs={"temperature":0.5, "max_length":500})

def get_db(chunks, embeddings, llm):
    
    db = Chroma.from_documents(chunks, embeddings, persist_directory='app_database/db', llm=llm)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def build_model_and_db(chunks, model_repo, model_type, model_name='hkunlp/instructor-large', st=None, new_files=False):

    st.write('Building the embeddings...')
    embeddings = get_embeddings(model_name)

    st.write('Building the model...')
    llm = get_model(model_name)

    st.write('Building the database...')
    qa = get_db(chunks, embeddings, llm)

    return qa