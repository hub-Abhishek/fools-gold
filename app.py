import os 
import streamlit as st
import json
from src.doc_utils import process_file
from src.text_preprocessing import get_chunks
from src.models import Chatbot
from src.app_utils import generate_sidebar

st.set_page_config(page_title='DocBot', layout = 'wide', initial_sidebar_state = 'auto')

st.write('Welcome to DocBot!')
st.write('Here, you can chat with your documents and get answers to your questions.')
st.write(st.session_state)
uploaded_file = st.file_uploader(label='Upload your files here!', 
                                 type=['pdf', 'txt'], 
                                 accept_multiple_files=True, 
                                 key='uploaded_file', 
                                 help='Upload the files you want to chat with. You can drag and drop, or browse through your files by clicking inside the upload box.', )

model_repo, embeddings_model_name, model_name, model_kwargs, api_token, run_local = generate_sidebar(st)
st.write(f'Using {model_repo} - {model_name} for retrieval')

os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

if len(uploaded_file)!=0 and api_token != 'api_token':
    documents, new_files = process_file(uploaded_file, st)
    
    chunks = get_chunks(documents, new_files)

    chatbot = Chatbot(chunks=chunks, model_repo=model_repo, model_type=None, 
                      embeddings_model_name=embeddings_model_name, model_name=model_name, 
                      st=st, model_kwargs=model_kwargs, new_files=new_files, run_local=run_local)
    qa = chatbot.build_model_and_db()
    
    st.write('Ready to chat!')
    query = st.text_input('Enter your query here!', value="query", max_chars=500, key="query", type="default", help='Enter your query here!', )

    if query is not None:
        result = qa({"query": query})
        st.write(result['result'])
        with open('app_database/result.txt', 'a', encoding="utf-8") as f:           
            f.write(str(result))
            f.write('/n/n')
            f.close()
