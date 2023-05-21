import os 
import streamlit as st
import json
from src.doc_utils import process_file
from src.text_preprocessing import get_chunks
from src.models import build_model_and_db

st.set_page_config(page_title='DocBot', layout = 'wide', initial_sidebar_state = 'auto') # page_icon=None
# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)

st.write('Welcome to DocBot!')
st.write('Here, you can chat with your documents and get answers to your questions.')
# st.write('Please upload your document below.')

uploaded_file = st.file_uploader(label='Upload your files here!', 
                 type=['pdf', 'txt'], 
                 accept_multiple_files=True, 
                 key='uploaded_file', 
                 help='Upload the files you want to chat with. You can drag and drop, or browse through your files by clicking inside the upload box.', 
                 )

model_repo = st.text_input('MODEL REPO', value="huggingface", max_chars=500, key="model_repo", type="default", help='Enter your model repo here!',)
# model_type = st.text_input('MODEL TYPE', value="t5", max_chars=500, key="model_type", type="default", help='Enter your model type here!',)
model_name = st.text_input('MODEL NAME', value="google/flan-t5-base", max_chars=500, key="model_name", type="default", help='Enter your model name here!',)
model_kwargs = st.text_input('MODEL KWARGS', value="{'temperature':0.5, 'max_length':500}", max_chars=500, key="model_kwargs", type="default", help='Enter your model kwargs here!',)

st.write(f'using {model_repo} for retrieval')
api_token = st.text_input('API TOKEN', value="api_token", max_chars=500, key="api_token", type="default", help='Enter your API token here!',)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

if len(uploaded_file)!=0 is not None and api_token != 'api_token':
    documents, new_files = process_file(uploaded_file, st)
    
    chunks = get_chunks(documents, new_files)

    qa = build_model_and_db(chunks, model_repo, model_type=None, model_name=model_name, st=st, new_files=new_files)
    
    st.write('Ready to chat!')
    query = st.text_input('Enter your query here!', value="query", max_chars=500, key="query", type="default", help='Enter your query here!', )

    if query is not None:
        result = qa({"query": query})
        st.write(result)
        with open('app_database/result.txt', 'a') as f:           
            f.write(str(result))
            f.write('/n/n')
            f.close()
