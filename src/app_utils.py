def generate_sidebar(st):
    with st.sidebar:
        st.sidebar.title("Options")
        model_repo = st.selectbox('Where are your models hosted?',
                                ('huggingface',), 
                                key="model_repo", 
                                help='Enter your model provider here!')
        
        embeddings_model_name = st.selectbox('Which embeddings model do you want to try out?',
                                ('google/flan-t5-base', 'google/flan-t5-small', 'hkunlp/instructor-large', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'other'), 
                                key="embeddings_model_name", 
                                help='Enter your embeddings model name here!')
        
        embeddings_model_name_text = st.text_input('Embeddings model name if not available in dropdown list', 
                                        value=embeddings_model_name if embeddings_model_name != 'other' else '',
                                        # max_chars=500, 
                                        key="embeddings_model_name_text", 
                                        type="default", 
                                        help='Enter your custom huggingface model name here!', 
                                        label_visibility='visible')
        
        if embeddings_model_name_text != '' and embeddings_model_name_text != 'other' and embeddings_model_name == 'other':
            embeddings_model_name = embeddings_model_name_text
        
        model_name = st.selectbox('Which model do you want to try out?',
                                ('google/flan-t5-base', 'google/flan-t5-small', 'hkunlp/instructor-large', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'other'), 
                                key="model_name", 
                                help='Enter your model name here!')
        
        model_name_text = st.text_input('Model name if not available in dropdown list', 
                                        value=model_name if model_name != 'other' else '',
                                        # max_chars=500, 
                                        key="model_name_text", 
                                        type="default", 
                                        help='Enter your custom huggingface model name here!', 
                                        label_visibility='visible')
        
        if model_name_text != '' and model_name_text != 'other' and model_name == 'other':
            model_name = model_name_text
        
        
        model_kwargs = eval(st.text_input('Additional kwargs for the model', 
                                    value="{'temperature':0.5, 'max_length':500}", 
                                    max_chars=500, 
                                    key="model_kwargs", 
                                    type="default", 
                                    help='Enter your model kwargs here!',))

        api_token = st.text_input('API TOKEN', value="api_token", max_chars=500, key="api_token", type="default", help='Enter your API token here!',)
        
        run_local = st.checkbox('Run locally', value=False, key='run_local', help='Run the models locally? The data and models will be stored in session state, and will be deleted after the execition.')

    return model_repo, embeddings_model_name, model_name, model_kwargs, api_token, run_local