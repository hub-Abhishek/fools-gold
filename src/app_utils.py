import os
from PIL import Image
import streamlit as st
from src.utils import print_message


class frontend_manager():
    def __init__(self, st, logger, app_config, prompt_container, config) -> None:
        self.st = st
        self.logger = logger
        self.app_config = app_config
        self.prompt_container = prompt_container
        self.config = config

        self.avtar_width = 40
        
        self.get_or_create_key_session_state('bot_avtar_pic')
        self.get_or_create_key_session_state('your_avtar_pic')
        self.get_or_create_key_session_state('last_state')
        self.get_or_create_key_session_state('file_names', [])


    def generate_sidebar(self):

        with self.st.sidebar:

            # if "ready" not in self.st.session_state:
            #     self.st.session_state.ready = None
            message = 'Welcome to DocBot!\n Chatting is always better than stalking! Chat with your documents to find the answers you need!'
            self.st.sidebar.title(message)

            uploaded_file, model_repo, embeddings_model_name, model_name = self.get_options_v2()
            # self.other_options()
            self.st.markdown("""---""")             
            
            file_box = self.st.container()
            self.list_files(file_box, uploaded_file)
            self.st.markdown("""---""") 


            col1, col2= st.columns(2)
            with col1:
                ready = self.st.button('Build the bot!', key='ready')
            with col2:
                self.st.button('Reset memory!', key='Reset', on_click=self.reset_bot)
            self.st.markdown("""---""") 
            # self.st.markdown("""---""") 
            
            self.st.write(self.st.session_state)
            
            
        return {'uploaded_file': uploaded_file,
                'model_repo': model_repo, 
                'embeddings_model_name': embeddings_model_name, 
                'model_name': model_name, 
                'api_token': None, 
                'run_local': None, 
                'model_kwargs': None,
                'ready': self.st.session_state.ready
                }
    
    def list_files(self, file_box, uploaded_file):
            self.checkboxes = []
            file_box.write('Files found:')
            for i, file in enumerate(uploaded_file):
                self.checkboxes.append(file_box.checkbox(file.replace(f"{self.config['uploaded_file_loc']}\\", ''), key=f'flie_{i}'))
            delete_button_enabled = True if sum(self.checkboxes)==0 else False
            file_box.button('Delete selected files!', on_click=self.delete_files, args=[uploaded_file], disabled=delete_button_enabled, key='delete_files', help='Delete the selected files!')
        
    def delete_files(self, uploaded_file):
        for i, file in enumerate(self.checkboxes):
            if file:
                if os.path.exists(uploaded_file[i]):
                    os.remove(uploaded_file[i])
                # self.st.session_state.file_names.remove(uploaded_file[i])
        

    def display_avtar(self, container, location, n_images): 
        cols = container.columns(n_images)
        for dir, _, files in os.walk(location):
            for i, image in enumerate(files):
                image_file = get_avtar_pic(dir, image)
                cols[i].image(image_file, use_column_width=True)

    def get_or_create_key_session_state(self, name, init_value=None):
        if not name in self.st.session_state:
            self.st.session_state[name] = init_value
    
    # @st.cache
    def other_options(self):

        bot_avtar, your_avtar = self.st.tabs(["Bot Avatars", "Your Avtars"])

        with bot_avtar:
            self.display_avtar(bot_avtar, self.config["bot_avatar_folder"], 3)

        with your_avtar:
            self.display_avtar(your_avtar, self.config["your_avatar_folder"], 3)

    def get_options_v2(self):
        step_1, step_2, Info = self.st.tabs(["Step 1", "Step 2", "Status"])
        uploaded_file = []
        doc_info = ''

        with step_1:
            self.upload_files()
            uploaded_file = self.get_files()
            # if len(uploaded_file) > 0:
            #     if len(uploaded_file) == 1:
            #         doc_info = [file.name for file in uploaded_file]
            #         doc_info = ', '.join(doc_info)
            #     elif len(uploaded_file) > 2:
            #         doc_info = f"{uploaded_file[0].name} and {len(uploaded_file)-1} other files"

        with step_2:
            model_repo, embeddings_model_name, model_name = self.get_model_info()

        with Info:
            self.st.write('Status')
            self.st.write(f"Using {model_repo} - {embeddings_model_name} for embeddings and {model_name} for retrieval")# on documents - {doc_info}")

        return uploaded_file, model_repo, embeddings_model_name, model_name

    def upload_files(self):
        upload_files = False
        if self.st.session_state.last_state != 'uploading':
            upload_files = self.st.button('Upload files?', )

        if upload_files or self.st.session_state.last_state == 'uploading':
            self.st.session_state.last_state = 'uploading'
            uploaded_file = self.st.file_uploader(label='Upload your files here!', 
                                    type=['pdf', 'txt', 'csv'], 
                                    accept_multiple_files=True, 
                                    key='uploaded_file', 
                                    help='Upload the files you want to chat with. You can drag and drop, or browse through your files by clicking inside the upload box.', )
            
            for file in uploaded_file:
                if not os.path.exists(os.path.join(self.config['uploaded_file_loc'], file.name)):
                    with open(os.path.join(self.config['uploaded_file_loc'], file.name),"wb") as f: 
                        f.write(file.getbuffer())  
                os.walk(self.config['uploaded_file_loc'])
            
            # del st.session_state['uploaded_file']

    def get_files(self):
        uploaded_files = []
        self.st.session_state.file_names = []
        for dir, _, files in os.walk('uploaded_files'):
            for file in files:
                uploaded_files.append(os.path.join(dir, file))
                if file not in self.st.session_state.file_names:
                    self.st.session_state.file_names.append(file)
        return uploaded_files


    def get_model_info(self):
        model_repo = self.st.selectbox('Where are your models hosted?',
                                (self.app_config['model_repo']), 
                                key="model_repo", 
                                help='Enter your model provider here!')
        
        embeddings_model_name = self.st.selectbox('Which embeddings model do you want to try out?',
                                self.app_config['embeddings_model_name'], 
                                key="embeddings_model_name", 
                                help='Enter your embeddings model name here!')
        
        model_name = self.st.selectbox('Which model do you want to try out?',
                                self.app_config['model_name'], 
                                key="model_name", 
                                help='Enter your model name here!')

        # api_token = st.text_input('API TOKEN', value="api_token", max_chars=500, key="api_token", type="default", help='Enter your API token here!',)
        
        # run_local = st.checkbox('Run locally', value=False, key='run_local', help='Run the models locally? The data and models will be stored in session state, and will be deleted after the execition.')

        # model_kwargs = st.text_input('Model kwargs', value="model_kwargs", max_chars=500, key="model_kwargs", type="default", help='Enter your model kwargs here!',)

        return model_repo, embeddings_model_name, model_name

    def process_messages_for_frontend(self):
        
        if "messages" not in self.st.session_state:
            self.st.session_state.messages = []
        
        for message in self.st.session_state.messages:
            avtar = self.st.session_state.your_avtar_pic \
                if message["role"] == "user" else self.st.session_state.bot_avtar_pic
            self.st.chat_message(message["role"], avatar=avtar).write(message["content"])

    def write_new_query(self, content, role, add_to_queue=True):
        
        avtar = self.st.session_state.your_avtar_pic \
            if role == "user" else self.st.session_state.bot_avtar_pic
        
        if content:
            self.st.chat_message(role, avatar=avtar).write(content)

        if add_to_queue and content:
            self.st.session_state.messages.append({"role": role, "content": content})

        if "query" in self.st.session_state:
            del self.st.session_state["query"]
        
        self.st.session_state.last_state = 'query'
        

    def handle_queries(self, query, sidebar_data):
        if (len(sidebar_data['uploaded_file'])==0 ) and query:
            self.handle_queries_without_files(query)

    def handle_queries_without_files(self, query):
            return_query = "You haven't finished the set up process! Please upload a file to start chatting!"
            self.process_messages_for_frontend(return_query, "assistant")


    def reset_bot(self):
        del self.st.session_state["messages"]
        del self.st.session_state["file_names"]
        self.st.session_state["prompting"] = False
        self.st.session_state["last_state"] = None
        self.prompt_container.empty()
        self.delete_files()


@st.cache_data# (allow_output_mutation = True)
def get_avtar_pic(dir, image):
    image_file = Image.open(os.path.join(dir, image))
    return image_file