# Milton's Opus
import os 
import streamlit as st
from src.utils import get_args, load_config, set_token, print_message, get_logger, get_secrets
# from src.doc_utils import process_file
# from src.text_preprocessing import get_chunks
from src.chatbot_v3 import bot
from src.app_utils import frontend_manager

if __name__ == '__main__':
    args = get_args()
    config = load_config(args.config_loc)
    secrets = get_secrets(config)
    logger = get_logger()

    st.set_page_config(page_title='DocBot', layout = 'wide', initial_sidebar_state = 'auto')

    prompt_container = st.container()
    prompt_container.empty()
    
    frontend = frontend_manager(st, logger, app_config=config['app_config'], prompt_container=prompt_container, config=config)

    initial_content = "Hello 👋! Please open the sidebar and follow the instructions to start chatting! Your first query might take a minute, please bear with us while we fix this issue."
    frontend.write_new_query(initial_content, "assistant", add_to_queue=False)

    sidebar_data = frontend.generate_sidebar()
    query = prompt_container.chat_input(placeholder="Your message", key="query", disabled=True)
    
    frontend.process_messages_for_frontend()
    frontend.write_new_query(query, "user")
    
    set_token(secrets, sidebar_data['api_token'], sidebar_data['replicate_api_token'])
    st.session_state['prompting'] = False if 'prompting' not in st.session_state else st.session_state['prompting']

    if ((len(sidebar_data['uploaded_file'])!=0 )
        and os.environ["REPLICATE_API_TOKEN"]!=''
        # and os.environ["HUGGINGFACEHUB_API_TOKEN"] 
        # and (st.session_state.ready or st.session_state.prompting)
        # and query is not None
        ):
        
        st.session_state['prompting'] = True

        chatbot = bot(secrets=secrets, config=config, bot_config=sidebar_data, local=False, logger=logger)
        chatbot.initialize()
        
        chatbot.process_file(sidebar_data['uploaded_file'])
        
        query = prompt_container.chat_input(placeholder="Memento mori!", disabled=False)  

        if query:  
            frontend.write_new_query(query, "user")
            answer = chatbot.process_and_predict(query)
            frontend.write_new_query(answer, "assistant")
