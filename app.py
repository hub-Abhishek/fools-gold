# Milton's Opus
import os 
import streamlit as st
from src.utils import get_args, load_config, set_token, print_message, get_logger, get_secrets
from src.doc_utils import process_file
from src.text_preprocessing import get_chunks
from src.chatbot import bot
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

    initial_content = "Hello 👋! Please input a file to start chatting!"
    frontend.write_new_query(initial_content, "assistant", add_to_queue=False)

    sidebar_data = frontend.generate_sidebar()
    query = prompt_container.chat_input(placeholder="Your message", key="query", disabled=True)
    
    frontend.process_messages_for_frontend()
    frontend.write_new_query(query, "user")

    set_token(secrets, sidebar_data['api_token'])
    st.session_state['prompting'] = False if 'prompting' not in st.session_state else st.session_state['prompting']

    if ((len(sidebar_data['uploaded_file'])!=0 )
        # and os.environ["HUGGINGFACEHUB_API_TOKEN"] 
        # and (st.session_state.ready or st.session_state.prompting)
        # and query is not None
        ):
        
        st.session_state['prompting'] = True
        documents, _ = process_file(sidebar_data['uploaded_file'])
        
        chunks = get_chunks(documents)

        chatbot = bot(secrets=secrets, config=config, bot_config=sidebar_data, local=False)
        chatbot.initialize()
        
        query = prompt_container.chat_input(placeholder="Memento mori!", disabled=False)  

        if query:  
            frontend.write_new_query(query, "user")
            answer = chatbot.process_and_predict(query)
            frontend.write_new_query(answer, "assistant")
        # message = 'Ready to chat!'
        # print_message(message, st=assistant)
        # query = prompt_container.text_input('Enter your query here!', value="query", max_chars=500, key="query", type="default", help='Enter your query here!', )

        # if query:
        #     chatbot.process_query(query)
                
        #     result = qa({"query": query})

            # st.write(result['result'])
        #     with open('app_database/result.txt', 'a', encoding="utf-8") as f:           
        #         f.write(str(result))
        #         f.write('/n/n')
        #         f.close()
        #     print_message(chatbot.db.get(), st)
