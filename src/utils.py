import platform
import argparse
import os
import yaml

def get_args():
    parser = argparse.ArgumentParser(description='DocBot')
    parser.add_argument('--config_loc', type=str, default='./config.yaml', help='Location of the config file')
    return parser.parse_args()

def load_config(config_loc):
    with open(config_loc, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def set_token(config, api_token):
    if platform.system()=='Windows' and os.name=='nt':
        with open(config['secrets_file_loc'], "r") as stream:
            try:
                secrets = yaml.safe_load(stream)
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets['HUGGINGFACEHUB_API_TOKEN']
            except yaml.YAMLError as exc:
                print(exc)
    else:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

def get_files(st):
    uploaded_file = st.file_uploader(label='Upload your files here!', 
                                 type=['pdf', 'txt'], 
                                 accept_multiple_files=True, 
                                 key='uploaded_file', 
                                 help='Upload the files you want to chat with. You can drag and drop, or browse through your files by clicking inside the upload box.', )
    return uploaded_file

def print_message(message, st=None, logger=None, print_message=True):
    if st:
        st.write(message)
    if logger:
        logger.info(message)
    if print_message and not st and not logger:
        print(message)    
