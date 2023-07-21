import tarfile, os, yaml, argparse, platform, logging
import boto3

def get_logger():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(name)s [%(levelname)s] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
        )
    logger = logging.getLogger("Glitter")
    logger.setLevel(logging.DEBUG)
    return logger

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

def get_secrets(config):
    # if platform.system()=='Windows' and os.name=='nt':
        
    with open(config['secrets_file_loc'], "r") as stream:
        try:
            secrets = yaml.safe_load(stream)
            return secrets
        except yaml.YAMLError as exc:
            print(exc)
    # else:
    #     return None

def set_token(secrets, api_token=None):
    if secrets:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets['HUGGINGFACEHUB_API_TOKEN']
    else:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

def print_message(message, st=None, logger=None, print_message=True):
    if st:
        st.write(message)
    if logger:
        logger.info(message)
    if print_message and not st and not logger:
        print(message)    


def make_tarfile(output_filename, source_dir, logger=None):
    dir_path = os.path.dirname(output_filename)
    if not os.path.exists(dir_path):
        logger.info("creating directory %s" % dir_path)
        os.makedirs(dir_path)
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    #     files = tar.get_members()
    # for i in files:
    #     print_message(i, logger=logger)

def get_aws_info(secrets):
    s3_client = boto3.client('s3')
    sagemaker_client = boto3.client('sagemaker')
    sagemaker_runtime_client = boto3.client('sagemaker-runtime')
    sess = boto3.Session()
    sagemaker_session_bucket = secrets['s3_bucket']
    role = secrets['role_name']
    
    return {"s3_client": s3_client, 
            "sagemaker_client": sagemaker_client, 
            "sagemaker_runtime_client": sagemaker_runtime_client,
            "session": sess, 
            "sagemaker_session_bucket": sagemaker_session_bucket, 
            "role": role}