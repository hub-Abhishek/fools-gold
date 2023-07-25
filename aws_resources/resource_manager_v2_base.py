import sys
sys.path.append(r'D:\New folder\project\miniprojects.me\fools-gold')
from abc import ABC, abstractmethod
from typing import List
from src.utils import get_logger, print_message, load_config, set_token, get_secrets, make_tarfile, get_aws_info

import sagemaker, boto3

class Resources(ABC):
    """Interface for aws resources."""

    @abstractmethod
    def get_model_name(self):
        """Get model name."""
        return NotImplementedError

    @abstractmethod
    def check_model(self):
        """Check if model exists in ecosystem."""
        raise NotImplementedError

    @abstractmethod
    def deploy_model(self):
        """Deploy model."""
        raise NotImplementedError


class Models(Resources):
    
    def __init__(self, names, config, secrets=None, aws_info=None):
        self.config = config
        self.secrets = secrets
        self.names = names
        self.aws_info = aws_info

        self.model_name = None
        self.model = None


    def get_model_name(self, names):
        """Get model name."""
        if self.model_name:
            return self.model_name
        
        model_name = ''
        for name in names:
            if model_name != '':
                model_name += '--'
            model_name += name.replace("/", "-")
        self.model_name = model_name
        return model_name
    
    
    def check_model(self):
        """Check if model exists in ecosystem."""
        if self.model:
            return True
        if not self.model_name:
            self.get_model_name(self.names)
        endpoints = aws_info['sagemaker_client'].list_endpoints()
        for endpoint in endpoints['Endpoints']:
            if self.model_name == endpoint['EndpointName']:
                return True
        return False    

    
def get_tokenizer(tokenizer_name, config, secrets=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer

def get_model(model_name, config, secrets=None):
    # from transformers import AutoModelForSeq2SeqLM
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    import sentence_transformers
    model = sentence_transformers.SentenceTransformer(model_name_or_path="google/flan-t5-base")
    return model

def get_model_name(names, config=None, secrets=None):
    model_name = ''
    for name in names:
        if model_name != '':
            model_name += '--'
        model_name += name.replace("/", "-")
    return model_name
    
if __name__=="__main__":
    # embeddings = Embeddings()
    # model = Models()
    
    tokenizer_name = 'google/flan-t5-base'
    model_name = 'llama-2-7b-chat'
    model_save_name = get_model_name([tokenizer_name, model_name])

    logger = get_logger()
    config = load_config('config.yaml')
    secrets = get_secrets(config)
    aws_info = get_aws_info(secrets)

    set_token(secrets, api_token=None)
    print_message(f"config: {config}", logger=logger)

    local_s3_folder = f"{config['local_s3_folder']}/{model_save_name}/resources" # 'aws_resources
    tar_file_source_folder = config['tar_folder'] # 'tar_files'
    tokenizer_folder = config['tokenizer_folder'] # 'tokenizers'
    model_folder = config['model_folder'] # 'core_models'
    models_source_dir = config['models_source_dir'] # 'models'

    pretrained_loc = f'{local_s3_folder}/{models_source_dir}' # 'aws_resources/models'


    # tokenizer = get_tokenizer(tokenizer_name, config)
    model = get_model(model_name, config)

    model.save(f'{pretrained_loc}/{model_folder}/{model_name}/')
    logger.info(f"model saved to {pretrained_loc}/{model_folder}/{model_name}/")

    local_location = f'{tar_file_source_folder}/{model_save_name}/model.tar.gz' # 'tar_files/google-flan-t5-base--google-flan-t5-base/model.tar.gz'

    make_tarfile(local_location, f'{local_s3_folder}/', logger=logger)
    logger.info(f"tar file saved to {local_location}")

    response = aws_info['s3_client'].upload_file(local_location, secrets['s3_bucket'], f'{config["s3_models_source_dir"]}/{model_save_name}/model.tar.gz')
    logger.info(f"tar file uploaded to {config['s3_models_source_dir']}/{model_save_name}/model.tar.gz")

    from sagemaker.pytorch import PyTorchModel
    pytorch_model = PyTorchModel(model_data=f's3://{secrets["s3_bucket"]}/{config["s3_models_source_dir"]}/{model_save_name}/model.tar.gz', 
                             role=secrets['role_name'], 
                            #  source_dir='aws_resources/model_resources',
                            #  entry_point='model_resources/code/inference.py', 
                             framework_version='2.0', 
                             py_version='py310', 
                             name=f'{model_save_name}',
                            #  endpoint_name=f'{model_save_name}'
                             )
    
    predictor = pytorch_model.deploy(instance_type='ml.m5.2xlarge', initial_instance_count=1, endpoint_name=f'{model_save_name}')
    # import pdb; pdb.set_trace()
    print_message(response, logger=logger)

    