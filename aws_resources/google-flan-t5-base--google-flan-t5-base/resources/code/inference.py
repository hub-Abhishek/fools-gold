# filename: inference.py
import os
import json
import torch
import numpy as np
from six import BytesIO, StringIO
import logging
from sagemaker_inference import content_types, decoder
import sentence_transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def get_logger():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(name)s [%(levelname)s] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
        )
    logger = logging.getLogger("Glitter")
    logger.setLevel(logging.DEBUG)
    return logger

def model_fn(model_dir): 
    name = 'model_fn'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger()
    logger.info(f"{name} - Loading the model from {model_dir} to {device}")
    dir_list = os.listdir(model_dir)
    logger.info(dir_list)

    for path,subdir,files in os.walk(model_dir):
        for name in subdir:
            logger.info( os.path.join(path,name)) # will print path of directories
        for name in files:    
            logger.info( os.path.join(path,name)) # will print path of files

    # auto_model = AutoModel.from_pretrained("../aws_resources/models/core_models/google/flan-t5-base")
    # auto_tokenizer = AutoTokenizer.from_pretrained("../aws_resources/models/tokenizers/google/flan-t5-base")

    tokenizer = AutoTokenizer.from_pretrained("models/core_models/google/flan-t5-base")
    seq2seqmodel = AutoModelForSeq2SeqLM.from_pretrained('models/core_models/google/flan-t5-base')
    model = sentence_transformers.SentenceTransformer(model_name_or_path='models/core_models/google/flan-t5-base')
    model.to(device).eval()
    logger.info(f'{name} - Done loading model')
    return {'tokenizer': tokenizer, 'seq2seqmodel': seq2seqmodel, 'model': model}


def input_fn(input_data, content_type, context):
    """A default input_fn that can handle JSON, CSV and NPZ formats.
        
    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type

    Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
    """
    name = 'input_fn'
    logger = get_logger()
    logger.info(f'{name} - Deserializing the input data.')
    logger.info(f'{name} - content_type: {content_type}')
    logger.info(f'{name} - input_data: {input_data}')    
    logger.info(f'{name} - context: {context}')
    # logger.info(f'{name} - {decoder.decode(input_data, content_type)}')
    
    stream = None
    if type(input_data)==bytearray:
        stream = BytesIO(input_data)
        stream_value = stream.getvalue().decode()
        logger.info(f'{name} - stream_value: {stream_value}')
        
        decoded = json.loads(stream_value)
        logger.info(f'{name} - {type(decoded)}')
        logger.info(f'{name} - {decoded}')
        
    if type(decoded['data'])==str:
        decoded['data'] = [decoded['data']]
    logger.info(f'{name} - {decoded}')
    return decoded
            # if type(decoded)==str:
            #     decoded = [decoded]
        # elif type(input_data)==list:
        #     decoded = [input_data]





        # if content_type == 'application/x-npy':
        #     if type(input_data["data"])==str:
        #         decoded = {"data": [input_data["data"]], "decode_level": input_data["decode_level"]}
        #         # decoded = [[input_data[0]], input_data[1]]
        #     elif type(input_data["data"])==list:
        #         decoded = input_data
        # elif content_type == 'application/local-npy':
        #     stream = BytesIO(input_data)
        #     decoded = stream.getvalue().decode()
        #     logger.info(f'{name} - {decoded}')
        #     logger.info(f'{name} - {type(decoded)}')
            # decoded = json.loads(input_data)
        # else:
        #     stream = BytesIO(input_data)
        #     decoded = stream.getvalue().decode()
        #     # decoded = np.load(stream, allow_pickle=True).tolist()
        #     logger.info(f'{name} - {decoded}')
        #     logger.info(f'{name} - {type(decoded)}')
        #     if type(decoded)==str:
        #         decoded = [decoded]
            # logger.info(f'{name} - BytesIO')
            # works for strings and lists




# def input_fn(request_body, content_type='application/x-npy'):
#     logger.info('Deserializing the input data.')
#     logger.info(f'content_type: {content_type}')
#     logger.info(f'request_body: {request_body}')
    
#     input_data = json.loads(request_body)
#     logger.info(f'input_data: {input_data}')
#     recieved_string = [input_data['text']]
#     logger.info(f'recieved_string: {recieved_string}')
    
#     return recieved_string

    # if content_type == 'text/csv':
    #     input_data = json.loads(request_body)
    #     recieved_string = [input_data['text']]
    #     return recieved_string

         
#     elif content_type == 'application/json':
#         input_data = json.loads(request_body)
#         url = input_data['url']
#         logger.info(f'Image url: {url}')
#         image_data = Image.open(requests.get(url, stream=True).raw)
        
#         image_transform = transforms.Compose([
#             transforms.Resize(size=256),
#             transforms.CenterCrop(size=224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
# ​
#         return image_transform(image_data)
#     raise Exception(f'Requested unsupported ContentType in content_type {content_type}')


def predict_fn(input_data, model):
    name = 'predict_fn'
    logger = get_logger()
    logger.info(f'{name} - predict function')
    logger.info(f'{name} - {input_data}')
    logger.info(f'{name} - type - {type(input_data["data"])}')
    
    tokenizer = model['tokenizer']
    seq2seqmodel = model['seq2seqmodel']
    model = model['model']
    
    if input_data["decode_level"]=='tokenize':
        tokenized_inputs = tokenizer(input_data["data"], truncation=True, padding=True,max_length = 30)
        logger.info(f"{name} - {tokenized_inputs}")
        logger.info(f"{name} - {tokenized_inputs['input_ids'], tokenized_inputs['attention_mask']}")

        return tokenized_inputs['input_ids']
    
    elif input_data["decode_level"]=='generate':
        tokenized_inputs = tokenizer(input_data["data"], truncation=True, padding=True,max_length = 30)
        logger.info(f"{name} - {tokenized_inputs}")
        logger.info(f"{name} - {tokenized_inputs['input_ids'], tokenized_inputs['attention_mask']}")
    
        pred = seq2seqmodel.generate(torch.tensor(tokenized_inputs[0].ids).view(1,-1))
        translated_pred = tokenizer.decode(pred[0], skip_special_tokens=True)
        logger.info(f'{name} - outputs generated')
        logger.info(translated_pred)
        return translated_pred
            
    elif input_data["decode_level"]=='embed':
        logger.info(f'{name} - {input_data}')
        logger.info(f'{name} - {input_data["data"]}')
        encoded = model.encode(input_data["data"])
        logger.info(f'{name} - encoded - {encoded}')
        logger.info(type(encoded))
        return encoded

    # decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
    # decoder_input_ids = llm._shift_right(decoder_input_ids)

    
    # outputs = []
    # for i in range(len(tokenized_inputs.input_ids)):
    #     outputs.append(llm(torch.tensor(tokenized_inputs[0].ids).view(1,-1), 
    #                                  torch.tensor(tokenized_inputs[0].attention_mask).view(1,-1),
    #                                  decoder_input_ids=decoder_input_ids)
    #                                  .logits.argmax().tolist())
    
    

        


# def output_fn(prediction, content_type):
#     logger.info(f'output function')
#     logger.info(prediction)
#     return prediction