# filename: inference.py
import os
import json
import logging
from six import BytesIO, StringIO
from sagemaker_inference import content_types, decoder

import torch
import numpy as np
import sentence_transformers
from sentence_transformers.cross_encoder import CrossEncoder
# from ctransformers import AutoModelForCausalLM

preprompt = "Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful.\n"


qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt_temp = preprompt + qa_template


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

    tokenizer = sentence_transformers.SentenceTransformer(model_name_or_path="models/core_models/google/flan-t5-base")
    # tokenizer.to(device).eval()

    cross_encoder = CrossEncoder("models/core_models/cross-encoder/ms-marco-MiniLM-L-6-v2")
    # cross_encoder.to(device).eval()

    # values = {
    # 'model': 'models/core_models/llama2/llama-2-7b-chat.ggmlv3.q8_0.bin',
    # 'model_type': 'llama',
    # 'model_file': None,
    # 'config': {'max_new_tokens': 256, 'temperature': 0.01}
    # }
    
    # llm = AutoModelForCausalLM.from_pretrained(
    #     values["model"],
    #     model_type=values["model_type"],
    #     model_file=values["model_file"],
    #     # lib=values["lib"],
    #     **values['config'],
    #     )
    # llm.to(device).eval()
    logger.info(f'{name} - Done loading model')
    return {'tokenizer': tokenizer, 
            'cross_encoder': cross_encoder, 
            # 'llm': llm
            }


def input_fn(input_data, content_type, context):
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
    

def predict_fn(input_data, model):
    name = 'predict_fn'
    logger = get_logger()
    logger.info(f'{name} - predict function')
    logger.info(f'{name} - {input_data}')
    logger.info(f'{name} - {input_data["data"]}')
    logger.info(f'{name} - type - {type(input_data["data"])}')
    logger.info(f'{name} - {input_data["decode_level"]}')


    if input_data["decode_level"]=='embed':
        tokenizer = model['tokenizer']
        del model

        logger.info(f'{name} - {input_data}')
        logger.info(f'{name} - {input_data["data"]}')
        encoded = tokenizer.encode(input_data["data"])
        logger.info(f'{name} - encoded - {encoded}')
        logger.info(type(encoded))
        return encoded
    
    # elif input_data["decode_level"]=='generate':
    #     cross_encoder = model['cross_encoder']
    #     llm = model['llm']
    #     del model

    #     query = input_data["data"]
    #     logger.info(f'{name} - query - {query}')

    #     documents = input_data["documents"]
    #     logger.info(f'{name} - documents - {documents}')

    #     data_limit = input_data.get('data_limit', 5)
    #     logger.info(f'{name} - documents - {documents}')

    #     doc_pairs = [[query[0], document] for document in documents]
    #     logger.info(f'{name} - doc_pairs - {doc_pairs}')

    #     scores = cross_encoder.predict(doc_pairs)
    #     logger.info(f'{name} - scores - {scores}')
        
        
    #     sorted_idx = np.argsort(scores)[::-1]
    #     logger.info(f'{name} - sorted_idx - {sorted_idx}')
       
    #     results = []
    #     for idx in sorted_idx[:data_limit]:
    #         results.append(documents[idx])
    #     results = ' '.join(results)
    #     logger.info(f'{name} - results - {results}')

    #     final_prompt = prompt_temp.format(context = results, question = query[0])
    #     logger.info(f'{name} - final_prompt - {final_prompt}')

    #     response = llm(final_prompt)
    #     logger.info(f'{name} - response - {response}')

    #     return response