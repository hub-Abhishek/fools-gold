import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = 'gle-flan-t5-base--cross-encoder-ms-marco-MiniLM-L-6-v2--llama-2' # os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print(f'event - {event}')
    # print(f'context - {context}')
    
    data = json.loads(json.dumps(event))
    # print("Received event: " + data)
    
    payload = data['data']
    decode_level = data.get('decode_level', 'complete')
    
    print(f'payload - {payload}')
    print(f'decode_level - {decode_level}')
    
    if decode_level=='embed':
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/x-npy',
                                       Body=json.dumps({'data': payload, 'decode_level': decode_level}))
                                  
    elif decode_level=='generate':
        documents = data['documents']
        print(f'documents - {documents}')
        
        data_limit = data["data_limit"]
        print(f'data_limit - {data_limit}')
        
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      ContentType='application/x-npy',
                                      Body=json.dumps({'data': payload, 
                                      'documents': documents, 
                                      'decode_level': decode_level,
                                      'data_limit': data_limit
                                      }))
        
    response_body = response['Body'].read()
    print('response_body')
    print(response_body)
    decoded = response_body.decode('utf-8')
    print(decoded)
    return decoded