import os
from src.utils import print_message
from PyPDF2 import PdfReader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.docstore.document import Document
import tempfile

def process_pdf(file, documents):
    pdfReader = PdfReader(file)
    # message = f'Number of pages in the uploaded pdf {file.name} - {len(pdfReader.pages)}'
    # print_message(message, st=st)
    
    for i, page in enumerate(pdfReader.pages):
        documents.append(Document(page_content=page.extract_text(),
                                metadata={'title': file, 
                                            'page_number': i}))
    return documents

def process_text(file, documents):
    text = ''
    for line in file:
        text +=  str(line.decode())
    documents.append(Document(page_content=text,
                            metadata={'title': file}))
    return documents

def process_csv(file, documents):
    if file :
    #use tempfile because CSVLoader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        # import pdb; pdb.set_trace()
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                    'delimiter': ','})
        data = loader.load()
        return data


def process_file(uploaded_file):
    
    # message = 'Processing your files, please wait. This will take a few mins.'
    # print_message(message, st=st)
    
    files = []
    documents = []
    for file in uploaded_file:
        
        # files.append(file.name)
        # message = f'Processing document - {file.name} which is a {file.type} file. Please wait...'
        # print_message(message, st=st)
        
        if file[-4:]=='.pdf': 
            documents = process_pdf(file, documents)
        
        if file[-4:]=='.txt':
            documents = process_text(file, documents)
        
        # if file.type=='text/csv':
        #     documents = process_csv(file, st, documents)

    # new_files = check_for_new_files(files, st)
    new_files = False
    return documents, new_files


def check_for_files(st):
    if not os.path.exists('app_database'):
        os.mkdir('app_database')
        message = 'Created new folder - app_database'
        print_message(message, st=st)
    if not os.path.exists('app_database/db'):
        os.mkdir('app_database/db')
        message = 'Created new folder - app_database/db'
        print_message(message, st=st)
    if not os.path.exists('app_database/files.txt'):
        with open('app_database/files.txt','w') as f:
            message = 'Created new file - files.txt'
            print_message(message, st=st)
            f.close()
    if not os.path.exists('app_database/result.txt'):
        with open('app_database/result.txt','w') as f:
            message = 'Created new file - result.txt'
            print_message(message, st=st)
            f.close()
    if not os.path.exists('app_database/query_ids.txt'):
        with open('app_database/query_ids.txt','w') as f:
            message = 'Created new file - query_ids.txt'
            print_message(message, st=st)
            f.close()

def check_for_new_files(files, st):
    check_for_files(st)
    files = list(set(files))
    old_files = read_old_file_names()
    if old_files==files:
        message = 'No new files uploaded. Using the existing database.'
        print_message(message, st=st)
        return False
    else:
        write_new_file_names(files)
        return True


def read_old_file_names():
    with open('app_database/files.txt','r') as f:
        files = f.readlines()
        f.close()
    files = [file.strip() for file in files]
    return files

def write_new_file_names(files):
    with open('app_database/files.txt','w') as f:
        for item in files:
            f.write(item+"\n")
        f.close()
    return None