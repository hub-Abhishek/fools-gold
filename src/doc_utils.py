import os
from PyPDF2 import PdfReader
from langchain.docstore.document import Document

def process_pdf(file, st, documents):
    pdfReader = PdfReader(file)
    st.write(f'Number of pages in the uploaded pdf {file.name} - {len(pdfReader.pages)}')
    
    for i, page in enumerate(pdfReader.pages):
        documents.append(Document(page_content=page.extract_text(),
                                metadata={'title': file.name, 
                                            'page_number': i}))
    return documents

def process_text(file, st, documents):
    text = ''
    for line in file:
        text +=  str(line.decode())
    documents.append(Document(page_content=text,
                            metadata={'title': file.name,}))
    return documents


def process_file(uploaded_file, st):
    
    st.write('Processing your files...')
    st.write('Please wait...')
    st.write('This may take a few minutes...')
    
    files = []
    documents = []
    for file in uploaded_file:
        
        files.append(file.name)
        st.write(f'Processing document - {file.name} which is a {file.type} file. Please wait...')
        
        if file.type=='application/pdf': 
            documents = process_pdf(file, st, documents)
        
        if file.type=='text/plain':
            documents = process_text(file, st, documents)

    new_files = check_for_new_files(files, st)
    
    return documents, new_files


def check_for_files(st):
    if not os.path.exists('app_database/files.txt'):
        with open('app_database/files.txt','w') as f:
            st.write('Created new file - files.txt')
            f.close()
    if not os.path.exists('app_database/result.txt'):
        with open('app_database/result.txt','w') as f:
            st.write('Created new file - result.txt')
            f.close()

def check_for_new_files(files, st):
    check_for_files(st)
    files = list(set(files))
    old_files = read_old_file_names()
    if old_files==files:
        st.write('No new files uploaded. Using the existing database.')
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