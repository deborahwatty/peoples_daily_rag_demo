import sys
import importlib

# Force the use of pysqlite3 instead of the standard sqlite3
importlib.import_module('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Now, import other modules
import sqlite3
print("SQLite version being used:", sqlite3.sqlite_version)
import os
import getpass
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings




# Step 1: Load the data
df = pd.read_csv('JanFeb2020FilteredFixedReady.tsv', sep='\t', index_col='Unnamed: 0.1')

# Step 2: Initialize the text splitter
text_splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    model_name='text-embedding-3-small'
)

# Step 3: Function to create chunked documents with metadata
def chunk_translation(row):
    metadata = {
        col: row[col] for col in df.columns
        if col not in ['translation', 'merged_content', 'doc_content']
    }
    doc = Document(page_content=row['merged_content'], metadata=metadata)
    chunks = text_splitter.split_documents(documents=[doc])
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata['total_chunks'] = total_chunks
        chunk.metadata['chunk_number'] = i + 1
    return chunks

# Step 4: Apply the function and flatten the chunks
df['chunked_doc'] = df.apply(chunk_translation, axis=1)
all_docs = []
df['chunked_doc'].apply(all_docs.extend)

# Step 5: Prompt for OpenAI API key securely
openai_api_key = getpass.getpass("Enter your OpenAI API key: ")

# Step 6: Initialize OpenAI embeddings and Chroma vectorstore
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(all_docs, embeddings, persist_directory='chroma_chinese')


print("Vectorstore created and persisted successfully.")
