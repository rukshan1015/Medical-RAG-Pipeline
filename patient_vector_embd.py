import pandas as pd
import numpy as np
import os, json
from langchain.schema import Document
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from glob import glob
from langchain.embeddings import HuggingFaceEmbeddings

dir_path=r"PATIENT NOTES DIRECTORY - patient_notes"

db_name = 'encounter_records'

text_loader_kwags = {'encoding':'utf-8'}

folder = glob(dir_path)

docs=[]

# Creating a list of documents
loader = DirectoryLoader(folder, glob = "**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwags)
folder_docs = loader.load()
for doc in folder_docs:
    docs.append(doc)

# Splitting documents into chunks - we use same tokenizer in embedding model to count the tokens for a chunk, 
# and only emit chunks up to a chosen token budget (plus overlap). So, every chunk is guaranteed to fit the embedder’s with no silent truncation.

tok = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
tok_length = lambda x: len(tok.encode(x, add_special_tokens=False))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=450, 
    chunk_overlap=80,
    length_function=tok_length,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator=False
)
chunks = splitter.split_documents(docs)

# Create vector database 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(
    documents=chunks,
    persist_directory=db_name,
    collection_name='medical_notes',
    embedding=embeddings
)

