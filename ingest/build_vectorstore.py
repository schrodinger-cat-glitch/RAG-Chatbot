# ingest/build_vectorstore.py

from langchain.vectorstores import FAISS
from rag.embeddings import embedding_model
from ingest.load_pdfs import load_pdfs
from ingest.load_website import load_website
import os

VECTOR_DIR = "vectorstore"

def build_index():
    # Load all documents from PDFs and website
    docs = load_pdfs("data/pdfs") + load_website("https://www.iqra.edu.pk/")
    
    # Build FAISS index
    if not os.path.exists(VECTOR_DIR):
        os.makedirs(VECTOR_DIR)
        
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(VECTOR_DIR)
    print("FAISS index built successfully")
