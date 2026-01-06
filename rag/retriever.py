import os
from rag.embeddings import embedding_model
from langchain.vectorstores import FAISS
from ingest.build_vectorstore import build_index  # your function

VECTOR_DIR = "vectorstore"

# If FAISS index doesn't exist, rebuild it
if not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
    print("FAISS index not found. Building index...")
    build_index()  # your function that scrapes PDFs/website and builds FAISS

# Now load the FAISS vectorstore
db = FAISS.load_local(VECTOR_DIR, embedding_model)
retriever = db.as_retriever()
