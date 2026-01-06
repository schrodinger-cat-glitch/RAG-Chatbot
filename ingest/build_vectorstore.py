# ingest/build_vectorstore.py

from langchain.vectorstores import FAISS
from rag.embeddings import embedding_model
from ingest.load_pdfs import load_pdfs
from ingest.load_website import load_website
import os

VECTOR_DIR = "vectorstore"

def build_index():
    """
    Build a FAISS vectorstore from PDFs and website content.
    Saves index to VECTOR_DIR.
    """

    # 1️⃣ Load documents from PDFs
    pdf_docs = load_pdfs("data/pdfs")
    print(f"Loaded {len(pdf_docs)} PDF documents.")

    # 2️⃣ Load documents from university website
    website_docs = load_website("https://www.iqra.edu.pk/")
    print(f"Loaded {len(website_docs)} website documents.")

    # Combine all docs
    docs = pdf_docs + website_docs

    # 3️⃣ Ensure VECTOR_DIR exists
    if not os.path.exists(VECTOR_DIR):
        os.makedirs(VECTOR_DIR)

    # 4️⃣ Build FAISS vectorstore
    db = FAISS.from_documents(docs, embedding_model)

    # 5️⃣ Save locally
    db.save_local(VECTOR_DIR)
    print("FAISS index built successfully at", VECTOR_DIR)
