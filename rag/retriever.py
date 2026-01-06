import os
from rag.embeddings import embedding_model
from langchain.vectorstores import FAISS
from ingest.build_vectorstore import build_index

VECTOR_DIR = "vectorstore"

def get_retriever():
    # Rebuild vectorstore if missing
    if not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        build_index()

    db = FAISS.load_local(VECTOR_DIR, embedding_model)
    return db.as_retriever()
