import os
from rag.embeddings import embedding_model
from langchain.vectorstores import FAISS
from ingest.build_vectorstore import build_index  # should now work

VECTOR_DIR = "vectorstore"

if not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
    build_index()

db = FAISS.load_local(VECTOR_DIR, embedding_model)
retriever = db.as_retriever()
