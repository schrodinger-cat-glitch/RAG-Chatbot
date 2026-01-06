from langchain_community.vectorstores import FAISS
from rag.embeddings import embedding_model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")

# Load FAISS only once (global cache)
db = FAISS.load_local(VECTOR_DIR, embedding_model)

def get_retriever():
    return db.as_retriever(search_kwargs={"k": 4})
