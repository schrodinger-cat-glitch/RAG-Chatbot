import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PDF_DIR = "data/pdfs"
VECTOR_DIR = "vectorstore"

documents = []

for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        path = os.path.join(PDF_DIR, filename)
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

docs = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embedding_model)

os.makedirs(VECTOR_DIR, exist_ok=True)
db.save_local(VECTOR_DIR)

print("Vectorstore created successfully!")
