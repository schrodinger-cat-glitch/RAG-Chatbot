from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from rag.embeddings import embedding_model
from ingest.load_pdfs import load_pdfs
from ingest.load_website import load_website

def build_vectorstore():
    docs = []
    docs.extend(load_pdfs("data/pdfs"))
    docs.extend(load_website("https://iqra.edu.pk/"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local("vectorstore")

if __name__ == "__main__":
    build_vectorstore()
