from langchain.document_loaders import PyPDFLoader
from pathlib import Path

def load_pdfs(pdf_dir: str):
    documents = []
    for pdf in Path(pdf_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        documents.extend(loader.load())
    return documents
