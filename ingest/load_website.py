import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

def load_website(url: str):
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    text = " ".join(p.get_text() for p in soup.find_all("p"))

    return [Document(page_content=text, metadata={"source": url})]
