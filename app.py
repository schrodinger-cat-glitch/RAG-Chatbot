import streamlit as st
from rag.retriever import get_retriever
from rag.prompt import PROMPT
from rag.llm import llm
import os
from ingest.build_vectorstore import build_index  # your function

if not os.path.exists("vectorstore/index.faiss"):
    build_index()


st.set_page_config(page_title="IQRA University Chatbot", page_icon="🎓")

st.title("🎓 IQRA University AI Assistant")
st.caption("Ask about admissions, policies, programs, and more")

@st.cache_resource
def load_retriever():
    return get_retriever()

retriever = load_retriever()

question = st.text_input("Ask your question:")

if question:
    with st.spinner("Thinking..."):
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(d.page_content for d in docs)

        chain = PROMPT | llm
        response = chain.invoke({
            "context": context,
            "question": question
        })

    st.markdown("### 📌 Answer")
    st.write(response.content)

if question:
    with st.spinner("Thinking..."):
        st.write("🔹 Retrieving documents...")
        docs = retriever.get_relevant_documents(question)

        st.write("🔹 Building context...")
        context = "\n\n".join(d.page_content for d in docs)

        st.write("🔹 Calling LLM...")
        chain = PROMPT | llm
        response = chain.invoke({
            "context": context,
            "question": question
        })

        st.write("🔹 LLM finished")
