import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile



def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    #os.unlink(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, 
        chunk_overlap=128,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(docs)



if __name__ == "__main__":
    # sidebar for document upload
    with st.sidebar:
        st.set_page_config(page_title="RAG App")
        st.header("RAG App")
        uploaded_file = st.file_uploader(
            "Upload PDF file",
            type=["pdf"],
            accept_multiple_files=False
        )

        # chunk document and store in vector database
        process = st.button(
            "Chunk and store in vector db"
        )

    if uploaded_file and process:
        all_splits = process_document(uploaded_file=uploaded_file)
        st.write(all_splits)


    