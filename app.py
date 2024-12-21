import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)



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



def get_vector_collection() -> chromadb.Collection:
    
    # use ollama embedding function
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    # persist data on disk
    chroma_client = chromadb.PersistentClient(path="./rag_chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )



def add_to_vector_collection(all_splits: list[Document], file_name: str):
   
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    # create or update data
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")



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
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)


    