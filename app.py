import os
import ollama
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)



system_prompt = """ 
    You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so.
"""



def call_llm(context: str, prompt: str):
   
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break



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



def query_collection(prompt: str, n_results: int = 10):
   
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results



def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=5)
    #st.write(ranks)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])
    #st.write(relevant_text)
    #st.divider()

    return relevant_text, relevant_text_ids



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

    st.header("RAG Question")
    prompt = st.text_area("Ask a question related to document:")
    ask = st.button(
        "Ask",
    )

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0] # nested list in dict, get 0th index of it
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
        response = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)

        with st.expander("Retrieved documents"):
            st.write(results)

        with st.expander("Most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)


    