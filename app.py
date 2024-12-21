import streamlit as st



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




