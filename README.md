# streamlit_rag_app

Streamlit RAG app solution with llama3, LangChain, Ollama and ChromaDB.

## Prerequisites:
1. Python 3.12.5
2. NVidia GPU with 8GB VRAM. 

## Setup:
1. Create new virtual env:
``` sh
python -m venv venv
```
2. Activate your virtual env:
``` sh
venv/Scripts/activate
```
3. Install packages from included requirements.txt:
``` sh
pip install -r .\requirements.txt
```
4. Install Ollama: https://ollama.com/


## Run:
``` sh
streamlit run app.py
```

## Usage:
1. Upload source PDF file.
2. Ask a question about file content.
   
![alt text](https://github.com/dawmro/streamlit_rag_app/blob/main/image.PNG?raw=true)
