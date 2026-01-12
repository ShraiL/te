# ═══════════════════════════════════════════════════════════════
# INGEST.PY - Load documents into ChromaDB
# RUN THIS FIRST!
# ═══════════════════════════════════════════════════════════════

import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CHANGE THESE IF NEEDED                                       ║
# ╚═══════════════════════════════════════════════════════════════╝
DATA_DIR = 'data'                    # Folder with .txt files
DB_DIR = 'vectorstore'               # Where to save ChromaDB
EMBEDDING_MODEL = 'nomic-embed-text' # Or 'llama3.1'
CHUNK_SIZE = 2000                    # Characters per chunk
CHUNK_OVERLAP = 50                   # Overlap between chunks


def load_doc():
    """Load all .txt files from DATA_DIR folder"""
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.txt'):
            path = os.path.join(DATA_DIR, filename)
            loader = TextLoader(path, autodetect_encoding=True)
            docs.extend(loader.load())
    print(f'Loaded {len(docs)} documents')
    return docs


def split_docs(docs):
    """Split documents into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f'Created {len(chunks)} chunks')
    return chunks


def create_vectorstore(chunks):
    """Create embeddings and store in ChromaDB"""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    db.persist()
    print(f'Vectorstore saved to {DB_DIR}/')


if __name__ == '__main__':
    print("Starting ingestion...")
    docs = load_doc()
    chunks = split_docs(docs)
    create_vectorstore(chunks)
    print('DONE!')