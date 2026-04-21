import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "nomic-embed-text"

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from {os.path.basename(file_path)}")
    return pages

def chunk_documents(pages, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")
    return chunks

def embed_and_store(chunks):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="medical_docs",
        persist_directory=VECTORSTORE_DIR,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

def ingest_file(file_path: str, chunk_size=500, chunk_overlap=50):
    pages = load_pdf(file_path)
    chunks = chunk_documents(pages, chunk_size, chunk_overlap)
    vs = embed_and_store(chunks)
    return vs, len(chunks)

def ingest_from_wikipedia(query: str, max_docs: int = 2):
    print(f"Fetching Wikipedia: '{query}'")
    loader = WikipediaLoader(query=query, load_max_docs=max_docs)
    pages = loader.load()
    print(f"Got {len(pages)} Wikipedia article(s)")
    chunks = chunk_documents(pages)
    vs = embed_and_store(chunks)
    titles = [p.metadata.get("title", query) for p in pages]
    return vs, len(chunks), titles
 
 
def ingest_from_url(url: str):
    print(f"Fetching URL: {url}")
    loader = WebBaseLoader(url)
    pages = loader.load()
    print(f"Loaded {len(pages)} page(s) from URL")
    chunks = chunk_documents(pages)
    vs = embed_and_store(chunks)
    return vs, len(chunks)


def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        collection_name="medical_docs",
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    return vectorstore
