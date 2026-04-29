import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_PATH = "rag/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    docs = []
    for file in sorted(os.listdir(DATA_PATH)):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_PATH, file), encoding="utf-8")
            docs.extend(loader.load())
    return docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n== ", "\n\n", "\n", ". ", " "],
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_PATH)

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No .txt documents found in {DATA_PATH}")

    print("Splitting documents...")
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} documents and created {len(chunks)} chunks.")

    print("Creating embeddings & saving FAISS index...")
    create_vectorstore(chunks)

    print("Done. Vector DB created.")