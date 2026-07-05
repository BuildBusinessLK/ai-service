import os
import io
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    # Prefer community loaders where available
    from langchain_community.document_loaders import TextLoader, CSVLoader
except Exception:
    from langchain_community.document_loaders import TextLoader
    CSVLoader = None

try:
    # PDF loaders (try a few options)
    from langchain.document_loaders import PyPDFLoader as PyPDFLoader
except Exception:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except Exception:
        PyPDFLoader = None

try:
    from langchain.document_loaders import Docx2txtLoader
except Exception:
    Docx2txtLoader = None

try:
    from langchain.document_loaders import UnstructuredPowerPointLoader
except Exception:
    UnstructuredPowerPointLoader = None

try:
    from langchain.document_loaders import UnstructuredExcelLoader
except Exception:
    UnstructuredExcelLoader = None

try:
    import pandas as pd
except Exception:
    pd = None

DATA_PATH = os.getenv("DATA_PATH", "data/")
DB_PATH = os.getenv("VECTORSTORE_PATH", "rag/vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _load_via_pandas(path: str) -> List:
    """Load spreadsheet-like files into a single text Document via pandas."""
    if pd is None:
        # fallback: read as text
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return [TextLoader(path, encoding="utf-8").load()[0]]
    try:
        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
        text = df.to_csv(index=False)
        from langchain.schema import Document

        return [Document(page_content=text, metadata={"source": path})]
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return [TextLoader(path, encoding="utf-8").load()[0]]


def load_documents() -> List:
    docs = []
    for root, _, files in os.walk(DATA_PATH):
        for file in sorted(files):
            path = os.path.join(root, file)
            lower = file.lower()
            try:
                if lower.endswith(".txt") or lower.endswith(".md"):
                    loader = TextLoader(path, encoding="utf-8")
                    docs.extend(loader.load())

                elif lower.endswith(".csv"):
                    if CSVLoader is not None:
                        loader = CSVLoader(path)
                        docs.extend(loader.load())
                    else:
                        docs.extend(_load_via_pandas(path))

                elif lower.endswith(".pdf"):
                    if PyPDFLoader is not None:
                        loader = PyPDFLoader(path)
                        docs.extend(loader.load())
                    else:
                        # fallback to raw bytes -> note this will lose structure
                        with open(path, "rb") as fh:
                            text = fh.read().decode("latin-1", errors="ignore")
                            from langchain.schema import Document

                            docs.append(Document(page_content=text, metadata={"source": path}))

                elif lower.endswith(".docx"):
                    if Docx2txtLoader is not None:
                        loader = Docx2txtLoader(path)
                        docs.extend(loader.load())
                    else:
                        docs.extend(_load_via_pandas(path))

                elif lower.endswith(".pptx"):
                    if UnstructuredPowerPointLoader is not None:
                        loader = UnstructuredPowerPointLoader(path)
                        docs.extend(loader.load())
                    else:
                        with open(path, "rb") as fh:
                            text = fh.read().decode("latin-1", errors="ignore")
                            from langchain.schema import Document

                            docs.append(Document(page_content=text, metadata={"source": path}))

                elif lower.endswith(".xlsx") or lower.endswith(".xls"):
                    if UnstructuredExcelLoader is not None:
                        loader = UnstructuredExcelLoader(path)
                        docs.extend(loader.load())
                    else:
                        docs.extend(_load_via_pandas(path))

                elif lower.endswith(".html") or lower.endswith(".htm"):
                    # simple HTML fallback: read as text
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                        from langchain.schema import Document

                        docs.append(Document(page_content=text, metadata={"source": path}))

                else:
                    # Unknown file type — attempt text loader
                    loader = TextLoader(path, encoding="utf-8")
                    docs.extend(loader.load())
            except Exception:
                # Ignore single-file errors and continue
                continue
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
        raise RuntimeError(f"No documents found in {DATA_PATH}")

    print("Splitting documents...")
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} documents and created {len(chunks)} chunks.")

    print("Creating embeddings & saving FAISS index...")
    create_vectorstore(chunks)

    print("Done. Vector DB created.")