from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def build_vectorstore(kb_path: str, persist_dir: str):
    """Build a FAISS vector store from the medical knowledge base JSONL file."""
    loader = JSONLoader(file_path=kb_path, jq_schema='.[]', text_content=False)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist index to disk so it doesn't rebuild on every run
    vectorstore.save_local(persist_dir)
    print(f"FAISS index built and saved to {persist_dir}")

    return vectorstore


def load_vectorstore(persist_dir: str):
    """Load an existing FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True  # Required by LangChain for local FAISS loads
    )
    print(f"FAISS index loaded from {persist_dir}")
    return vectorstore