from knowledge_base.loader import load_vectorstore, build_vectorstore
import os

PERSIST_DIR = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
KB_PATH = os.getenv("MEDICAL_KB_PATH", "./data/medical_kb.jsonl")


def get_retriever():
    """
    Load FAISS index from disk if it exists, otherwise build and save it.
    Returns a LangChain retriever ready for semantic search.
    """
    if os.path.exists(PERSIST_DIR):
        vectorstore = load_vectorstore(PERSIST_DIR)
    else:
        vectorstore = build_vectorstore(KB_PATH, PERSIST_DIR)

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Return top 4 matching documents
    )