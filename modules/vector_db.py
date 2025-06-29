import os
import json
from typing import List, Dict, Any
from modules.document import Document


def load_documents_from_jsonl(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                documents.append(Document(page_content=data['page_content'], metadata=data['metadata']))
        print(f"Successfully loaded {len(documents)} documents from {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while loading documents: {e}")
    return documents


def initialize_embeddings(model_name: str = "BAAI/bge-large-en-v1.5"):
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    try:
        embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)
        print(f"Using HuggingFaceBgeEmbeddings: {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}. Trying fallback 'all-MiniLM-L6-v2'.")
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Falling back to SentenceTransformerEmbeddings: all-MiniLM-L6-v2.")
    return embeddings


def build_and_save_faiss_vector_db(documents: List[Document], embeddings, faiss_index_path: str,
                                   index_name: str = "bible_faiss_index"):
    from langchain_community.vectorstores import FAISS
    os.makedirs(faiss_index_path, exist_ok=True)
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("FAISS vector database built successfully.")
        vectorstore.save_local(faiss_index_path, index_name=index_name)
        print(f"FAISS index saved to: {os.path.join(faiss_index_path, index_name)}.faiss")
    except Exception as e:
        print(f"An error occurred during FAISS building or saving: {e}")


# def load_faiss_vectorstore(faiss_index_path: str, embeddings, index_name: str = "bible_faiss_index"):
#     from langchain_community.vectorstores import FAISS
#     try:
#         vectorstore = FAISS.load_local(
#             faiss_index_path,
#             embeddings,
#             index_name=index_name,
#             allow_dangerous_deserialization=True
#         )
#         print("FAISS vector database loaded successfully.")
#         return vectorstore
#     except Exception as e:
#         print(f"Error loading FAISS index: {e}")
#         return None
