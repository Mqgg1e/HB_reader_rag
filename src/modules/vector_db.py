import json
import os
from typing import List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.modules.document import Document


class EmbeddingsModel:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Loaded embedding model: {model_name} on {device}")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return vector.reshape(1, -1)


class SimpleRetriever:
    def __init__(self, index: faiss.Index, documents: List[Document], embeddings: EmbeddingsModel, top_k: int = 5):
        self.index = index
        self.documents = documents
        self.embeddings = embeddings
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        scores, indices = self.index.search(query_vector, self.top_k)
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append(self.documents[idx])
        return results


def initialize_embeddings(model_name: str = "BAAI/bge-large-en-v1.5"):
    try:
        return EmbeddingsModel(model_name)
    except Exception as e:
        print(f"Error loading embedding model {model_name}: {e}")
        # print("Falling back to all-MiniLM-L6-v2")
        # return EmbeddingsModel("all-MiniLM-L6-v2")


def use_faiss_gpu() -> bool:
    try:
        return faiss.get_num_gpus() > 0
    except Exception:
        return False


def _get_gpu_index(cpu_index: faiss.Index, gpu_id: int = 0) -> faiss.Index:
    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)


def build_and_save_faiss_vector_db(documents: List[Document], embeddings: EmbeddingsModel, faiss_index_path: str,
                                   index_name: str = "bible_faiss_index", use_gpu: bool = True):
    os.makedirs(faiss_index_path, exist_ok=True)
    try:
        texts = [doc.page_content for doc in documents]
        vectors = embeddings.embed_documents(texts)
        dim = vectors.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)
        index = cpu_index
        if use_gpu and use_faiss_gpu():
            index = _get_gpu_index(cpu_index)
        index.add(vectors)

        index_file = os.path.join(faiss_index_path, f"{index_name}.index")
        if use_gpu and use_faiss_gpu():
            save_index = faiss.index_gpu_to_cpu(index)
        else:
            save_index = index
        faiss.write_index(save_index, index_file)

        metadata_file = os.path.join(faiss_index_path, f"{index_name}_metadata.jsonl")
        with open(metadata_file, "w", encoding="utf-8") as f:
            for doc in documents:
                entry = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print("FAISS vector database built successfully.")
        print(f"FAISS index saved to: {index_file}")
        print(f"Metadata saved to: {metadata_file}")
    except Exception as e:
        print(f"An error occurred during FAISS building or saving: {e}")
        raise


def load_documents_from_metadata(metadata_path: str) -> List[Document]:
    documents = []
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_path}")
    return documents


def load_faiss_vectorstore(faiss_index_path: str, embeddings: EmbeddingsModel, top_k: int = 5,
                           index_name: str = "bible_faiss_index", use_gpu: bool = True) -> SimpleRetriever:
    index_file = os.path.join(faiss_index_path, f"{index_name}.index")
    metadata_file = os.path.join(faiss_index_path, f"{index_name}_metadata.jsonl")

    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index file not found: {index_file}")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    index = faiss.read_index(index_file)
    if use_gpu and use_faiss_gpu():
        index = _get_gpu_index(index)
    documents = load_documents_from_metadata(metadata_file)
    retriever = SimpleRetriever(index, documents, embeddings, top_k=top_k)
    print("FAISS vector database loaded successfully.")
    return retriever
