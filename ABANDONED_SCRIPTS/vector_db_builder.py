import os
import json
from typing import List, Dict, Any

# Assuming Document class is available from langchain_core.documents or similar for structured data
# If you don\'t use LangChain\'s Document, you can use a simple dict for each chunk.
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback to local simplified Document if langchain_core is not fully installed or used differently
    class Document:
        """A simplified Document class to hold content and metadata."""

        def __init__(self, page_content: str, metadata: Dict[str, Any]):
            self.page_content = page_content
            self.metadata = metadata

        def __repr__(self):
            return f"Document(page_content=\'{self.page_content[:50]}...\', metadata={self.metadata})"

        def to_dict(self):
            return {"page_content": self.page_content, "metadata": self.metadata}


def get_project_root() -> str:
    """Gets the absolute path to the project root directory."""
    # This assumes the script is in a subdirectory (e.g., 'scripts') within the project root.
    # Adjust as needed based on your actual project structure.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_documents_from_jsonl(file_path: str) -> List[Document]:
    """
    Loads Document objects from a JSONL file.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        List[Document]: A list of loaded Document objects.
    """
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


# --- Main execution for building the Vector DB ---
if __name__ == "__main__":
    # Define paths
    project_root = os.path.join(get_project_root(),"HB_reader_rag")  # Ensure absolute path
    data_dir = os.path.join(project_root, r"data\processed_data")
    processed_data_file = os.path.join(data_dir, "akjv_verses.jsonl")

    # Path to save the FAISS index
    faiss_index_path = os.path.join(project_root, "faiss_index")
    faiss_index_file = os.path.join(faiss_index_path, "bible_faiss_index.faiss")

    # Ensure FAISS index directory exists
    os.makedirs(faiss_index_path, exist_ok=True)

    # 1. Load processed Bible documents
    print(f"Attempting to load documents from: {processed_data_file}")
    documents = load_documents_from_jsonl(processed_data_file)

    if not documents:
        print("No documents loaded. Please ensure 'parsed_bible_verses.jsonl' exists and contains data.")
    else:
        print(f"Loaded {len(documents)} Bible verses.")

        # 2. Initialize Embedding Model
        print("Initializing embedding model: BAAI/bge-large-en-v1.5...")
        # from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        from langchain_huggingface import HuggingFaceEmbeddings

        # Using BAAI/bge-large-en-v1.5 as chosen by the user
        model_name = "BAAI/bge-large-en-v1.5"
        try:
            # Note: For BGE models, it's often recommended to pass query_instruction
            # for search tasks, but for document embedding it's usually fine without it.
            # However, for consistency when querying, you might add it during query time.
            # Here, we're just initializing the document embedder.
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            print(f"Successfully initialized HuggingFaceBgeEmbeddings: {model_name}")
        except Exception as e:
            print(
                f"Error loading {model_name}: {e}. Please ensure 'sentence-transformers' is installed correctly and you have sufficient memory/internet access to download the model.")
            print("Exiting as embedding model initialization failed.")
            exit()  # Exit if the chosen model cannot be loaded

        # 3. Build and save the FAISS vector database
        print("Building FAISS vector database...")
        from langchain_community.vectorstores import FAISS

        try:
            # Create the vector store from documents and embeddings
            vectorstore = FAISS.from_documents(documents, embeddings)
            print("FAISS vector database built successfully.")

            # Save the vector store to disk
            vectorstore.save_local(faiss_index_path, index_name="bible_faiss_index")
            print(f"FAISS index saved to: {faiss_index_path}/bible_faiss_index.faiss")

            # --- Verification (Optional) ---
            print("\n--- Performing a sample similarity search to verify ---")
            # Load the index back to test
            # allow_dangerous_deserialization=True is needed for loading FAISS indexes from disk
            loaded_vectorstore = FAISS.load_local(faiss_index_path, embeddings, index_name="bible_faiss_index",
                                                  allow_dangerous_deserialization=True)

            query = "What did Jesus say about love and commandments?"
            print(f"Query: '{query}'")

            # Perform a similarity search
            results = loaded_vectorstore.similarity_search(query, k=3)  # Get top 3 results

            for i, doc in enumerate(results):
                print(f"\nResult {i + 1}:")
                print(f"  Content: {doc.page_content.strip()[:150]}...")
                print(f"  Location: {doc.metadata.get('location', 'N/A')}")
                print(
                    f"  Book: {doc.metadata.get('book_full_name', 'N/A')}, Chapter: {doc.metadata.get('chapter', 'N/A')}, Verse: {doc.metadata.get('verse', 'N/A')}")
            print("-----------------------------------------------------")

        except Exception as e:
            print(f"An error occurred during FAISS building or saving: {e}")
            print(
                "Please ensure 'faiss-cpu' and 'sentence-transformers' are installed correctly and you have sufficient memory.")