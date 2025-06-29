import os
import json
from typing import List, Dict, Any

# Assuming Document class is available or fallback to local simplified Document
try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        """A simplified Document class to hold content and metadata."""

        def __init__(self, page_content: str, metadata: Dict[str, Any]):
            self.page_content = page_content
            self.metadata = metadata

        def __repr__(self):
            return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

        def to_dict(self):
            return {"page_content": self.page_content, "metadata": self.metadata}


def get_project_root() -> str:
    """Gets the absolute path to the project root directory."""
    # Adjust this if your script structure is different (e.g., if this script is in `scripts` folder)
    return os.path.dirname(os.path.abspath(__file__))


def load_documents_from_jsonl(file_path: str) -> List[Document]:
    """
    Loads Document objects from a JSONL file.
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


# --- Main execution for building the RAG Chain ---
if __name__ == "__main__":
    # Define paths
    project_root = os.path.join(get_project_root(), "HB_reader_rag")  # Ensure absolute path
    data_dir = os.path.join(project_root, r"data\processed_data")
    processed_data_file = os.path.join(data_dir, "akjv_verses.jsonl")

    # Path to save the FAISS index
    faiss_index_path = os.path.join(project_root, "faiss_index")
    faiss_index_file = os.path.join(faiss_index_path, "bible_faiss_index.faiss")

    if not os.path.exists(processed_data_file):
        print(f"Error: Processed data file not found at {processed_data_file}.")
        print("Please run the data processing step first (e.g., your data_processor.py).")
        exit()

    faiss_index_dir = os.path.join(faiss_index_path, "bible_faiss_index")
    if not os.path.exists(faiss_index_dir + ".faiss") or not os.path.exists(faiss_index_dir + ".pkl"):
        print(f"Error: FAISS index not found at {faiss_index_dir}.faiss/.pkl.")
        print("Please run the vector database building step first (e.g., your vector_db_builder.py).")
        exit()

    print("--- Starting RAG Chain Building ---")

    # 1. Initialize Embedding Model (Must be the same one used to build the FAISS index)
    print("Initializing embedding model (must match the one used for FAISS index)...")
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    # Assuming BAAI/bge-small-en-v1.5 or BAAI/bge-large-en-v1.5 was used
    # Ensure you use the exact model that generated your FAISS embeddings!
    # If you used 'BAAI/bge-large-en-v1.5', change model_name accordingly.
    embedding_model_name = "BAAI/bge-large-en-v1.5"  # Or "BAAI/bge-small-en-v1.5"
    try:
        embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_name)
        print(f"Using HuggingFaceBgeEmbeddings: {embedding_model_name}")
    except Exception as e:
        print(f"Error loading {embedding_model_name}: {e}. Trying fallback 'all-MiniLM-L6-v2'.")
        from langchain_community.embeddings import SentenceTransformerEmbeddings

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print(
            "Falling back to SentenceTransformerEmbeddings: all-MiniLM-L6-v2. Please ensure this matches your FAISS index.")

    # 2. Load the FAISS Vector Store
    print("Loading FAISS vector database...")
    from langchain_community.vectorstores import FAISS

    try:
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, index_name="bible_faiss_index",
                                       allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant documents
        print("FAISS vector database loaded successfully.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        print(
            "Please ensure the FAISS files (.faiss and .pkl) exist in 'faiss_index' directory and the embedding model matches.")
        exit()

    # 3. Initialize the Large Language Model (LLM)
    print("Initializing Large Language Model (LLM)...")
    # Choose ONE of the following LLM options:

    # Option A: OpenAI GPT Models (Requires OPENAI_API_KEY environment variable)
    # from langchain_openai import ChatOpenAI
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Replace with your actual key or set as env var
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    # print("Using OpenAI GPT-3.5-turbo.")

    # Option B: Google Generative AI Models (Requires GOOGLE_API_KEY environment variable)
    from langchain_google_genai import ChatGoogleGenerativeAI

    # os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY" # Replace with your actual key or set as env var
    # Make sure you have your GOOGLE_API_KEY set as an environment variable
    # For Gemini, model_name is typically "gemini-pro" or "gemini-1.5-pro-latest"
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
        print("Using Google Gemini Pro.")
    except Exception as e:
        print(f"Error initializing Google Gemini Pro: {e}. Make sure GOOGLE_API_KEY is set and valid.")
        print("Falling back to a placeholder LLM if available, or consider using another option.")
        llm = None  # Set to None if initialization fails

    # Option C: Local Ollama Models (Requires Ollama server running with model pulled, e.g., 'ollama run llama3')
    # from langchain_community.llms import Ollama
    # llm = Ollama(model="llama3") # Make sure you have 'llama3' pulled via `ollama pull llama3`
    # print("Using local Ollama Llama3.")

    # Option D: Local HuggingFace Models (More complex setup, requires a downloaded model and potentially GPU)
    # from langchain_community.llms import HuggingFacePipeline
    # from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    # model_id = "mistralai/Mistral-7B-Instruct-v0.2" # Example model
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    # llm = HuggingFacePipeline(pipeline=pipe)
    # print(f"Using local HuggingFace model: {model_id}")

    if llm is None:
        print(
            "LLM initialization failed. Cannot proceed with RAG chain. Please check your API keys or local model setup.")
        exit()

    # 4. Define the Prompt Template
    print("Defining prompt template...")
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # We'll use a more advanced prompt template that can handle chat history later if needed.
    # For now, a simple question-answering template with context.
    # The prompt explicitly asks the LLM to refer to the source.
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that answers questions about the Bible. "
                       "Use the provided context to answer the question. "
                       "If the answer is not in the context, state that you don't know. "
                       "Always cite the Bible verse (e.g., [John 3:16]) if you use information from it. "
                       "Context: {context}"),
            ("user", "{input}"),
        ]
    )
    print("Prompt template defined.")

    # 5. Build the RAG Chain
    print("Building the RAG chain...")
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    # First, create a chain that takes a question and a list of retrieved documents
    # and combines them into a single prompt for the LLM.
    document_chain = create_stuff_documents_chain(llm, prompt_template)

    # Then, create the retrieval chain that first retrieves documents
    # and then passes them to the document chain.
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG chain built successfully.")

    # 6. Test the RAG Chain
    print("\n--- Testing the RAG Chain ---")
    while True:
        user_query = input("\nEnter your Bible question (or 'quit' to exit): ").strip()
        if user_query.lower() == 'quit':
            print("Exiting RAG system. Goodbye!")
            break

        print(f"Processing query: '{user_query}'...")
        try:
            response = retrieval_chain.invoke({"input": user_query})

            # The response structure depends on the chain. For create_retrieval_chain:
            # response = {'input': 'user_query', 'context': [Doc1, Doc2...], 'answer': 'Generated Answer'}

            print("\n--- Answer ---")
            print(response["answer"])

            # Optionally, print the retrieved context for debugging/verification
            print("\n--- Sources Used ---")
            if "context" in response and response["context"]:
                for i, doc in enumerate(response["context"]):
                    location = doc.metadata.get("location", "N/A")
                    print(f"  Source {i + 1}: {location} - {doc.page_content.strip()[:100]}...")  # Show first 100 chars
            else:
                print("No specific sources retrieved or provided by the LLM.")
            print("--------------------")

        except Exception as e:
            print(f"An error occurred during query processing: {e}")
            print("Please ensure your LLM and vector store are properly configured.")
