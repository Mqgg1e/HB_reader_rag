import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from typing import List, Dict, Any


def load_faiss_vectorstore(faiss_index_path: str, embeddings, index_name: str = "bible_faiss_index", top_k: int = 5):
    from langchain_community.vectorstores import FAISS
    try:
        vectorstore = FAISS.load_local(
            faiss_index_path,
            embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        print("FAISS vector database loaded successfully.")
        return retriever
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None


# def initialize_llm():
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     try:
#         llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
#         print("Using Google Gemini Pro.")
#     except Exception as e:
#         print(f"Error initializing Google Gemini Pro: {e}.")
#         llm = None
#     return llm


def initialize_llm(model_name_or_path: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_huggingface import HuggingFacePipeline
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Load model in 4-bit (or 8-bit) quantization for lower memory usage if GPU is available
        # If no GPU, or you have sufficient RAM, remove load_in_4bit=True
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",  # Automatically maps model to available devices (GPU/CPU)
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance and memory on some GPUs
            load_in_4bit=False  # Crucial for fitting larger models on limited GPU memory
            # If load_in_4bit causes issues or you have ample GPU RAM, try load_in_8bit=True or remove it for full precision
        )

        # Create a Hugging Face pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,  # Max tokens the LLM will generate for an answer
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1  # Avoid repetitive answers
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"Using local HuggingFace model: {model_name_or_path.split('/')[-3]}/{model_name_or_path.split('/')[-2]}")

    except Exception as e:
        print(f"Error initializing local HuggingFace model: {e}")
        print(
            "Please ensure the model path is correct, and necessary libraries (transformers, accelerate, bitsandbytes) are installed.")
        print("Also check if you have added the model as 'Input' in Kaggle Notebook.")
        llm = None  # Set to None if initialization fails
    return llm


def get_prompt_template():
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that answers questions about the Bible. "
                       "Use the provided context to answer the question. "
                       "If the answer is not in the context, state that you don't know. "
                       "Always cite the Bible verse (e.g., [John 3:16]) if you use information from it. "
                       "Context: {context}"),
            ("user", "{input}"),
        ]
    )


def build_rag_chain_with_summary(retriever, llm, prompt_template):
    def convert_summary_to_original_content(docs: List[Document]) -> List[Document]:
        converted_docs = []
        for doc in docs:
            original_content = doc.metadata.get("original_chapter_content")
            location = doc.metadata.get("location", "Unknown")
            if original_content:
                new_doc = Document(
                    page_content=original_content,
                    metadata={
                        "original_summary": doc.page_content,  # optional to keep summary in metadata
                        **doc.metadata
                    }
                )
                converted_docs.append(new_doc)
            else:
                # fallback and exception
                print(f"Warning: original_chapter_content not found for {location}. Using summary as context.")
                converted_docs.append(doc)
        return converted_docs

    document_chain = create_stuff_documents_chain(llm, prompt_template)

    from langchain_core.runnables import RunnablePassthrough

    rag_chain = (
            RunnablePassthrough.assign(context=retriever | convert_summary_to_original_content)
            | document_chain
    )

    print("RAG chain built: Retriever fetches summaries, then original content is passed to LLM.")
    return rag_chain


def run_rag_query(retrieval_chain):
    """
    Runs an interactive RAG query loop, adapting source display based on context type.

    Args:
        retrieval_chain: The RAG chain built by build_rag_chain.
                         This chain should return documents in its 'context' key
                         that have appropriate metadata (e.g., 'context_source_type').
    """
    while True:
        user_query = input("\nEnter your Bible question (or 'quit' to exit): ").strip()
        if user_query.lower() == 'quit':
            print("Exiting RAG system. Goodbye!")
            break
        print(f"Processing query: '{user_query}'...")
        try:
            # Invoke the RAG chain with the user's input
            response = retrieval_chain.invoke({"input": user_query})

            print("\n--- Answer ---")
            print(response["answer"])

            print("\n--- Sources Used ---")
            if "context" in response and response["context"]:
                for i, doc in enumerate(response["context"]):
                    location = doc.metadata.get("location", "N/A")
                    context_type = doc.metadata.get("context_source_type", "unknown")

                    print(f"  Source {i + 1}: {location}")

                    # Adapt source display based on context_source_type
                    if context_type == "original_chapter_content":
                        # This means the LLM received the full original chapter
                        original_summary_preview = doc.metadata.get("retrieved_summary_text",
                                                                    "No summary preview").strip()[:100]
                        original_content_preview = doc.page_content.strip()[:100]
                        print(f"    - **Retrieved Summary (used for search preview):** {original_summary_preview}...")
                        print(
                            f"    - **Context Provided to LLM (Original Chapter Content preview):** {original_content_preview}...")
                    elif context_type == "summary_fallback":
                        # This means original content was missing, and the summary itself was used
                        summary_content_preview = doc.page_content.strip()[:100]
                        print(
                            f"    - **Context Provided to LLM (Summary Fallback preview):** {summary_content_preview}...")
                    else:
                        # For cases where no specific context_source_type is set (e.g., old Document structure)
                        # or if you decided to bypass the conversion for some reason.
                        # We'll just print the page_content as the context that was used.
                        context_preview = doc.page_content.strip()[:100]
                        print(
                            f"    - **Context Provided to LLM (General Document Content preview):** {context_preview}...")
            else:
                print("No specific sources retrieved or provided by the LLM.")
            print("--------------------")
        except Exception as e:
            print(f"An error occurred during query processing: {e}")


def build_rag_chain(retriever, llm, prompt_template):
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


# if __name__ == "__main__":
#     project_root = os.path.dirname(os.path.abspath(__file__))
#     data_dir = os.path.join(project_root, "data", "processed_data")
#     processed_data_file = os.path.join(data_dir, "akjv_verses.jsonl")
#     faiss_index_path = os.path.join(project_root, "faiss_index")
#
#     if not os.path.exists(processed_data_file):
#         print(f"Error: Processed data file not found at {processed_data_file}.")
#         exit()
#
#     from vector_db import initialize_embeddings
#
#     retriever = load_faiss_vectorstore(faiss_index_path, initialize_embeddings())
#     if retriever is None:
#         print("Could not load FAISS vector store. Exiting.")
#         exit()
#
#     # llm is mistral 7b instruct v0.1 hf, fill the model name or path accordingly
#     llm = initialize_llm("mistralai/Mistral-7B-Instruct-v0.1")
#     if llm is None:
#         print("LLM initialization failed. Exiting.")
#         exit()
#
#     prompt_template = get_prompt_template()
#     rag_chain = build_rag_chain(retriever, llm, prompt_template)
#     run_rag_query(rag_chain)
