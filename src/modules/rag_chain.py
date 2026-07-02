import os
from typing import Callable, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from src.modules.document import Document


class SimpleLLM:
    def __init__(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            # load_in_4bit=False, removed for transformers >= 5.0.0
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            return_full_text=False,
            # device=0 if torch.cuda.is_available() else -1,
        )

    def generate(self, prompt: str) -> str:
        result = self.pipe(prompt)
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        return ""


def load_faiss_vectorstore(faiss_index_path: str, embeddings, top_k: int = 5, index_name: str = "bible_faiss_index"):
    from src.modules.vector_db import load_faiss_vectorstore as load_store
    return load_store(faiss_index_path, embeddings, top_k=top_k, index_name=index_name)


def initialize_llm(model_name_or_path: str):
    try:
        llm = SimpleLLM(model_name_or_path)
        print(f"Using local HuggingFace model: {model_name_or_path}")
    except Exception as e:
        print(f"Error initializing local HuggingFace model '{model_name_or_path}': {e}")
        print("Attempting fallback to smaller model 'distilgpt2'...")
        try:
            llm = SimpleLLM('distilgpt2')
            print("Using fallback model: distilgpt2")
        except Exception as e2:
            print(f"Fallback model initialization also failed: {e2}")
            llm = None
    return llm


def get_prompt_template() -> str:
    return (
        "You are a helpful assistant that answers questions about the Bible. "
        "Use the provided context to answer the question. "
        "If the answer is not in the context, state that you don't know. "
        "Always cite the Bible verse (e.g., [John 3:16]) if you use information from it.\n\n"
        "Context:\n{context}\n\nQuestion: {input}\nAnswer:"
    )


def build_rag_chain(retriever, llm, prompt_template: str) -> Callable[[str], Dict[str, object]]:
    def retrieval_chain(query: str) -> Dict[str, object]:
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(
            f"{doc.metadata.get('location', 'Unknown')}: {doc.page_content}"
            for doc in docs
        )
        prompt = prompt_template.format(context=context, input=query)
        answer = llm.generate(prompt) if llm else "LLM is not available."
        return {
            "answer": answer,
            "source_documents": docs,
        }

    return retrieval_chain


def run_rag_query(retrieval_chain: Callable[[str], Dict[str, object]]):
    while True:
        user_query = input("\nEnter your Bible question (or 'quit' to exit): ").strip()
        if user_query.lower() == 'quit':
            print("Exiting RAG system. Goodbye!")
            break
        print(f"Processing query: '{user_query}'...")
        try:
            response = retrieval_chain(user_query)
            print("\n--- Answer ---")
            print(response.get("answer", ""))

            source_docs = response.get("source_documents", [])
            if source_docs:
                print("\n--- Sources ---")
                for i, doc in enumerate(source_docs):
                    location = doc.metadata.get("location", "N/A")
                    print(f"  Source {i + 1}: {location}")
                    print(f"    {doc.page_content.strip()[:200]}...\n")
            else:
                print("No retrieved source documents available.")
            print("--------------------")
        except Exception as e:
            print(f"An error occurred during query processing: {e}")


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
