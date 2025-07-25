import os
import yaml

from src.modules.data_processing import (
    read_text_file,
    clean_bible_text_header,
    save_cleaned_text_to_file,
    save_documents_to_jsonl,
    parse_bible_verses,
    load_documents_from_jsonl,
)
from src.modules.vector_db import (
    build_and_save_faiss_vector_db,
    initialize_embeddings,
)
from src.modules.rag_chain import (
    load_faiss_vectorstore,
    initialize_llm,
    build_rag_chain,
    get_prompt_template,
    run_rag_query,
)

class BaseWorkflow:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'Config.yaml')
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        # Expose commonly used functions as instance attributes
        self.read_text_file = read_text_file
        self.clean_bible_text_header = clean_bible_text_header
        self.save_cleaned_text_to_file = save_cleaned_text_to_file
        self.save_documents_to_jsonl = save_documents_to_jsonl
        self.parse_bible_verses = parse_bible_verses
        self.load_documents_from_jsonl = load_documents_from_jsonl
        self.build_and_save_faiss_vector_db = build_and_save_faiss_vector_db
        self.initialize_embeddings = initialize_embeddings
        self.load_faiss_vectorstore = load_faiss_vectorstore
        self.initialize_llm = initialize_llm
        self.build_rag_chain = build_rag_chain
        self.get_prompt_template = get_prompt_template
        self.run_rag_query = run_rag_query

