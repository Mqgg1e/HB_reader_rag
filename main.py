import os
import yaml

from modules.data_processing import (
    read_text_file,
    clean_bible_text_header,
    save_cleaned_text_to_file,
    save_documents_to_jsonl,
    parse_bible_verses,
    load_documents_from_jsonl,
)
from modules.vector_db import (
    build_and_save_faiss_vector_db,
    initialize_embeddings,
)
from modules.rag_chain import (
    load_faiss_vectorstore,
    initialize_llm,
    build_rag_chain,
    get_prompt_template,
    run_rag_query,
)


# read configuration from Config.yaml

config_path = os.path.join(os.path.dirname(__name__), 'config', 'Config.yaml')

with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# data processing

raw_bible_text = read_text_file(config['bible_file_path'])
cleaned_bible_text = clean_bible_text_header(raw_bible_text)

save_cleaned_text_to_file(cleaned_bible_text,
                         os.path.join(config['processed_bible_file_path'], "cleaned_bible.txt"))
save_documents_to_jsonl(
    parse_bible_verses(cleaned_bible_text),
    config['processed_jsonl_file']
)

# os.makedirs(config['faiss_index_path'], exist_ok=True)
documents = load_documents_from_jsonl(config['processed_jsonl_file'])
# print(documents[:500])


#embeddings

embeddings = initialize_embeddings(model_name=config['embedding_model_name'])
build_and_save_faiss_vector_db(documents, embeddings, config['faiss_index_path'])

# rag chain

retriever = load_faiss_vectorstore(
    config['faiss_index_path'], embeddings, index_name=config['faiss_index_name'], top_k=config['top_k']
)
llm = initialize_llm(model_name_or_path=config['llm_model'])
rag_chain = build_rag_chain(retriever, llm, get_prompt_template())
run_rag_query(rag_chain)
