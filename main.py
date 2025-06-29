import os
import yaml

from modules import data_processing
from modules import vector_db
from modules import rag_chain


# read configuration from Config.yaml

config_path = os.path.join(os.path.dirname(__name__), 'config', 'Config.yaml')

with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# data processing

raw_bible_text = data_processing.read_text_file(config['bible_file_path'])
cleaned_bible_text = data_processing.clean_bible_text_header(raw_bible_text)

data_processing.save_cleaned_text_to_file(cleaned_bible_text,
                                          os.path.join(config['processed_bible_file_path'], "cleaned_bible.txt"))
data_processing.save_documents_to_jsonl(
    data_processing.parse_bible_verses(cleaned_bible_text),
    config['processed_jsonl_file']
)

# os.makedirs(config['faiss_index_path'], exist_ok=True)
documents = data_processing.load_documents_from_jsonl(config['processed_jsonl_file'])
# print(documents[:500])


#embeddings

embeddings = vector_db.initialize_embeddings(model_name=config['embedding_model_name'])
vector_db.build_and_save_faiss_vector_db(documents, embeddings, config['faiss_index_path'])

# rag chain

retriever = rag_chain.load_faiss_vectorstore(
    config['faiss_index_path'], embeddings, index_name=config['faiss_index_name'], top_k=config['top_k']
)
llm = rag_chain.initialize_llm(model_name_or_path=config['llm_model'])
rag_chain = rag_chain.build_rag_chain(retriever, llm, rag_chain.get_prompt_template())
rag_chain.run_rag_query(rag_chain)
