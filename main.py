import os

from base_workflow import BaseWorkflow


class MainWorkflow(BaseWorkflow):
    def run(self):
        config = self.config

        # data processing
        raw_bible_text = self.read_text_file(config['bible_file_path'])
        cleaned_bible_text = self.clean_bible_text_header(raw_bible_text)

        self.save_cleaned_text_to_file(
            cleaned_bible_text,
            os.path.join(config['processed_bible_file_path'], "cleaned_bible.txt")
        )
        self.save_documents_to_jsonl(
            self.parse_bible_verses(cleaned_bible_text),
            config['processed_jsonl_file']
        )

        documents = self.load_documents_from_jsonl(config['processed_jsonl_file'])

        # embeddings
        embeddings = self.initialize_embeddings(model_name=config['embedding_model_name'])
        self.build_and_save_faiss_vector_db(documents, embeddings, config['faiss_index_path'])

        # rag chain
        retriever = self.load_faiss_vectorstore(
            config['faiss_index_path'], embeddings, index_name=config['faiss_index_name'], top_k=config['top_k']
        )
        llm = self.initialize_llm(model_name_or_path=config['llm_model'])
        rag_chain = self.build_rag_chain(retriever, llm, self.get_prompt_template())
        self.run_rag_query(rag_chain)


if __name__ == "__main__":
    MainWorkflow().run()
