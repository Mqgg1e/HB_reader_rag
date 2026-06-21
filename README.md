# HB_reader_rag

A simple workflow for preparing and querying English Bible text.

This repository processes the King James Version Bible source text, converts it into structured verse documents, saves the cleaned data in JSONL format, and builds a FAISS index for fast semantic retrieval.

## What it does

- reads the raw Bible text file from `data/raw_data/AKJV.txt`
- cleans the file header and removes non-text artifacts
- parses Bible verses into document records with metadata
- stores the cleaned Bible text and structured verse data under `data/processed_data`
- builds a FAISS vector index under `faiss_index`
- loads the index and runs a retrieval-based query workflow

## Repository structure

- `main.py` - orchestrates the full workflow from text ingestion to query execution
- `base_workflow.py` - common workflow class that loads config and exposes processing functions
- `config/Config.yaml` - project settings and file paths
- `data/raw_data/AKJV.txt` - raw English Bible source text
- `data/processed_data/` - output directory for cleaned text and JSONL documents
- `faiss_index/` - output directory for the saved index files
- `src/modules/` - implementation modules for text processing, vector database handling, and query chaining
- `ABANDONED_SCRIPTS/` - older scripts and experimental work

## Key files

- `src/modules/data_processing.py` - utilities for reading, cleaning, parsing, and storing Bible text
- `src/modules/vector_db.py` - functions for building and saving the FAISS vector database
- `src/modules/rag_chain.py` - query workflow helpers and index loading logic

## Getting started

1. Make sure you have Python installed.
2. Install the required Python packages for YAML parsing, FAISS, and model/runtime support.
3. Update `config/Config.yaml` if your file paths or settings differ from the defaults.
4. Run:

```bash
python main.py
```

## Configuration

The main configuration file is `config/Config.yaml`. It includes settings for:

- `bible_file_path` - path to the raw Bible text file
- `processed_bible_file_path` - directory to save cleaned text and processed outputs
- `processed_jsonl_file` - JSONL file for parsed verse documents
- `faiss_index_path` - folder where the FAISS index is stored
- `faiss_index_name` - name used for the saved index
- `top_k` - number of nearest results to retrieve during queries

## Usage

Run the script from the repository root:

```bash
python main.py
```

The script performs all main steps in sequence:

1. read and clean the raw Bible text
2. save the cleaned text to `data/processed_data/cleaned_bible.txt`
3. parse the text into verse documents
4. save verse documents to `data/processed_data/akjv_verses.jsonl`
5. build and save a FAISS vector database under `faiss_index`
6. load the saved index and run a retrieval query loop

## Notes

- The project is organized as a modular pipeline, so each step can be inspected and updated independently.
- The `config/Config.yaml` file is the main place to adjust paths and runtime options.
- The `src/modules` package contains the implementation details used by `main.py` and `base_workflow.py`.


For direct experiment, you may visit this notebook: https://www.kaggle.com/code/maedaky/hbreader