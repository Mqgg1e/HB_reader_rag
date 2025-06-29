import os
import re
import json
from typing import List, Dict, Any
from modules.document import Document

def read_text_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        raise FileNotFoundError(f"File not found at: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        print(f"Successfully read file: '{file_path}'")
        return text_content
    except IOError as e:
        print(f"Error: An I/O error occurred while reading file '{file_path}': {e}")
        raise

def clean_bible_text_header(raw_text: str) -> str:
    delimiter = "--------------------------------------------------------------------------------"
    parts = raw_text.split(delimiter)
    if len(parts) >= 3:
        cleaned_text = parts[2].strip()
    else:
        print("Warning: Delimiter pattern not found as expected. Returning original text.")
        return raw_text
    cleaned_text = re.sub(r'^\s*\n', '', cleaned_text, flags=re.MULTILINE)
    return cleaned_text

def save_cleaned_text_to_file(cleaned_text: str, output_file_path: str) -> None:
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"Cleaned text successfully saved to: '{output_file_path}'")
    except IOError as e:
        print(f"Error: An I/O error occurred while writing to file '{output_file_path}': {e}")
        raise

def parse_bible_verses(cleaned_text: str) -> List[Document]:
    verses = []
    current_book_full_name = None
    verse_pattern = re.compile(r'(\d?[A-Za-z]+)\.(\d+):(\d+)\s(.*)')
    book_abbr_to_full_name = {
        "Gen": "Genesis", "Exo": "Exodus", "Lev": "Leviticus", "Num": "Numbers", "Deu": "Deuteronomy",
        "Jos": "Joshua", "Jdg": "Judges", "Rut": "Ruth", "1Sam": "1 Samuel", "2Sam": "2 Samuel",
        "1Ki": "1 Kings", "2Ki": "2 Kings", "1Ch": "1 Chronicles", "2Ch": "2 Chronicles",
        "Ezr": "Ezra", "Neh": "Nehemiah", "Est": "Esther", "Job": "Job", "Psa": "Psalms",
        "Pro": "Proverbs", "Ecc": "Ecclesiastes", "Son": "Song of Solomon", "Isa": "Isaiah",
        "Jer": "Jeremiah", "Lam": "Lamentations", "Eze": "Ezekiel", "Dan": "Daniel",
        "Hos": "Hosea", "Joe": "Joel", "Amo": "Amos", "Oba": "Obadiah", "Jon": "Jonah",
        "Mic": "Micah", "Nah": "Nahum", "Hab": "Habakkuk", "Zep": "Zephaniah", "Hag": "Haggai",
        "Zec": "Zechariah", "Mal": "Malachi", "Mat": "Matthew", "Mar": "Mark", "Luk": "Luke",
        "Joh": "John", "Act": "Acts", "Rom": "Romans", "1Co": "1 Corinthians", "2Co": "2 Corinthians",
        "Gal": "Galatians", "Eph": "Ephesians", "Phi": "Philippians", "Col": "Colossians",
        "1Th": "1 Thessalonians", "2Th": "2 Thessalonians", "1Ti": "1 Timothy", "2Ti": "2 Timothy",
        "Tit": "Titus", "Phm": "Philemon", "Heb": "Hebrews", "Jam": "James", "1Pe": "1 Peter",
        "2Pe": "2 Peter", "1Jo": "1 John", "2Jo": "2 John", "3Jo": "3 John", "Jud": "Jude",
        "Rev": "Revelation"
    }
    for line in cleaned_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line.split()) <= 3 and not verse_pattern.match(line) and not re.match(r'^\d+:', line):
            current_book_full_name = line
            continue
        match = verse_pattern.match(line)
        if match and current_book_full_name:
            book_abbr = match.group(1)
            chapter = match.group(2)
            verse_num = match.group(3)
            verse_content = match.group(4).strip()
            book_display_name = book_abbr_to_full_name.get(book_abbr, current_book_full_name)
            metadata = {
                "book_full_name": book_display_name,
                "book_abbreviation": book_abbr,
                "chapter": int(chapter),
                "verse": int(verse_num),
                "location": f"{book_display_name} {chapter}:{verse_num}"
            }
            verses.append(Document(page_content=verse_content, metadata=metadata))
        elif match:
            print(f"Warning: Verse found without preceding book name. Line: {line}")
        else:
            print(f"Info: Skipping unrecognized line (not book title or verse): {line[:100]}...")
    print(f"Finished parsing. Total verses extracted: {len(verses)}")
    return verses

def save_documents_to_jsonl(documents: List[Document], output_file_path: str):
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                json_line = json.dumps(doc.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')
        print(f"Successfully saved {len(documents)} documents to '{output_file_path}'")
    except IOError as e:
        print(f"Error: Could not save documents to '{output_file_path}': {e}")
        raise

def load_documents_from_jsonl(input_file_path: str) -> List[Document]:
    documents = []
    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' not found.")
        raise FileNotFoundError(f"File not found: {input_file_path}")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                doc = Document(page_content=data["page_content"], metadata=data["metadata"])
                documents.append(doc)
        print(f"Successfully loaded {len(documents)} documents from '{input_file_path}'")
        return documents
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{input_file_path}': {e}")
        raise
    except IOError as e:
        print(f"Error: Could not load documents from '{input_file_path}': {e}")
        raise

