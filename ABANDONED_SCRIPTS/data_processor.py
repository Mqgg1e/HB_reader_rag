import os
import re
from typing import List, Dict, Any
import json

def read_text_file(file_path: str) -> str:
    """
    Reads the content of a text file from the specified path.

    Args:
        file_path (str): The full path to the text file.

    Returns:
        str: The complete content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If any other I/O error occurs during file reading.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        raise FileNotFoundError(f"File not found at: {file_path}")

    try:
        # Using 'utf-8' encoding is a best practice for text files, especially for diverse content.
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        print(f"Successfully read file: '{file_path}'")
        return text_content
    except IOError as e:
        print(f"Error: An I/O error occurred while reading file '{file_path}': {e}")
        raise


# --- Usage Example ---
# IMPORTANT: Replace 'path/to/your/english_bible.txt' with the actual path to your Bible TXT file.
# For instance:
# bible_file_path = "/Users/yourname/Documents/english_bible.txt"
# bible_file_path = "data/english_bible.txt" # If the file is in a 'data' subdirectory in your project

bible_file_path = "../data/raw_data/AKJV.txt"  # Placeholder
processed_bible_file_path = "../data/processed_data"  # Placeholder for output

#
# try:
#     raw_bible_text = read_text_file(bible_file_path)
#     # Print the first 500 characters to verify successful reading
#     print("\n--- File Content Preview (First 500 characters) ---")
#     print(raw_bible_text[:500])
#     print("--------------------------------------------------")
#     # Also print the total character count
#     print(f"Total characters read: {len(raw_bible_text)}")
#
# except (FileNotFoundError, IOError):
#     # The error message is already printed by the function, so we just pass here.
#     pass

# Assuming read_text_file_robustly is in the same module or imported
# (Previous code for read_text_file_robustly not repeated here for brevity,
# but it should be available in your data_processor.py)

def clean_bible_text_header(raw_text: str) -> str:
    """
    Cleans the header section from the raw Bible text based on delimiters.
    Assumes the header is delimited by '--------------------------------------------------------------------------------'
    and everything after the second delimiter is the actual Bible content.

    Args:
        raw_text (str): The raw text content of the Bible file.

    Returns:
        str: The cleaned text with the header removed.
    """
    delimiter = "--------------------------------------------------------------------------------"

    # Split the text by the delimiter
    parts = raw_text.split(delimiter)

    # The actual Bible content starts after the *second* delimiter.
    # parts[0] = header before 1st delimiter
    # parts[1] = content between 1st and 2nd delimiter
    # parts[2] = Bible content after 2nd delimiter
    if len(parts) >= 3:
        cleaned_text = parts[2].strip()
    else:
        # If the delimiter isn't found as expected, return original text or handle as error
        print("Warning: Delimiter pattern not found as expected. Returning original text.")
        return raw_text

    # Further cleanup: remove any leading blank lines that might remain at the very beginning
    # This regex removes one or more whitespace characters followed by a newline at the start of a string (multiline mode)
    cleaned_text = re.sub(r'^\s*\n', '', cleaned_text, flags=re.MULTILINE)

    return cleaned_text


#A function to save the cleaned text to a file
def save_cleaned_text_to_file(cleaned_text: str, output_file_path: str) -> None:
    """
    Saves the cleaned Bible text to a specified output file.

    Args:
        cleaned_text (str): The cleaned Bible text to save.
        output_file_path (str): The path where the cleaned text will be saved.

    Raises:
        IOError: If an error occurs while writing to the file.
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"Cleaned text successfully saved to: '{output_file_path}'")
    except IOError as e:
        print(f"Error: An I/O error occurred while writing to file '{output_file_path}': {e}")
        raise


# Assuming Document class is available from langchain_core.documents or similar for structured data
# If you don't use LangChain's Document, you can use a simple dict for each chunk.
class Document:
    """A simplified Document class to hold content and metadata."""

    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

    def to_dict(self):
        return {
            "page_content": self.page_content,
            "metadata": self.metadata
        }

def parse_bible_verses(cleaned_text: str) -> List[Document]:
    """
    Parses the cleaned Bible text, extracts verses and their precise location metadata.

    Args:
        cleaned_text (str): The Bible text after header cleaning.

    Returns:
        List[Document]: A list of Document objects, each representing a verse
                        with content and metadata (book_full, book_abbr, chapter, verse).
    """
    verses = []
    current_book_full_name = None

    # Regex to match verse pattern: e.g., "Gen.1:1 "
    # Group 1: Book Abbreviation (e.g., Gen)
    # Group 2: Chapter Number (e.g., 1)
    # Group 3: Verse Number (e.g., 1)
    # Group 4: The rest of the verse content
    # verse_pattern = re.compile(r'([A-Za-z]+)\.(\d+):(\d+)\s(.*)')
    verse_pattern = re.compile(r'(\d?[A-Za-z]+)\.(\d+):(\d+)\s(.*)')

    # A simple mapping for common abbreviations to full book names
    # You might need to expand this for all 66 books if your file uses abbreviations
    # A more robust solution would map all 66 canonical abbreviations
    # book_abbr_to_full_name = {
    #     "Gen": "Genesis", "Exo": "Exodus", "Lev": "Leviticus", "Num": "Numbers", "Deu": "Deuteronomy",
    #     "Jos": "Joshua", "Jdg": "Judges", "Rut": "Ruth", "1Sa": "1 Samuel", "2Sa": "2 Samuel",
    #     "1Ki": "1 Kings", "2Ki": "2 Kings", "1Ch": "1 Chronicles", "2Ch": "2 Chronicles",
    #     "Ezr": "Ezra", "Neh": "Nehemiah", "Est": "Esther", "Job": "Job", "Psa": "Psalms",
    #     "Pro": "Proverbs", "Ecc": "Ecclesiastes", "Son": "Song of Solomon", "Isa": "Isaiah",
    #     "Jer": "Jeremiah", "Lam": "Lamentations", "Eze": "Ezekiel", "Dan": "Daniel",
    #     "Hos": "Hosea", "Joe": "Joel", "Amo": "Amos", "Oba": "Obadiah", "Jon": "Jonah",
    #     "Mic": "Micah", "Nah": "Nahum", "Hab": "Habakkuk", "Zep": "Zephaniah", "Hag": "Haggai",
    #     "Zec": "Zechariah", "Mal": "Malachi", "Mat": "Matthew", "Mar": "Mark", "Luk": "Luke",
    #     "Joh": "John", "Act": "Acts", "Rom": "Romans", "1Co": "1 Corinthians", "2Co": "2 Corinthians",
    #     "Gal": "Galatians", "Eph": "Ephesians", "Phi": "Philippians", "Col": "Colossians",
    #     "1Th": "1 Thessalonians", "2Th": "2 Thessalonians", "1Ti": "1 Timothy", "2Ti": "2 Timothy",
    #     "Tit": "Titus", "Phm": "Philemon", "Heb": "Hebrews", "Jam": "James", "1Pe": "1 Peter",
    #     "2Pe": "2 Peter", "1Jo": "1 John", "2Jo": "2 John", "3Jo": "3 John", "Jud": "Jude",
    #     "Rev": "Revelation"
    # }

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
            # Skip empty lines
            continue

        # Check if the line is a book title (e.g., "Genesis")
        # Assuming book titles are usually short and don't match verse patterns
        if len(line.split()) <= 3 and not verse_pattern.match(line) and not re.match(r'^\d+:', line):  # Simple heuristic
            # Update the current full book name if a new book title is found
            current_book_full_name = line
            continue  # Move to the next line

        # Try to match the verse pattern
        match = verse_pattern.match(line)
        if match and current_book_full_name:
            book_abbr = match.group(1)
            chapter = match.group(2)
            verse_num = match.group(3)
            verse_content = match.group(4).strip()  # The actual text of the verse

            # Map abbreviation to full name, or use current_book_full_name if not found in map
            # This handles cases where the abbr might not be in our map, but full name is known
            book_display_name = book_abbr_to_full_name.get(book_abbr, current_book_full_name)

            metadata = {
                "book_full_name": book_display_name,
                "book_abbreviation": book_abbr,
                "chapter": int(chapter),  # Convert to int for easier sorting/filtering later
                "verse": int(verse_num),  # Convert to int
                "location": f"{book_display_name} {chapter}:{verse_num}"  # A human-readable location string
            }
            verses.append(Document(page_content=verse_content, metadata=metadata))
        elif match:
            # This case means a verse pattern was found, but we haven't identified a book name yet.
            # This could happen if the book name is very close to the first verse without a clear break,
            # or if the first book name was missed.
            print(f"Warning: Verse found without preceding book name. Line: {line}")
        else:
            # This is a line that is neither a book title nor a verse.
            # Could be a footer or an anomaly not handled by initial cleaning.
            # For Bible, generally expect only book titles or verses after header.
            print(f"Info: Skipping unrecognized line (not book title or verse): {line[:100]}...")

    print(f"Finished parsing. Total verses extracted: {len(verses)}")
    return verses


def save_documents_to_jsonl(documents: List[Document], output_file_path: str):
    """
    Saves a list of Document objects to a JSONL file.

    Args:
        documents (List[Document]): The list of Document objects to save.
        output_file_path (str): The path to the output JSONL file.
                                 Example: "processed_data/bible_verses.jsonl"
    """
    # Ensure the directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                # Convert Document object to a dictionary for JSON serialization
                json_line = json.dumps(doc.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')
        print(f"Successfully saved {len(documents)} documents to '{output_file_path}'")
    except IOError as e:
        print(f"Error: Could not save documents to '{output_file_path}': {e}")
        raise

# --- Function to Load Data (for completeness) ---

def load_documents_from_jsonl(input_file_path: str) -> List[Document]:
    """
    Loads a list of Document objects from a JSONL file.

    Args:
        input_file_path (str): The path to the input JSONL file.

    Returns:
        List[Document]: A list of loaded Document objects.
    """
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


# # bible_raw_txt_file = os.path.join("../data/raw_data", "AKJV.txt")
# processed_jsonl_file = os.path.join("../data/processed_data", "akjv_verses.jsonl")
#
# raw_bible_text = read_text_file(bible_file_path)
# #
# cleaned_bible_text = clean_bible_text_header(raw_bible_text)
# #
# # save_cleaned_text_to_file(cleaned_bible_text, os.path.join(processed_bible_file_path, "cleaned_bible.txt"))
# save_documents_to_jsonl(
#     parse_bible_verses(cleaned_bible_text),
#     processed_jsonl_file
# )
# # print(cleaned_bible_text[:500])
