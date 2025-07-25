from typing import List
from src.modules.document import Document


# --- New Function to Chunk by Chapter ---

def chunk_bible_by_chapter(verse_documents: List[Document]) -> List[Document]:
    """
    Chunks the list of verse documents into larger documents, each representing a full chapter.

    Args:
        verse_documents (List[Document]): A list of Document objects, where each represents a single verse.

    Returns:
        List[Document]: A list of new Document objects, where each represents a chapter,
                        with combined content and updated metadata.
    """
    chapter_chunks = []
    current_chapter_content = []
    current_chapter_metadata = {}

    # Track the last processed book and chapter to detect changes
    last_book_full_name = None
    last_chapter_num = None

    for i, doc in enumerate(verse_documents):
        # Extract metadata for the current verse
        book_full_name = doc.metadata.get("book_full_name")
        book_abbreviation = doc.metadata.get("book_abbreviation")
        chapter_num = doc.metadata.get("chapter")
        verse_num = doc.metadata.get("verse")  # Keep verse_num for range in metadata if needed later

        # Ensure we have valid metadata to avoid errors
        if book_full_name is None or chapter_num is None:
            print(f"Warning: Skipping verse due to missing essential metadata: {doc.metadata}")
            continue

        # Check if we've moved to a new chapter or a new book
        # If it's the very first verse, or if book/chapter has changed
        if (book_full_name != last_book_full_name) or (chapter_num != last_chapter_num):
            # If we have accumulated content for a previous chapter, save it as a chunk
            if current_chapter_content:  # Only save if content exists (not for the very first iteration)
                full_chapter_content = "\n".join(current_chapter_content)
                # Ensure start and end verses are accurate for the stored chapter
                # This needs careful tracking; for simplicity, we'll store chapter-level metadata here

                # Update location to be chapter-based
                current_chapter_metadata["location"] = \
                    f"{current_chapter_metadata['book_full_name']} {current_chapter_metadata['chapter']}"

                chapter_chunks.append(Document(
                    page_content=full_chapter_content,
                    metadata=current_chapter_metadata
                ))

            # Reset for the new chapter
            current_chapter_content = []
            # Copy relevant metadata from the current verse for the new chapter chunk
            current_chapter_metadata = {
                "book_full_name": book_full_name,
                "book_abbreviation": book_abbreviation,
                "chapter": chapter_num,
                # For chapter chunks, 'verse' is less meaningful.
                # You might add 'start_verse' and 'end_verse' if desired.
                # "start_verse": verse_num # You could add logic to track first and last verse for the chapter
            }
            last_book_full_name = book_full_name
            last_chapter_num = chapter_num

        # Add the current verse's content to the current chapter's content list
        # Prepend with verse number for context within the chapter chunk, if desired
        current_chapter_content.append(f"({verse_num}) {doc.page_content}")

    # After the loop, save the very last accumulated chapter chunk
    if current_chapter_content:
        full_chapter_content = "\n".join(current_chapter_content)
        current_chapter_metadata["location"] = \
            f"{current_chapter_metadata['book_full_name']} {current_chapter_metadata['chapter']}"
        chapter_chunks.append(Document(
            page_content=full_chapter_content,
            metadata=current_chapter_metadata
        ))

    print(f"Finished chunking by chapter. Total chapter chunks created: {len(chapter_chunks)}")
    return chapter_chunks


# # --- End-to-End Usage Example with Chapter Chunking ---
#
#
# # 1. Define file paths
# bible_raw_txt_file = os.path.join("data", "raw_data", "AKJV.txt")  # Assuming your raw TXT is here
# # processed_verses_jsonl_file = os.path.join("data", "processed_data", "akjv_verses.jsonl")
# processed_verses_jsonl_file = "../data/processed_data/akjv_verses.jsonl"
# processed_chapters_jsonl_file = "../data/processed_data/akjv_chapters.jsonl"
#
# # For demonstration, we'll use a simulated raw text content
# # This `sample_raw_bible_text` should reflect content after header cleaning for parse_bible_verses
# sample_cleaned_bible_text_for_demo = """
# Genesis
#
# Gen.1:1 In the beginning God created the heaven and the earth.
# Gen.1:2 And the earth was without form, and void; and darkness was on the face of the deep. And the Spirit of God moved on the face of the waters.
# Gen.1:3 And God said, Let there be light: and there was light.
# Gen.1:4 And God saw the light, that it was good: and God divided the light from the darkness.
# Gen.1:5 And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day.
#
# Gen.2:1 Thus the heavens and the earth were finished, and all the host of them.
# Gen.2:2 And on the seventh day God ended his work which he had made; and he rested on the seventh day from all his work which he had made.
#
# Exodus
#
# Exo.1:1 Now these are the names of the children of Israel, which came into Egypt with Jacob: every man and his household came with them.
# Exo.1:2 Reuben, Simeon, Levi, and Judah, Issachar, Zebulun, and Benjamin,
# Exo.1:3 Dan, and Naphtali, Gad, and Asher.
# """
#
# from modules.data_processing import load_documents_from_jsonl
# from modules.data_processing import save_documents_to_jsonl
#
# try:
#     # # --- Part 1: Initial Processing (Verse Level) ---
#     # # raw_bible_text_content = read_text_file_robustly(bible_raw_txt_file, relative_to_script=True)
#     # # cleaned_text = clean_bible_text_header(raw_bible_text_content)
#     # # Using the sample for demo:
#     # cleaned_text = sample_cleaned_bible_text_for_demo
#     #
#     # parsed_verse_documents = parse_bible_verses(cleaned_text)
#     # print("\n--- Verses Parsed ---")
#     # print(
#     #     f"Example verse: {parsed_verse_documents[0].metadata['location']} - {parsed_verse_documents[0].page_content[:50]}...")
#     #
#     # # Save parsed verses (optional, but good for intermediate storage)
#     # save_documents_to_jsonl(parsed_verse_documents, processed_verses_jsonl_file)
#     # print("\n--- Parsed Verses Saved ---")
#
#     # --- Part 2: Chunking by Chapter ---
#     parsed_verse_documents = load_documents_from_jsonl(processed_verses_jsonl_file)
#     chapter_documents = chunk_bible_by_chapter(parsed_verse_documents)
#     print("\n--- Chapters Chunked ---")
#     print(
#         f"Example chapter: {chapter_documents[0].metadata['location']} - {chapter_documents[0].page_content[:100]}...")
#
#     # Save chunked chapters
#     save_documents_to_jsonl(chapter_documents, processed_chapters_jsonl_file)
#     print("\n--- Chapter Chunks Saved ---")
#
#     # # (Optional) Load the chapter documents back to verify
#     # loaded_chapter_docs = load_documents_from_jsonl(processed_chapters_jsonl_file)
#     # print("\n--- Chapter Chunks Loaded ---")
#     # print(f"Loaded {len(loaded_chapter_docs)} chapter documents. First loaded chapter: {loaded_chapter_docs[0]}")
#
# except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
#     print(f"An error occurred during the workflow: {e}")
