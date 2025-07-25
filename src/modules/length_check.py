import json
import os
from typing import List, Dict, Any, Tuple


class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def to_dict(self):
        return {"page_content": self.page_content, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(page_content=data['page_content'], metadata=data['metadata'])


def load_documents_from_jsonl(file_path: str) -> List[Document]:
    """Loads a list of Document objects from a JSONL file."""
    documents = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}. Returning empty list.")
        return documents
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            documents.append(Document.from_dict(data))
    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents


def analyze_chapter_lengths(chapter_documents_path: str) -> Tuple[int, int, float, List[Dict[str, Any]]]:
    """
    Analyzes the length (in characters) of each chapter document's page_content.

    Args:
        chapter_documents_path (str): The file path to the JSONL file containing chapter Documents.

    Returns:
        Tuple[int, int, float, List[Dict[str, Any]]]:
            - max_length (int): The maximum character length found among chapters.
            - min_length (int): The minimum character length found among chapters.
            - avg_length (float): The average character length of all chapters.
            - length_details (List[Dict[str, Any]]): A list of dictionaries,
                                                      each containing 'location' and 'length' for a chapter.
                                                      Sorted by length in descending order.
    """
    print(f"\n--- Analyzing Chapter Lengths from '{chapter_documents_path}' ---")

    chapter_documents = load_documents_from_jsonl(chapter_documents_path)

    if not chapter_documents:
        print("No chapter documents found for analysis.")
        return 0, 0, 0.0, []

    lengths = []
    length_details = []

    for doc in chapter_documents:

        chapter_content_length = len(doc.page_content)
        lengths.append(chapter_content_length)


        location = doc.metadata.get('location', 'Unknown Chapter')
        length_details.append({"location": location, "length": chapter_content_length})

    if not lengths:
        return 0, 0, 0.0, []

    max_length = max(lengths)
    min_length = min(lengths)
    avg_length = sum(lengths) / len(lengths)


    length_details_sorted = sorted(length_details, key=lambda x: x['length'], reverse=True)

    print(f"\nAnalysis Results:")
    print(f"  Total chapters analyzed: {len(lengths)}")
    print(f"  Maximum chapter length: {max_length} characters")
    print(f"  Minimum chapter length: {min_length} characters")
    print(f"  Average chapter length: {avg_length:.2f} characters")

    print("\n--- Top 5 Longest Chapters ---")
    for i in range(min(5, len(length_details_sorted))):
        chap = length_details_sorted[i]
        print(f"  {chap['location']}: {chap['length']} characters")

    print("\n--- Bottom 5 Shortest Chapters ---")
    for i in range(max(0, len(length_details_sorted) - 5), len(length_details_sorted)):
        chap = length_details_sorted[i]
        print(f"  {chap['location']}: {chap['length']} characters")

    return max_length, min_length, avg_length, length_details_sorted



if __name__ == "__main__":

    mock_chapter_data_path = "../../data/processed_data/akjv_chapters.jsonl"
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")

    mock_chapters = [
        Document(page_content="This is a very short chapter.", metadata={"location": "Short.1"}),
        Document(page_content="This is a moderately long chapter with more content.",
                 metadata={"location": "Medium.1"}),
        Document(
            page_content="This is a much longer chapter. It contains a lot more sentences and paragraphs, simulating a typical long bible chapter. This text needs to be long enough to potentially exceed an LLM's context window if we are not careful about its length. We need to ensure that the content is sufficiently extensive to represent a real-world scenario where context length can become an issue. So, let's add more and more words to make it truly representative of a verbose chapter that might cause problems for a language model.",
            metadata={"location": "Long.1"}),
        Document(page_content="Shortest.", metadata={"location": "Shortest.1"}),
        Document(
            page_content="Another medium sized chapter with some additional text to make it slightly longer than the first medium one but still not very long.",
            metadata={"location": "Medium.2"}),
        Document(
            page_content="This chapter is quite lengthy and would definitely need to be chunked or summarized carefully before being fed into certain language models. The problem of context window limits is significant when dealing with documents that vary greatly in size. Ensuring that the models receive input within their operational limits is paramount for successful and efficient processing, especially for tasks like summarization where the entire context is often required.",
            metadata={"location": "VeryLong.1"})
    ]
    # save_documents_to_jsonl(mock_chapters, mock_chapter_data_path)
    # print(f"Mock chapter data saved to: {mock_chapter_data_path}")

    max_len, min_len, avg_len, details = analyze_chapter_lengths(mock_chapter_data_path)


    llm_context_limit = 1024

    print(f"\n--- Chapters Potentially Exceeding LLM Context Limit (e.g., > {llm_context_limit} chars) ---")
    exceeding_chapters = [chap for chap in details if chap['length'] > llm_context_limit]
    if exceeding_chapters:
        for chap in exceeding_chapters:
            print(
                f"  {chap['location']}: {chap['length']} characters (consider further chunking/summarization strategy)")
    else:
        print("  No chapters found exceeding the defined LLM context limit.")