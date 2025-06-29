from typing import Dict, Any


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
