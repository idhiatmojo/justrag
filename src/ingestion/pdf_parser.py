import os
import re
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

class PDFParser:
    """Extracts raw text and metadata page by page from curriculum PDF files."""
    
    def __init__(self, subject: str, grade_level: str = "Kelas X"):
        self.subject = subject.lower().strip()
        self.grade_level = grade_level

    def clean_text(self, text: str) -> str:
        """Clean redundant whitespaces and common artifact characters."""
        if not text:
            return ""
        # Normalize non-breaking spaces and redundant whitespaces
        text = text.replace('\xa0', ' ')
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize excessive newlines (keep max 2 for paragraph separation)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def parse_pdf(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """
        Parses a PDF file and returns a list of page records with metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file tidak ditemukan: {file_path}")
            
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        pages_data = []

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            cleaned = self.clean_text(raw_text)
            
            if len(cleaned) < 20:
                # Skip virtually empty pages (cover blank pages, blank separators)
                continue
                
            pages_data.append({
                "source_file": path.name,
                "file_path": str(path.resolve()),
                "subject": self.subject,
                "grade_level": self.grade_level,
                "page_number": page_num,
                "total_pages": total_pages,
                "text": cleaned
            })

        return pages_data
