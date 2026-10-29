import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import RAW_PDF_DIR, ALLOWED_SUBJECTS
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import ContextualChunker
from src.storage.vector_store import ChromaVectorStore
from src.storage.bm25_store import BM25SubjectStore
from src.llm_factory import BaseLLMClient

class DynamicIngestionPipeline:
    """End-to-end dynamic ingestion manager for curriculum PDFs."""
    
    def __init__(self, vector_store: Optional[ChromaVectorStore] = None, bm25_store: Optional[BM25SubjectStore] = None):
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_store = bm25_store or BM25SubjectStore()
        self.chunker = ContextualChunker()

    def ingest_single_pdf(
        self,
        pdf_path: str | Path,
        subject: str,
        embedder: BaseLLMClient,
        grade_level: str = "Kelas X"
    ) -> Dict[str, Any]:
        """Ingests a single PDF file dynamically into vector and lexical indexes."""
        subject = subject.lower().strip()
        if subject not in ALLOWED_SUBJECTS:
            raise ValueError(f"Subject '{subject}' tidak terdaftar. Pilihan: {list(ALLOWED_SUBJECTS.keys())}")

        parser = PDFParser(subject=subject, grade_level=grade_level)
        pages = parser.parse_pdf(pdf_path)
        if not pages:
            return {"status": "empty", "pages": 0, "chunks": 0, "file": Path(pdf_path).name}

        chunks = self.chunker.process_pages(pages)

        # 1. Upsert into Dense Vector Store (ChromaDB)
        inserted_dense = self.vector_store.upsert_chunks(chunks, embedder=embedder)

        # 2. Update BM25 Sparse Index
        self.bm25_store.update_subject(subject, chunks)

        return {
            "status": "success",
            "subject": subject,
            "file": Path(pdf_path).name,
            "pages": len(pages),
            "chunks": len(chunks),
            "inserted_dense": inserted_dense
        }

    def ingest_all_in_folder(self, embedder: BaseLLMClient) -> List[Dict[str, Any]]:
        """Scans data/raw_pdf/ across all subjects and ingests new documents."""
        results = []
        for sub_key in ALLOWED_SUBJECTS.keys():
            sub_folder = RAW_PDF_DIR / sub_key
            if not sub_folder.exists():
                continue
            for pdf_file in sub_folder.glob("*.pdf"):
                res = self.ingest_single_pdf(pdf_file, subject=sub_key, embedder=embedder)
                results.append(res)
        return results
