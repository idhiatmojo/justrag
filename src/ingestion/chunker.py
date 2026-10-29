import hashlib
from typing import List, Dict, Any

class ContextualChunker:
    """
    Splits document pages into chunks while preserving sentence boundaries
    and injecting contextual headers (Contextual Retrieval approach).
    """
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = page_data["text"]
        subject = page_data["subject"]
        grade = page_data["grade_level"]
        source_file = page_data["source_file"]
        page_num = page_data["page_number"]

        # Simple sentence splitter on common delimiters
        sentences = []
        raw_sentences = text.replace('\n', ' ').split('. ')
        for s in raw_sentences:
            s_clean = s.strip()
            if s_clean:
                if not s_clean.endswith('.'):
                    s_clean += '.'
                sentences.append(s_clean)

        chunks = []
        current_sentences = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > self.chunk_size and current_sentences:
                chunk_body = " ".join(current_sentences)
                chunk_id = hashlib.md5(f"{source_file}_p{page_num}_{len(chunks)}_{chunk_body[:30]}".encode('utf-8')).hexdigest()[:16]
                
                # Contextual Prefix for retrieval enhancement
                context_prefix = f"[Mapel: {subject.upper()} | {grade} | Sumber: {source_file} (Hal. {page_num})]\n"
                full_text = context_prefix + chunk_body

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": full_text,
                    "body_only": chunk_body,
                    "subject": subject,
                    "grade_level": grade,
                    "source_file": source_file,
                    "page_number": page_num
                })

                # Maintain overlap by keeping trailing sentences
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_sentences = overlap_sentences
                current_len = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_len += sentence_len

        # Remaining sentences
        if current_sentences:
            chunk_body = " ".join(current_sentences)
            chunk_id = hashlib.md5(f"{source_file}_p{page_num}_{len(chunks)}_{chunk_body[:30]}".encode('utf-8')).hexdigest()[:16]
            context_prefix = f"[Mapel: {subject.upper()} | {grade} | Sumber: {source_file} (Hal. {page_num})]\n"
            full_text = context_prefix + chunk_body

            chunks.append({
                "chunk_id": chunk_id,
                "text": full_text,
                "body_only": chunk_body,
                "subject": subject,
                "grade_level": grade,
                "source_file": source_file,
                "page_number": page_num
            })

        return chunks

    def process_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        for page in pages:
            chunks = self.chunk_page(page)
            all_chunks.extend(chunks)
        return all_chunks
