import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.config import BM25_PERSIST_DIR

class BM25SubjectStore:
    """Sparse Lexical Inverted Index using BM25, partitioned and stored per subject."""
    
    def __init__(self, persist_dir: str = str(BM25_PERSIST_DIR)):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.indices: Dict[str, BM25Okapi] = {}
        self.chunk_stores: Dict[str, List[Dict[str, Any]]] = {}
        self._load_all()

    def _tokenize(self, text: str) -> List[str]:
        """Case-insensitive alphanumeric tokenization."""
        tokens = re.findall(r'\b[a-zA-Z0-9_\-]{2,}\b', text.lower())
        return tokens

    def _get_path(self, subject: str) -> Path:
        return self.persist_dir / f"bm25_{subject.lower().strip()}.pkl"

    def _load_all(self):
        for pkl_file in self.persist_dir.glob("bm25_*.pkl"):
            sub = pkl_file.stem.replace("bm25_", "")
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                    self.indices[sub] = data["bm25"]
                    self.chunk_stores[sub] = data["chunks"]
            except Exception as e:
                print(f"[Warning] Gagal membaca cache BM25 {pkl_file}: {e}")

    def update_subject(self, subject: str, chunks: List[Dict[str, Any]]):
        """Rebuilds or adds to BM25 index for a given subject."""
        sub = subject.lower().strip()
        existing = self.chunk_stores.get(sub, [])
        
        # Merge by unique chunk_id
        chunk_map = {c["chunk_id"]: c for c in existing}
        for c in chunks:
            chunk_map[c["chunk_id"]] = c
        
        merged_chunks = list(chunk_map.values())
        corpus = [c["text"] for c in merged_chunks]
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        if tokenized_corpus:
            bm25 = BM25Okapi(tokenized_corpus)
            self.indices[sub] = bm25
            self.chunk_stores[sub] = merged_chunks
            
            # Persist to disk
            with open(self._get_path(sub), "wb") as f:
                pickle.dump({"bm25": bm25, "chunks": merged_chunks}, f)

    def search(self, query: str, subject: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches BM25 index. If subject is provided, searches that subject's index."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        targets = [subject.lower().strip()] if (subject and subject.strip() in self.indices) else list(self.indices.keys())
        all_hits = []

        for sub in targets:
            bm25 = self.indices.get(sub)
            chunks = self.chunk_stores.get(sub, [])
            if not bm25 or not chunks:
                continue

            scores = bm25.get_scores(query_tokens)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

            for idx in ranked_indices:
                score = float(scores[idx])
                if score > 0.0:
                    c = chunks[idx]
                    all_hits.append({
                        "chunk_id": c["chunk_id"],
                        "text": c["text"],
                        "metadata": {
                            "subject": c["subject"],
                            "grade_level": c["grade_level"],
                            "source_file": c["source_file"],
                            "page_number": int(c["page_number"])
                        },
                        "score": score,
                        "retrieval_type": "sparse_bm25"
                    })

        # Sort combined results by score and take top_k
        all_hits.sort(key=lambda x: x["score"], reverse=True)
        return all_hits[:top_k]
