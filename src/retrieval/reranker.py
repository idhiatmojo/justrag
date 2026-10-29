import re
from typing import List, Dict, Any

class LexicalSemanticReranker:
    """
    Reranker that re-scores retrieved candidate chunks using query term overlap,
    density of key entities, and positional context signals.
    """
    
    def __init__(self):
        pass

    def _extract_keywords(self, text: str) -> set:
        words = re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', text.lower())
        stopwords = {"dan", "atau", "yang", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "adalah", "the", "and", "or", "in", "on", "at", "to", "for", "with", "is", "are"}
        return {w for w in words if w not in stopwords}

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidates and returns the top_k most relevant items.
        """
        if not candidates:
            return []

        query_kw = self._extract_keywords(query)
        scored_candidates = []

        for item in candidates:
            text = item["text"].lower()
            text_kw = self._extract_keywords(text)

            # 1. Jaccard token overlap
            if query_kw:
                overlap = len(query_kw.intersection(text_kw)) / len(query_kw)
            else:
                overlap = 0.0

            # 2. Existing retrieval score (dense similarity or RRF score)
            base_score = item.get("rrf_score", item.get("score", 0.0))

            # Combined weighted score
            final_score = (0.6 * base_score) + (0.4 * overlap)
            
            scored_item = dict(item)
            scored_item["rerank_score"] = round(final_score, 4)
            scored_candidates.append(scored_item)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]
