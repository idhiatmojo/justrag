from typing import List, Dict, Any, Optional
from src.storage.vector_store import ChromaVectorStore
from src.storage.bm25_store import BM25SubjectStore
from src.llm_factory import BaseLLMClient
from src.config import RRF_K, TOP_K_RETRIEVAL

class HybridRetriever:
    """
    Orchestrates dense (vector) and sparse (BM25) search with Reciprocal Rank Fusion (RRF).
    Supports toggling filtering and modes for rigorous ablation experiments.
    """
    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_store: BM25SubjectStore,
        embedder: BaseLLMClient
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        rrf_k: int = RRF_K
    ) -> List[Dict[str, Any]]:
        """
        Combines ranked lists from dense and sparse retrieval using RRF algorithm.
        Score(d) = sum_{m in {dense, bm25}} 1 / (k + rank_m(d))
        """
        fused_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict[str, Any]] = {}

        # Dense ranking
        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            doc_store[cid] = item
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # BM25 ranking
        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            if cid not in doc_store:
                doc_store[cid] = item
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort combined documents by fused score
        sorted_cids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        fused_results = []
        for cid in sorted_cids:
            doc_item = dict(doc_store[cid])
            doc_item["rrf_score"] = fused_scores[cid]
            doc_item["retrieval_type"] = "hybrid_rrf"
            fused_results.append(doc_item)

        return fused_results

    def retrieve(
        self,
        query: str,
        subject: Optional[str] = None,
        mode: str = "hybrid",
        use_subject_filter: bool = True,
        top_k: int = TOP_K_RETRIEVAL
    ) -> List[Dict[str, Any]]:
        """
        Retrieves candidate chunks based on mode: 'hybrid', 'dense_only', or 'bm25_only'.
        """
        active_subject = subject if use_subject_filter else None
        
        # 1. Dense Search
        dense_hits = []
        if mode in ["hybrid", "dense_only"]:
            dense_hits = self.vector_store.search(
                query=query,
                embedder=self.embedder,
                subject=active_subject,
                top_k=top_k * 2
            )

        # 2. BM25 Sparse Search
        bm25_hits = []
        if mode in ["hybrid", "bm25_only"]:
            bm25_hits = self.bm25_store.search(
                query=query,
                subject=active_subject,
                top_k=top_k * 2
            )

        # 3. Fusion or Selection
        if mode == "hybrid":
            fused = self.reciprocal_rank_fusion(dense_hits, bm25_hits)
            return fused[:top_k]
        elif mode == "dense_only":
            return dense_hits[:top_k]
        elif mode == "bm25_only":
            return bm25_hits[:top_k]
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
