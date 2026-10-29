import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.config import CHROMA_PERSIST_DIR
from src.llm_factory import BaseLLMClient

class ChromaVectorStore:
    """Persistent ChromaDB vector store supporting dynamic document addition and metadata filtering."""
    
    COLLECTION_NAME = "curriculum_rag_docs"
    
    def __init__(self, persist_dir: str = str(CHROMA_PERSIST_DIR)):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embedder: BaseLLMClient):
        """
        Embeds and upserts chunks into ChromaDB with metadata.
        """
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "subject": c["subject"],
                "grade_level": c["grade_level"],
                "source_file": c["source_file"],
                "page_number": int(c["page_number"])
            }
            for c in chunks
        ]

        # Generate embeddings in batches of 16 to respect API rate limits
        batch_size = 16
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch_texts = documents[i:i + batch_size]
            batch_embs = embedder.embed_texts(batch_texts)
            all_embeddings.extend(batch_embs)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=all_embeddings,
            metadatas=metadatas
        )
        return len(ids)

    def search(
        self,
        query: str,
        embedder: BaseLLMClient,
        subject: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Queries the vector store. If subject is provided, applies metadata filtering.
        """
        query_emb = embedder.embed_texts([query])[0]
        
        where_filter = None
        if subject and subject.strip():
            where_filter = {"subject": subject.lower().strip()}

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                doc_text = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                # Cosine distance to similarity: similarity = 1 - distance
                similarity = 1.0 - float(distance) if distance is not None else 0.0

                formatted_results.append({
                    "chunk_id": doc_id,
                    "text": doc_text,
                    "metadata": metadata,
                    "score": similarity,
                    "retrieval_type": "dense"
                })

        return formatted_results

    def get_stats_by_subject(self) -> Dict[str, int]:
        """Returns document counts per subject."""
        try:
            data = self.collection.get(include=["metadatas"])
            counts = {}
            if data and "metadatas" in data and data["metadatas"]:
                for m in data["metadatas"]:
                    sub = m.get("subject", "unknown")
                    counts[sub] = counts.get(sub, 0) + 1
            return counts
        except Exception:
            return {}

    def clear_subject(self, subject: str):
        """Removes all chunks belonging to a specific subject."""
        self.collection.delete(where={"subject": subject.lower().strip()})
