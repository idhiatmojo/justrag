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

    def _get_collection(self, dim: int):
        """Dynamically routes to the collection matching the embedding dimension.
        - 3072: Gemini ('curriculum_rag_docs')
        - 384:  Local Fallback / all-MiniLM ('curriculum_rag_docs_local_384')
        - 1536: OpenAI ('curriculum_rag_docs_openai_1536')
        """
        if dim == 3072:
            col_name = "curriculum_rag_docs"
        elif dim == 384:
            col_name = "curriculum_rag_docs_local_384"
        elif dim == 1536:
            col_name = "curriculum_rag_docs_openai_1536"
        else:
            col_name = f"curriculum_rag_docs_dim_{dim}"

        return self.client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embedder: BaseLLMClient):
        """
        Embeds and upserts chunks into ChromaDB with metadata.
        Auto-routes to the dimension-matched collection.
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

        # Generate embeddings — batching and rate-limit retry are handled inside embed_texts()
        all_embeddings = embedder.embed_texts(documents)
        if not all_embeddings:
            return 0

        dim = len(all_embeddings[0])
        target_collection = self._get_collection(dim)

        target_collection.upsert(
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
        Auto-routes query to the collection corresponding to the embedder's dimension.
        """
        query_emb = embedder.embed_texts([query])[0]
        dim = len(query_emb)
        target_collection = self._get_collection(dim)
        
        where_filter = None
        if subject and subject.strip():
            where_filter = {"subject": subject.lower().strip()}

        results = target_collection.query(
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

    def get_stats_by_subject(self, embedder: Optional[BaseLLMClient] = None, *args, **kwargs) -> Dict[str, int]:
        """Returns document counts per subject for the active embedder collection."""
        try:
            if embedder is not None:
                sample = embedder.embed_texts(["probe"])
                dim = len(sample[0]) if sample else 3072
                collection = self._get_collection(dim)
            else:
                collection = self.collection

            data = collection.get(include=["metadatas"])
            counts = {}
            if data and "metadatas" in data and data["metadatas"]:
                for m in data["metadatas"]:
                    sub = m.get("subject", "unknown")
                    counts[sub] = counts.get(sub, 0) + 1
            return counts
        except Exception:
            return {}

    def clear_subject(self, subject: str):
        """Removes all chunks belonging to a specific subject across all dimension collections."""
        sub_filter = {"subject": subject.lower().strip()}
        for col in self.client.list_collections():
            try:
                col.delete(where=sub_filter)
            except Exception:
                pass
