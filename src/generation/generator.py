from typing import List, Dict, Any, Optional
from src.generation.prompts import (
    SYSTEM_PEDAGOGY_PROMPT,
    MODUL_AJAR_TEMPLATE,
    SOAL_HOTS_TEMPLATE,
    RINGKASAN_MATERI_TEMPLATE,
    QA_CHAT_TEMPLATE
)
from src.llm_factory import BaseLLMClient
from src.subject_manager import load_subjects

class PedagogicalGenerator:
    """Orchestrates generation of educational assets conditioned on retrieved context."""
    
    TEMPLATES = {
        "modul_ajar": MODUL_AJAR_TEMPLATE,
        "soal_hots": SOAL_HOTS_TEMPLATE,
        "ringkasan": RINGKASAN_MATERI_TEMPLATE,
        "qa_chat": QA_CHAT_TEMPLATE
    }

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats chunks into a clear referenced text block for LLM prompt."""
        if not retrieved_chunks:
            return "TIDAK ADA KONTEKS DOKUMEN YANG RELEVAN. Mohon ingatkan pengguna untuk mengunggah materi terkait."

        blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk.get("metadata", {})
            src = meta.get("source_file", "Dokumen")
            page = meta.get("page_number", "-")
            text = chunk.get("text", "").strip()
            blocks.append(f"--- [Kutipan {i}] (Sumber: {src}, Hal. {page}) ---\n{text}")

        return "\n\n".join(blocks)

    def generate(
        self,
        query: str,
        subject: str,
        retrieved_chunks: List[Dict[str, Any]],
        task_type: str = "qa_chat",
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Executes generation using the chosen template and retrieved context.
        """
        registered_subjects = load_subjects()
        sub_info = registered_subjects.get(subject.lower().strip(), {
            "display_name": subject,
            "description": ""
        })
        subject_display = sub_info["display_name"]
        
        template = self.TEMPLATES.get(task_type, QA_CHAT_TEMPLATE)
        formatted_context = self.format_context(retrieved_chunks)

        prompt = template.format(
            subject_display=subject_display,
            user_query=query,
            retrieved_context=formatted_context
        )

        response_text = self.llm_client.generate_text(
            prompt=prompt,
            system_instruction=SYSTEM_PEDAGOGY_PROMPT,
            temperature=temperature
        )

        # Extract citations
        sources = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            sources.append({
                "source_file": meta.get("source_file", "Unknown"),
                "page_number": meta.get("page_number", "-"),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0.0)))
            })

        return {
            "response": response_text,
            "sources": sources,
            "subject": subject,
            "task_type": task_type,
            "context_chunks_used": len(retrieved_chunks)
        }
