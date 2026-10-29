import os
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLLMClient(ABC):
    """Abstract Base Class for LLM providers."""
    
    @abstractmethod
    def generate_text(self, prompt: str, system_instruction: Optional[str] = None, temperature: float = 0.3) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of texts."""
        pass


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini LLM & Embedding provider (uses google.genai SDK)."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY tidak ditemukan. Mohon set di .env atau via input UI.")
        
        from google import genai
        from google.genai import types as genai_types
        self._client = genai.Client(api_key=self.api_key)
        self._types = genai_types
        self.model_name = model_name
        self.embedding_model = "gemini-embedding-001"
        
    def generate_text(self, prompt: str, system_instruction: Optional[str] = None, temperature: float = 0.3) -> str:
        config = self._types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return response.text if response and response.text else ""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            clean_text = text[:8000] if text else " "
            result = self._client.models.embed_content(
                model=self.embedding_model,
                contents=clean_text,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings


class OpenAILLMClient(BaseLLMClient):
    """OpenAI LLM & Embedding provider."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY tidak ditemukan. Mohon set di .env atau via input UI.")
        
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.embedding_model = "text-embedding-3-small"
        
    def generate_text(self, prompt: str, system_instruction: Optional[str] = None, temperature: float = 0.3) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature
        )
        return resp.choices[0].message.content or ""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        clean_texts = [t[:8000] if t else " " for t in texts]
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=clean_texts
        )
        return [item.embedding for item in resp.data]


class LocalFallbackEmbeddingClient:
    """Lightweight deterministic pseudo-semantic embedding for offline/local test scenarios without external API costs."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            # Deterministic hash-based projection with word-ngram weighting
            vec = np.zeros(self.dim, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h % 2 == 0) else -1.0
                vec[idx] += sign
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors


class MockLLMClient(BaseLLMClient):
    """Mock LLM Client for offline academic pipeline testing and unit tests."""
    
    def __init__(self, model_name: str = "mock-pedagogy-v1"):
        self.model_name = model_name
        self._embedder = LocalFallbackEmbeddingClient()
        
    def generate_text(self, prompt: str, system_instruction: Optional[str] = None, temperature: float = 0.3) -> str:
        return (
            "### [HASIL GENERATIF PEDAGOGIS - MODE SIMULASI OFFLINE]\n\n"
            "**Ringkasan Konsep Utama:**\n"
            "- Materi didasarkan pada dokumen kurikulum Fase E Kelas X yang tersimpan.\n"
            "- Konsep inti telah dipetakan secara terstruktur sesuai Capaian Pembelajaran.\n\n"
            "**Contoh Soal HOTS Terkait:**\n"
            "1. Analisislah skenario praktis berikut berdasarkan prinsip yang dijelaskan dalam modul.\n"
            "*Kunci Jawaban & Pembahasan*: Konsep ini menguji kemampuan level kognitif C4 (Menganalisis).\n\n"
            "*Catatan*: Untuk hasil generatif live penuh, masukkan Google Gemini API Key Anda."
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.embed_texts(texts)


def get_llm_client(
    provider: str = "gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> BaseLLMClient:
    """Factory function to instantiate the selected LLM provider."""
    provider = provider.lower().strip()
    
    if provider == "gemini":
        target_model = model_name or "gemini-3.5-flash"
        try:
            return GeminiLLMClient(api_key=api_key, model_name=target_model)
        except Exception as e:
            print(f"[Warning] Gagal inisialisasi Gemini ({e}). Menggunakan Fallback Mock Client.")
            return MockLLMClient()
            
    elif provider == "openai":
        target_model = model_name or "gpt-4o-mini"
        try:
            return OpenAILLMClient(api_key=api_key, model_name=target_model)
        except Exception as e:
            print(f"[Warning] Gagal inisialisasi OpenAI ({e}). Menggunakan Fallback Mock Client.")
            return MockLLMClient()
            
    elif provider == "mock":
        return MockLLMClient(model_name=model_name or "mock-test")
        
    else:
        raise ValueError(f"Provider '{provider}' tidak didukung. Pilihan: gemini, openai, mock.")
