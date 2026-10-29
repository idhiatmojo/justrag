import os
import re
import time
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
        """Embed a list of texts using Gemini embedding API.

        Sends texts in batches (up to EMBED_BATCH_SIZE per API call) so that
        a single ingest run does NOT blow through the RPM quota by firing one
        request per chunk.  When a 429 RESOURCE_EXHAUSTED error is returned
        the method waits the server-suggested retry delay (extracted from the
        error payload) and then retries the same batch — up to MAX_RETRIES
        times before re-raising.
        """
        EMBED_BATCH_SIZE = 100   # max texts per embed_content call
        MAX_RETRIES = 5
        DEFAULT_WAIT = 60        # fallback wait seconds when server gives no hint

        from google.genai import errors as genai_errors

        def _extract_retry_delay(err: Exception) -> float:
            """Parse the suggested retry delay (seconds) from a 429 error."""
            # Structured payload: error.details[].retryDelay
            payload = getattr(err, "response_json", None)
            if isinstance(payload, dict):
                for detail in payload.get("error", {}).get("details", []):
                    if isinstance(detail, dict) and "retryDelay" in detail:
                        delay_str = detail["retryDelay"]
                        if isinstance(delay_str, str) and delay_str.endswith("s"):
                            try:
                                return float(delay_str[:-1])
                            except ValueError:
                                pass
            # Fallback: scan the string representation
            match = re.search(r"retry\s+in\s+([0-9.]+)\s*s", str(err), re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
            return DEFAULT_WAIT

        clean_texts = [t[:8000] if t else " " for t in texts]
        all_embeddings: List[List[float]] = []

        for batch_start in range(0, len(clean_texts), EMBED_BATCH_SIZE):
            batch = clean_texts[batch_start : batch_start + EMBED_BATCH_SIZE]
            attempt = 0
            while True:
                try:
                    result = self._client.models.embed_content(
                        model=self.embedding_model,
                        contents=batch,
                    )
                    all_embeddings.extend(emb.values for emb in result.embeddings)
                    break  # success — move to next batch
                except genai_errors.ClientError as exc:
                    if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                        attempt += 1
                        if attempt > MAX_RETRIES:
                            raise
                        wait_secs = _extract_retry_delay(exc) + 1  # +1s safety buffer
                        print(
                            f"[GeminiLLMClient] Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                            f"Waiting {wait_secs:.1f}s before retry..."
                        )
                        time.sleep(wait_secs)
                    else:
                        raise

        return all_embeddings


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


class CustomOpenAICompatibleClient(BaseLLMClient):
    """Generic OpenAI-compatible client for third-party routers/proxies (e.g. JuanRouter).

    These endpoints expose the OpenAI Chat Completions API but usually do NOT
    provide an embedding endpoint.  Embeddings are therefore delegated to one
    of the supported fallback providers:
      - "gemini"  → GeminiLLMClient.embed_texts  (recommended; rate-limit safe)
      - "openai"  → OpenAILLMClient.embed_texts
      - "local"   → LocalFallbackEmbeddingClient  (no external API, hash-based)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "qwen3.8-max",
        embedding_provider: str = "gemini",
        embedding_api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("JUANROUTER_API_KEY") or os.getenv("CUSTOM_API_KEY")
        self.base_url = base_url or os.getenv("JUANROUTER_BASE_URL") or os.getenv("CUSTOM_BASE_URL")

        if not self.api_key:
            raise ValueError("Custom API key tidak ditemukan. Set via UI atau JUANROUTER_API_KEY di .env.")
        if not self.base_url:
            raise ValueError("Custom base URL tidak ditemukan. Set via UI atau JUANROUTER_BASE_URL di .env.")

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=120.0)
        self.model_name = model_name

        # Set up embedding fallback
        emb_prov = embedding_provider.lower().strip()
        if emb_prov == "gemini":
            gemini_key = embedding_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                self._embedder: BaseLLMClient | LocalFallbackEmbeddingClient = GeminiLLMClient(api_key=gemini_key)
            else:
                print("[CustomOpenAICompatibleClient] GEMINI_API_KEY tidak ada, beralih ke embedding lokal.")
                self._embedder = LocalFallbackEmbeddingClient()
        elif emb_prov == "openai":
            openai_key = embedding_api_key or os.getenv("OPENAI_API_KEY")
            if openai_key:
                self._embedder = OpenAILLMClient(api_key=openai_key)
            else:
                print("[CustomOpenAICompatibleClient] OPENAI_API_KEY tidak ada, beralih ke embedding lokal.")
                self._embedder = LocalFallbackEmbeddingClient()
        else:
            self._embedder = LocalFallbackEmbeddingClient()

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Delegate embedding to the configured fallback provider."""
        return self._embedder.embed_texts(texts)


class GrokLLMClient(BaseLLMClient):
    """xAI Grok LLM provider (OpenAI-compatible API).

    Grok does not offer a native embedding model, so embeddings are delegated
    to a configurable fallback:
      - "gemini"  → GeminiLLMClient.embed_texts  (recommended; requires GEMINI_API_KEY)
      - "openai"  → OpenAILLMClient.embed_texts   (requires OPENAI_API_KEY)
      - "local"   → LocalFallbackEmbeddingClient  (offline hash-based, no external API)
    """

    XAI_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "grok-3-mini",
        embedding_provider: str = "gemini",
        embedding_api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "XAI_API_KEY tidak ditemukan. Mohon set di .env atau via input UI."
            )

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.XAI_BASE_URL, timeout=120.0)
        self.model_name = model_name

        # Set up embedding fallback
        embedding_provider = embedding_provider.lower().strip()
        if embedding_provider == "gemini":
            gemini_key = embedding_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                self._embedder = GeminiLLMClient(api_key=gemini_key)
            else:
                print("[GrokLLMClient] GEMINI_API_KEY tidak ada, beralih ke embedding lokal.")
                self._embedder = LocalFallbackEmbeddingClient()
        elif embedding_provider == "openai":
            openai_key = embedding_api_key or os.getenv("OPENAI_API_KEY")
            if openai_key:
                self._embedder = OpenAILLMClient(api_key=openai_key)
            else:
                print("[GrokLLMClient] OPENAI_API_KEY tidak ada, beralih ke embedding lokal.")
                self._embedder = LocalFallbackEmbeddingClient()
        else:
            self._embedder = LocalFallbackEmbeddingClient()

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Delegate embedding to the configured fallback provider."""
        return self._embedder.embed_texts(texts)


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
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
) -> BaseLLMClient:
    """Factory function to instantiate the selected LLM provider.

    Parameters
    ----------
    provider:
        "gemini" | "openai" | "grok" | "custom" | "mock"
    model_name:
        Override the default model for the chosen provider.
    api_key:
        Primary API key for the LLM provider.
    base_url:
        (custom only) Base URL of the OpenAI-compatible endpoint.
    embedding_provider:
        (grok/custom) Which service to use for embeddings: "gemini" | "openai" | "local".
        Auto-detected from available env keys when not specified.
    embedding_api_key:
        (grok/custom) API key for the embedding provider if different from ``api_key``.
    """
    provider = provider.lower().strip()

    def _auto_emb_provider() -> str:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "gemini"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "local"

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

    elif provider == "grok":
        target_model = model_name or "grok-3-mini"
        emb_provider = embedding_provider or _auto_emb_provider()
        try:
            return GrokLLMClient(
                api_key=api_key,
                model_name=target_model,
                embedding_provider=emb_provider,
                embedding_api_key=embedding_api_key,
            )
        except Exception as e:
            print(f"[Warning] Gagal inisialisasi Grok ({e}). Menggunakan Fallback Mock Client.")
            return MockLLMClient()

    elif provider == "custom":
        target_model = model_name or "qwen3.8-max"
        emb_provider = embedding_provider or _auto_emb_provider()
        try:
            return CustomOpenAICompatibleClient(
                api_key=api_key,
                base_url=base_url,
                model_name=target_model,
                embedding_provider=emb_provider,
                embedding_api_key=embedding_api_key,
            )
        except Exception as e:
            print(f"[Warning] Gagal inisialisasi Custom endpoint ({e}). Menggunakan Fallback Mock Client.")
            return MockLLMClient()

    elif provider == "mock":
        return MockLLMClient(model_name=model_name or "mock-test")

    else:
        raise ValueError(f"Provider '{provider}' tidak didukung. Pilihan: gemini, openai, grok, custom, mock.")
