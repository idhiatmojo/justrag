import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI
from src.config import DATA_DIR

CUSTOM_PROVIDERS_FILE = DATA_DIR / "custom_providers.json"

TAMANDATA_ALL_MODELS = [
    "tamandata",
    "cx/tamandata-auto",
    "cx/gpt-5.6-sol",
    "cx/gpt-5.6-terra",
    "cx/gpt-5.6-luna",
    "solar-pro4",
    "cx/gpt-6-astra",
    "cx/gpt-6-sol",
    "cx/gpt-6-luna",
    "cx/gpt-6-sol-high",
    "cx/gpt-6-luna-high",
    "cx/gpt-6.1-sol",
    "cx/gpt-6.1-terra",
    "cx/gpt-5.6-sol-1m",
    "cx/gpt-5.6-terra-1m",
    "cx/gpt-5.6-luna-1m",
    "cx/gpt-6-luna-1m",
    "cx/gpt-6-sol-1m",
    "cx/gpt-5.6-sol-high",
    "cx/gpt-5.6-terra-high",
    "cx/gpt-5.6-luna-high",
    "cx/gpt-5.6-sol-max",
    "cx/gpt-5.6-terra-max",
    "cx/gpt-5.6-luna-max",
    "cx/gpt-5.6-sol-review",
    "cx/gpt-5.6-terra-review",
    "cx/gpt-5.6-luna-review",
    "cx/codex-auto-review",
    "cx/gpt-5.5",
    "cx/gpt-5.3-codex-spark",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "deepseek-v4.1-flash",
    "gemma-4-31b-it",
    "qwen3.6-35b-a3b",
    "step-3.7-flash",
    "step-explore",
    "longcat-2.0",
    "nemotron-3-nano-30b-a3b",
    "glm-5.2",
    "free",
]

DEFAULT_PRESETS: Dict[str, Dict[str, Any]] = {
    "TamanData - tamandata": {
        "base_url": "https://ai.tamandata.com/v1",
        "api_key": os.getenv("TAMANDATA_API_KEY", ""),
        "model_name": "tamandata",
        "embedding_provider": "gemini",
        "available_models": TAMANDATA_ALL_MODELS,
    },
    "JuanRouter - Qwen (Direkomendasikan)": {
        "base_url": "https://api.juanrouter.com/v1",
        "api_key": os.getenv("JUANROUTER_API_KEY", ""),
        "model_name": "qwen3.8-max",
        "embedding_provider": "gemini",
        "available_models": [
            "qwen3.8-max",
            "deepseek-v4-pro",
            "gemini-3.1-pro",
            "gpt-5.6-terra",
            "claude-opus-4-6-thinking",
            "gemini-3.6-flash"
        ]
    },
    "JuanRouter - DeepSeek": {
        "base_url": "https://api.juanrouter.com/v1",
        "api_key": os.getenv("JUANROUTER_API_KEY", ""),
        "model_name": "deepseek-v4-pro",
        "embedding_provider": "gemini",
        "available_models": [
            "deepseek-v4-pro",
            "qwen3.8-max",
            "gemini-3.1-pro",
            "gpt-5.6-terra"
        ]
    }
}

def load_custom_presets() -> Dict[str, Dict[str, Any]]:
    """Memuat daftar custom provider/preset yang tersimpan di disk.
    Jika belum ada file, buat file awal dengan template default."""
    if not CUSTOM_PROVIDERS_FILE.exists():
        save_all_presets(DEFAULT_PRESETS)
        return DEFAULT_PRESETS.copy()

    try:
        with open(CUSTOM_PROVIDERS_FILE, "r", encoding="utf-8") as f:
            presets = json.load(f)
            if isinstance(presets, dict) and presets:
                return presets
            # File kosong / format tidak valid, tulis default
            save_all_presets(DEFAULT_PRESETS)
            return DEFAULT_PRESETS.copy()
    except Exception as e:
        print(f"[Warning] Gagal membaca {CUSTOM_PROVIDERS_FILE}: {e}. Memakai default.")
        return DEFAULT_PRESETS.copy()

def save_all_presets(presets: Dict[str, Dict[str, Any]]) -> None:
    """Menyimpan seluruh kamus preset ke file JSON."""
    CUSTOM_PROVIDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_PROVIDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)

def save_custom_preset(name: str, config: Dict[str, Any]) -> None:
    """Menambah atau memperbarui preset custom tunggal."""
    presets = load_custom_presets()
    key_name = name.strip()
    existing = presets.get(key_name, {})
    presets[key_name] = {
        "base_url": config.get("base_url", "").strip(),
        "api_key": config.get("api_key", "").strip(),
        "model_name": config.get("model_name", "").strip(),
        "embedding_provider": config.get("embedding_provider", "gemini").strip(),
        "available_models": config.get("available_models", existing.get("available_models", []))
    }
    save_all_presets(presets)

def delete_custom_preset(name: str) -> bool:
    """Menghapus preset tertentu."""
    presets = load_custom_presets()
    if name in presets:
        del presets[name]
        save_all_presets(presets)
        return True
    return False

def fetch_available_models(base_url: str, api_key: str, timeout: float = 8.0) -> List[str]:
    """Mengambil daftar ID model secara otomatis dari endpoint yang mendukung /v1/models."""
    if not base_url or not api_key:
        return []
    try:
        client = OpenAI(api_key=api_key.strip(), base_url=base_url.strip(), timeout=timeout)
        models_data = client.models.list()
        return [m.id for m in models_data if hasattr(m, "id")]
    except Exception as e:
        print(f"[fetch_available_models] Error: {e}")
        return []

def test_custom_endpoint(
    base_url: str,
    api_key: str,
    model_name: str,
    timeout: float = 12.0
) -> Tuple[bool, str]:
    """Mengirim pesan singkat 1 token untuk memastikan kredensial dan model aktif."""
    if not base_url:
        return False, "Base URL wajib diisi."
    if not api_key:
        return False, "API Key wajib diisi."
    if not model_name:
        return False, "Model wajib diisi."

    try:
        client = OpenAI(api_key=api_key.strip(), base_url=base_url.strip(), timeout=timeout)
        resp = client.chat.completions.create(
            model=model_name.strip(),
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=5,
            temperature=0.1
        )
        reply = resp.choices[0].message.content or ""
        return True, f"Berhasil terhubung ke `{model_name}`! Respon: '{reply.strip()}'"
    except Exception as e:
        return False, f"Gagal menghubungi model: {str(e)}"

