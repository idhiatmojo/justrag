import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Allowed Subjects for Kelas X
ALLOWED_SUBJECTS = {
    "bahasa_inggris": {
        "display_name": "Bahasa Inggris (Kelas X)",
        "description": "Fase E SMA/SMK: Narrative, Descriptive, Recount, Report Text, Grammar, HOTS Reading",
        "folder": "bahasa_inggris"
    },
    "informatika": {
        "display_name": "Informatika (Kelas X)",
        "description": "Fase E SMA/SMK: Berpikir Komputasional, Algoritma Pemrograman, Jaringan, Analisis Data",
        "folder": "informatika"
    },
    "kka": {
        "display_name": "KKA - Koding & Kecerdasan Artifisial (Kelas X)",
        "description": "Fase E SMA/SMK: Logika Koding Python, Dasar AI/ML, Prompt Engineering, Etika AI",
        "folder": "kka"
    }
}

# Directories
DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdf"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
BM25_PERSIST_DIR = DATA_DIR / "bm25_db"
EVALUATION_DIR = DATA_DIR / "evaluation"

for d in [DATA_DIR, RAW_PDF_DIR, CHROMA_PERSIST_DIR, BM25_PERSIST_DIR, EVALUATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

for sub in ALLOWED_SUBJECTS.keys():
    (RAW_PDF_DIR / sub).mkdir(parents=True, exist_ok=True)

# API Keys & Providers
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Chunking & Retrieval Parameters
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 4
RRF_K = 60 # Reciprocal Rank Fusion constant
