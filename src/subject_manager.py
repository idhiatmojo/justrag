import json
import re
import shutil
from typing import Any, Dict, Tuple

from src.config import ALLOWED_SUBJECTS, DEFAULT_SUBJECTS, RAW_PDF_DIR, SUBJECTS_FILE


def slugify_subject_key(raw: str) -> str:
    """Normalizes an arbitrary string into a safe lowercase snake_case subject key."""
    if not raw:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw.lower().strip()).strip("_")
    return cleaned[:48]


def _normalize_subjects_dict(raw_subjects: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Ensures every subject entry contains complete metadata fields."""
    normalized: Dict[str, Dict[str, Any]] = {}
    for raw_key, info in raw_subjects.items():
        key = slugify_subject_key(raw_key)
        if not key or not isinstance(info, dict):
            continue
        grade = str(info.get("grade_level", "Kelas X")).strip() or "Kelas X"
        display = str(info.get("display_name", key.replace("_", " ").title())).strip()
        desc = str(info.get("description", "")).strip() or f"Materi pembelajaran {display} ({grade})"
        sample_q = (
            str(info.get("sample_query", "")).strip()
            or f"Buat modul ajar {display} beserta asesmen soal HOTS kontekstual"
        )
        normalized[key] = {
            "display_name": display,
            "description": desc,
            "folder": key,
            "grade_level": grade,
            "sample_query": sample_q,
        }
    return normalized


def _sync_runtime_subjects(subjects: Dict[str, Dict[str, Any]]) -> None:
    """Syncs in-memory ALLOWED_SUBJECTS and ensures raw_pdf directories exist."""
    ALLOWED_SUBJECTS.clear()
    ALLOWED_SUBJECTS.update(subjects)
    for sub_key in subjects.keys():
        (RAW_PDF_DIR / sub_key).mkdir(parents=True, exist_ok=True)


def save_all_subjects(subjects: Dict[str, Dict[str, Any]]) -> None:
    """Persists the entire subjects dictionary to data/subjects.json and syncs runtime state."""
    normalized = _normalize_subjects_dict(subjects)
    SUBJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUBJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
    _sync_runtime_subjects(normalized)


def load_subjects() -> Dict[str, Dict[str, Any]]:
    """Loads registered subjects from data/subjects.json, seeding defaults if needed."""
    if not SUBJECTS_FILE.exists():
        defaults = _normalize_subjects_dict(DEFAULT_SUBJECTS)
        save_all_subjects(defaults)
        return defaults

    try:
        with open(SUBJECTS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and raw:
            normalized = _normalize_subjects_dict(raw)
            if normalized:
                _sync_runtime_subjects(normalized)
                return normalized
    except Exception as e:
        print(f"[Warning] Gagal membaca {SUBJECTS_FILE}: {e}. Menggunakan default.")

    defaults = _normalize_subjects_dict(DEFAULT_SUBJECTS)
    save_all_subjects(defaults)
    return defaults


def upsert_subject(
    subject_key: str,
    display_name: str,
    description: str,
    grade_level: str = "Kelas X",
    sample_query: str = "",
    is_new: bool = False,
) -> Tuple[bool, str, str]:
    """Creates or updates a subject dynamically."""
    clean_display = (display_name or "").strip()
    if not clean_display:
        return False, "Nama tampilan mata pelajaran wajib diisi.", ""

    slug = slugify_subject_key(subject_key or clean_display)
    if len(slug) < 2:
        return False, "Kode mata pelajaran tidak valid (minimal 2 karakter huruf/angka).", ""

    subjects = load_subjects()
    if is_new and slug in subjects:
        return (
            False,
            f"Kode mata pelajaran '{slug}' sudah terdaftar. Silakan gunakan kode lain atau edit mapel tersebut.",
            slug,
        )
    if not is_new and slug not in subjects:
        return False, f"Mata pelajaran '{slug}' tidak ditemukan.", slug

    clean_grade = (grade_level or "Kelas X").strip() or "Kelas X"
    clean_desc = (description or "").strip() or f"Materi pembelajaran & asesmen {clean_display} ({clean_grade})"
    clean_sample = (
        (sample_query or "").strip()
        or f"Buat modul ajar {clean_display} beserta asesmen soal HOTS kontekstual"
    )

    subjects[slug] = {
        "display_name": clean_display,
        "description": clean_desc,
        "folder": slug,
        "grade_level": clean_grade,
        "sample_query": clean_sample,
    }
    save_all_subjects(subjects)
    action_word = "ditambahkan" if is_new else "diperbarui"
    return True, f"Mata pelajaran '{clean_display}' (`{slug}`) berhasil {action_word}!", slug


def delete_subject(
    subject_key: str,
    purge_data: bool = False,
    vector_store: Any = None,
    bm25_store: Any = None,
) -> Tuple[bool, str]:
    """Deletes a subject from configuration and optionally purges its PDF files and indexes."""
    slug = slugify_subject_key(subject_key)
    subjects = load_subjects()

    if slug not in subjects:
        return False, f"Mata pelajaran '{slug}' tidak ditemukan."

    if len(subjects) <= 1:
        return False, "Tidak dapat menghapus mata pelajaran terakhir. Minimal harus ada 1 mapel aktif."

    removed_info = subjects.pop(slug)
    save_all_subjects(subjects)

    if purge_data:
        if vector_store is not None and hasattr(vector_store, "clear_subject"):
            try:
                vector_store.clear_subject(slug)
            except Exception as e:
                print(f"[Warning] Gagal membersihkan vector store mapel {slug}: {e}")

        if bm25_store is not None and hasattr(bm25_store, "clear_subject"):
            try:
                bm25_store.clear_subject(slug)
            except Exception as e:
                print(f"[Warning] Gagal membersihkan BM25 store mapel {slug}: {e}")

        target_dir = (RAW_PDF_DIR / slug).resolve()
        if target_dir.exists() and target_dir.parent == RAW_PDF_DIR.resolve():
            shutil.rmtree(target_dir, ignore_errors=True)

    return True, f"Mata pelajaran '{removed_info.get('display_name', slug)}' berhasil dihapus."
