import hashlib
import hmac
import json
import os
import re
import secrets
from typing import Any, Dict, List, Optional, Tuple

from src.config import USERS_FILE

PBKDF2_ITERATIONS = 200_000

BADGE_OPTIONS = [
    "👨‍🏫 Guru / Pendidik",
    "👩‍🏫 Guru / Pendidik",
    "🛡️ Admin / Peneliti",
]


def hash_password(password: str) -> str:
    """Hashes a password using PBKDF2-HMAC-SHA256 with a cryptographic random salt."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${dk}"


def verify_password_hash(password: str, stored_hash: str) -> bool:
    """Verifies a plaintext password against a stored PBKDF2 hash in constant time."""
    if not password or not stored_hash:
        return False

    parts = stored_hash.split("$")
    if len(parts) == 4 and parts[0] == "pbkdf2_sha256":
        try:
            iterations = int(parts[1])
            salt = parts[2]
            expected_hex = parts[3]
            computed_hex = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("utf-8"),
                iterations,
            ).hex()
            return hmac.compare_digest(computed_hex, expected_hex)
        except (ValueError, TypeError):
            return False

    return False


def normalize_username(username: str) -> str:
    """Normalizes username to lowercase trimmed format."""
    return (username or "").lower().strip()


def validate_username(username: str) -> Tuple[bool, str]:
    """Validates username format (3-32 chars, lowercase letters, digits, _, ., -)."""
    uname = normalize_username(username)
    if not uname:
        return False, "Username wajib diisi."
    if not re.match(r"^[a-z0-9][a-z0-9_.-]{2,31}$", uname):
        return (
            False,
            "Username harus 3–32 karakter, diawali huruf/angka, dan hanya boleh memuat huruf, angka, titik (.), garis bawah (_), atau strip (-).",
        )
    return True, uname


def _build_default_users() -> Dict[str, Dict[str, Any]]:
    """Builds initial seed users from environment variables on first startup."""
    admin_user = normalize_username(os.getenv("APP_ADMIN_USER", "admin")) or "admin"
    admin_pass = os.getenv("APP_ADMIN_PASSWORD", "admin123")

    guru_user = normalize_username(os.getenv("APP_GURU_USER", "guru")) or "guru"
    guru_pass = os.getenv("APP_GURU_PASSWORD", "guru123")

    users: Dict[str, Dict[str, Any]] = {
        admin_user: {
            "username": admin_user,
            "name": "Administrator / Peneliti S2",
            "account_type": "admin",
            "role": "Peneliti (Full Access)",
            "badge": "🛡️ Admin / Peneliti",
            "allowed_subjects": ["*"],
            "is_active": True,
            "password_hash": hash_password(admin_pass),
        }
    }

    if guru_user != admin_user:
        users[guru_user] = {
            "username": guru_user,
            "name": "Guru Pamong / Pendidik Fase E",
            "account_type": "guru",
            "role": "Pendidik (Modul & Asesmen)",
            "badge": "👨‍🏫 Guru / Pendidik",
            "allowed_subjects": ["*"],
            "is_active": True,
            "password_hash": hash_password(guru_pass),
        }

    return users


def _normalize_user_record(uname: str, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalizes a single user record loaded from disk."""
    key = normalize_username(uname)
    if not key or not isinstance(info, dict):
        return None

    acc_type = str(info.get("account_type", "guru")).lower().strip()
    if acc_type not in ("admin", "guru"):
        acc_type = "guru"

    raw_allowed = info.get("allowed_subjects", ["*"])
    if acc_type == "admin":
        allowed = ["*"]
    elif isinstance(raw_allowed, list):
        cleaned = [str(s).lower().strip() for s in raw_allowed if str(s).strip()]
        allowed = ["*"] if "*" in cleaned else cleaned
    else:
        allowed = ["*"]

    default_badge = "🛡️ Admin / Peneliti" if acc_type == "admin" else "👨‍🏫 Guru / Pendidik"
    default_role = "Peneliti (Full Access)" if acc_type == "admin" else "Pendidik (Modul & Asesmen)"

    pwd_hash = str(info.get("password_hash", "")).strip()
    if not pwd_hash and info.get("password"):
        pwd_hash = hash_password(str(info["password"]))

    return {
        "username": key,
        "name": str(info.get("name", key)).strip() or key,
        "account_type": acc_type,
        "role": str(info.get("role", default_role)).strip() or default_role,
        "badge": str(info.get("badge", default_badge)).strip() or default_badge,
        "allowed_subjects": allowed,
        "is_active": bool(info.get("is_active", True)),
        "password_hash": pwd_hash,
    }


def save_all_users(users: Dict[str, Dict[str, Any]]) -> None:
    """Saves all user records to data/users.json."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def load_users() -> Dict[str, Dict[str, Any]]:
    """Loads all registered users from data/users.json, seeding defaults if missing."""
    if not USERS_FILE.exists():
        defaults = _build_default_users()
        save_all_users(defaults)
        return defaults

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and raw:
            normalized: Dict[str, Dict[str, Any]] = {}
            migrated_plaintext = False
            for k, v in raw.items():
                if isinstance(v, dict) and "password" in v and not v.get("password_hash"):
                    migrated_plaintext = True
                rec = _normalize_user_record(k, v)
                if rec and rec["password_hash"]:
                    normalized[rec["username"]] = rec
            if normalized:
                if migrated_plaintext:
                    save_all_users(normalized)
                return normalized
    except Exception as e:
        print(f"[Warning] Gagal membaca {USERS_FILE}: {e}. Menggunakan default.")

    defaults = _build_default_users()
    save_all_users(defaults)
    return defaults


def sanitize_user_public(user_rec: Dict[str, Any]) -> Dict[str, Any]:
    """Returns user metadata safe for session storage (excludes password_hash)."""
    return {k: v for k, v in user_rec.items() if k != "password_hash"}


def authenticate_user(username: str, password: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Authenticates username & password. Returns (public_user_dict, error_message)."""
    uname = normalize_username(username)
    if not uname or not password:
        return None, "Nama pengguna dan kata sandi wajib diisi."

    users = load_users()
    user_rec = users.get(uname)
    if not user_rec:
        return None, "Nama pengguna atau kata sandi yang Anda masukkan salah."

    if not user_rec.get("is_active", True):
        return None, "Akun Anda sedang dinonaktifkan. Silakan hubungi Administrator."

    if not verify_password_hash(password, user_rec.get("password_hash", "")):
        return None, "Nama pengguna atau kata sandi yang Anda masukkan salah."

    return sanitize_user_public(user_rec), ""


def _count_active_admins(users: Dict[str, Dict[str, Any]], exclude_username: Optional[str] = None) -> int:
    count = 0
    for uname, rec in users.items():
        if exclude_username and uname == exclude_username:
            continue
        if rec.get("account_type") == "admin" and rec.get("is_active", True):
            count += 1
    return count


def create_user(
    username: str,
    name: str,
    password: str,
    account_type: str = "guru",
    role: str = "",
    badge: str = "",
    allowed_subjects: Optional[List[str]] = None,
    is_active: bool = True,
) -> Tuple[bool, str]:
    """Registers a new Guru or Admin account."""
    ok_u, uname_or_err = validate_username(username)
    if not ok_u:
        return False, uname_or_err
    uname = uname_or_err

    clean_name = (name or "").strip()
    if not clean_name:
        return False, "Nama lengkap pengguna wajib diisi."

    if not password or len(password) < 4:
        return False, "Kata sandi minimal harus terdiri dari 4 karakter."

    acc_type = (account_type or "guru").lower().strip()
    if acc_type not in ("admin", "guru"):
        return False, "Tipe akun tidak valid (pilih 'guru' atau 'admin')."

    users = load_users()
    if uname in users:
        return False, f"Username '{uname}' sudah digunakan. Silakan pilih username lain."

    if acc_type == "admin":
        final_subjects = ["*"]
    else:
        raw_subs = allowed_subjects if allowed_subjects is not None else ["*"]
        cleaned_subs = [str(s).lower().strip() for s in raw_subs if str(s).strip()]
        if not cleaned_subs:
            return False, "Pilih minimal 1 mata pelajaran untuk akun guru ini (atau pilih Semua Mapel)."
        final_subjects = ["*"] if "*" in cleaned_subs else list(dict.fromkeys(cleaned_subs))

    default_role = "Peneliti (Full Access)" if acc_type == "admin" else "Pendidik (Modul & Asesmen)"
    default_badge = "🛡️ Admin / Peneliti" if acc_type == "admin" else "👨‍🏫 Guru / Pendidik"

    users[uname] = {
        "username": uname,
        "name": clean_name,
        "account_type": acc_type,
        "role": (role or "").strip() or default_role,
        "badge": (badge or "").strip() or default_badge,
        "allowed_subjects": final_subjects,
        "is_active": bool(is_active),
        "password_hash": hash_password(password),
    }
    save_all_users(users)
    label_tipe = "Admin" if acc_type == "admin" else "Guru"
    return True, f"Akun {label_tipe} '{clean_name}' (`{uname}`) berhasil dibuat!"


def update_user(
    username: str,
    name: str,
    account_type: str,
    role: str,
    badge: str,
    allowed_subjects: List[str],
    is_active: bool = True,
    new_password: str = "",
    actor_username: Optional[str] = None,
) -> Tuple[bool, str]:
    """Updates an existing user's details, permissions, status, and optionally resets password."""
    uname = normalize_username(username)
    actor = normalize_username(actor_username or "")
    users = load_users()

    if uname not in users:
        return False, f"Pengguna '{uname}' tidak ditemukan."

    clean_name = (name or "").strip()
    if not clean_name:
        return False, "Nama lengkap pengguna wajib diisi."

    acc_type = (account_type or "guru").lower().strip()
    if acc_type not in ("admin", "guru"):
        return False, "Tipe akun tidak valid."

    # Prevent self-lockout for currently logged-in admin
    if actor and uname == actor:
        if not is_active:
            return False, "Anda tidak dapat menonaktifkan akun Anda sendiri yang sedang digunakan."
        if acc_type != "admin":
            return False, "Anda tidak dapat menurunkan hak akses Admin akun Anda sendiri."

    # Ensure at least one active admin remains
    current_rec = users[uname]
    if current_rec.get("account_type") == "admin" and (acc_type != "admin" or not is_active):
        if _count_active_admins(users, exclude_username=uname) < 1:
            return False, "Sistem wajib memiliki minimal 1 akun Admin yang aktif."

    if acc_type == "admin":
        final_subjects = ["*"]
    else:
        cleaned_subs = [str(s).lower().strip() for s in (allowed_subjects or []) if str(s).strip()]
        if not cleaned_subs:
            return False, "Pilih minimal 1 mata pelajaran untuk akun guru ini (atau pilih Semua Mapel)."
        final_subjects = ["*"] if "*" in cleaned_subs else list(dict.fromkeys(cleaned_subs))

    pwd_hash = current_rec["password_hash"]
    if new_password:
        if len(new_password) < 4:
            return False, "Kata sandi baru minimal harus terdiri dari 4 karakter."
        pwd_hash = hash_password(new_password)

    default_role = "Peneliti (Full Access)" if acc_type == "admin" else "Pendidik (Modul & Asesmen)"
    default_badge = "🛡️ Admin / Peneliti" if acc_type == "admin" else "👨‍🏫 Guru / Pendidik"

    users[uname] = {
        "username": uname,
        "name": clean_name,
        "account_type": acc_type,
        "role": (role or "").strip() or default_role,
        "badge": (badge or "").strip() or default_badge,
        "allowed_subjects": final_subjects,
        "is_active": bool(is_active),
        "password_hash": pwd_hash,
    }
    save_all_users(users)
    return True, f"Data akun '{clean_name}' (`{uname}`) berhasil diperbarui!"


def change_own_password(username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
    """Allows a logged-in user to change their own password after verifying old password."""
    uname = normalize_username(username)
    users = load_users()
    rec = users.get(uname)
    if not rec:
        return False, "Akun tidak ditemukan."

    if not verify_password_hash(old_password, rec.get("password_hash", "")):
        return False, "Kata sandi lama tidak sesuai."

    if not new_password or len(new_password) < 4:
        return False, "Kata sandi baru minimal harus terdiri dari 4 karakter."

    rec["password_hash"] = hash_password(new_password)
    save_all_users(users)
    return True, "Kata sandi berhasil diperbarui!"


def delete_user(username: str, actor_username: Optional[str] = None) -> Tuple[bool, str]:
    """Deletes a user account with safety guards against self-deletion and last-admin removal."""
    uname = normalize_username(username)
    actor = normalize_username(actor_username or "")
    users = load_users()

    if uname not in users:
        return False, f"Pengguna '{uname}' tidak ditemukan."

    if actor and uname == actor:
        return False, "Anda tidak dapat menghapus akun Anda sendiri yang sedang login."

    target = users[uname]
    if target.get("account_type") == "admin" and _count_active_admins(users, exclude_username=uname) < 1:
        return False, "Tidak dapat menghapus satu-satunya akun Admin aktif di sistem."

    removed = users.pop(uname)
    save_all_users(users)
    return True, f"Akun '{removed.get('name', uname)}' (`{uname}`) berhasil dihapus."


def remove_subject_from_users(deleted_subject_key: str) -> None:
    """Removes a deleted subject key from specific subject assignments across guru accounts."""
    sub = (deleted_subject_key or "").lower().strip()
    if not sub:
        return
    users = load_users()
    changed = False
    for rec in users.values():
        allowed = rec.get("allowed_subjects", ["*"])
        if "*" not in allowed and sub in allowed:
            rec["allowed_subjects"] = [s for s in allowed if s != sub]
            changed = True
    if changed:
        save_all_users(users)


def get_accessible_subjects_for_user(
    user_info: Dict[str, Any],
    all_subjects: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Filters all_subjects to only those permitted for the given user."""
    if not user_info:
        return {}
    if user_info.get("account_type") == "admin":
        return all_subjects
    allowed = user_info.get("allowed_subjects", ["*"])
    if "*" in allowed:
        return all_subjects
    allowed_set = {s.lower().strip() for s in allowed}
    return {k: v for k, v in all_subjects.items() if k in allowed_set}
