import os
import streamlit as st
from typing import Optional, Dict, Any

from src.subject_manager import load_subjects
from src.user_manager import (
    authenticate_user,
    change_own_password,
    load_users,
    normalize_username,
    sanitize_user_public,
)

# Environment-configurable authentication gate
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")

SESSION_AUTH_KEYS = [
    "authenticated",
    "username",
    "user_name",
    "user_role",
    "user_badge",
    "account_type",
    "allowed_subjects",
]


def _clear_auth_session() -> None:
    for key in SESSION_AUTH_KEYS:
        st.session_state.pop(key, None)


def _populate_auth_session(user_info: Dict[str, Any]) -> None:
    st.session_state["authenticated"] = True
    st.session_state["username"] = user_info["username"]
    st.session_state["user_name"] = user_info["name"]
    st.session_state["user_role"] = user_info["role"]
    st.session_state["user_badge"] = user_info["badge"]
    st.session_state["account_type"] = user_info.get("account_type", "guru")
    st.session_state["allowed_subjects"] = user_info.get("allowed_subjects", ["*"])


def verify_credentials(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verifies username and password against persistent PBKDF2 user database."""
    user_info, _ = authenticate_user(username, password)
    return user_info


def get_current_user() -> Dict[str, Any]:
    """Returns the currently authenticated user profile (or synthetic admin if AUTH_ENABLED=false)."""
    if not AUTH_ENABLED:
        return {
            "username": "admin_dev",
            "name": "Mode Tanpa Autentikasi (Admin)",
            "account_type": "admin",
            "role": "Peneliti (Full Access)",
            "badge": "🛡️ Admin / Dev",
            "allowed_subjects": ["*"],
            "is_active": True,
        }

    uname = normalize_username(st.session_state.get("username", ""))
    users = load_users()
    if uname and uname in users:
        return sanitize_user_public(users[uname])

    return {
        "username": uname or "pengguna",
        "name": st.session_state.get("user_name", "Pengguna"),
        "account_type": st.session_state.get("account_type", "guru"),
        "role": st.session_state.get("user_role", "Pendidik"),
        "badge": st.session_state.get("user_badge", "👨‍🏫 Guru / Pendidik"),
        "allowed_subjects": st.session_state.get("allowed_subjects", ["*"]),
        "is_active": True,
    }


def is_current_user_admin() -> bool:
    """Returns True if the active session belongs to an Administrator account."""
    return get_current_user().get("account_type") == "admin"


def render_login_page() -> None:
    """Renders a modern, centered login interface for Kurikulum Merdeka RAG."""
    # Hide sidebar when on login screen
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        with st.container(border=True):
            st.markdown("## 🔐 Masuk ke Sistem RAGbot")
            st.caption(
                "Sistem RAG Generatif Pembelajaran Kurikulum Merdeka — "
                "Penelitian Pascasarjana / S2 Pendidikan & Ilmu Komputer."
            )
            st.divider()

            with st.form("login_form", clear_on_submit=False):
                username = st.text_input(
                    "Nama Pengguna (Username):",
                    placeholder="Contoh: admin atau guru",
                    autocomplete="username",
                )
                password = st.text_input(
                    "Kata Sandi (Password):",
                    type="password",
                    placeholder="Masukkan kata sandi Anda",
                    autocomplete="current-password",
                )

                submitted = st.form_submit_button(
                    "Masuk ke Sistem",
                    icon=":material/login:",
                    width="stretch",
                )

                if submitted:
                    user_info, err_msg = authenticate_user(username, password)
                    if user_info:
                        _populate_auth_session(user_info)
                        st.toast("Autentikasi berhasil! Selamat datang.", icon="✅")
                        st.rerun()
                    else:
                        st.error(err_msg, icon=":material/error:")

            # Quick credentials hint for evaluation & thesis testing
            with st.expander("ℹ️ Akun Uji Coba Demo Penelitian", expanded=False):
                st.markdown(
                    """
                    Gunakan kredensial pengujian bawaan berikut (atau akun guru yang didaftarkan Admin):
                    - **Akun Peneliti / Admin**: `admin` | Kata Sandi bawaan: `admin123`  
                      *(Akses penuh: konfigurasi model, manajemen mapel & guru, indeksing dokumen, dan evaluasi ablation study)*
                    - **Akun Guru**: `guru` | Kata Sandi bawaan: `guru123`  
                      *(Akses pendidik: pembuatan modul ajar, bank soal HOTS, ringkasan materi, dan upload PDF mapel yang diampu)*
                    """
                )


def render_user_profile_sidebar() -> None:
    """Renders logged-in user details, subject scope, password change, and logout button in sidebar."""
    if AUTH_ENABLED and not st.session_state.get("authenticated", False):
        return

    curr = get_current_user()
    user_name = curr.get("name", "Pengguna")
    user_badge = curr.get("badge", "👤 Pengguna")
    user_role = curr.get("role", "Pendidik")
    acc_type = curr.get("account_type", "guru")
    allowed = curr.get("allowed_subjects", ["*"])

    all_subjects = load_subjects()
    if acc_type == "admin" or "*" in allowed:
        scope_label = "Semua Mata Pelajaran"
    else:
        names = [all_subjects[s]["display_name"] for s in allowed if s in all_subjects]
        scope_label = ", ".join(names) if names else "(Belum ada mapel ditugaskan)"

    with st.sidebar.container(border=True):
        st.markdown(f"**{user_badge}**")
        st.caption(f"{user_name} ({user_role})")
        st.caption(f"📚 **Akses:** {scope_label}")

        if AUTH_ENABLED and curr.get("username"):
            with st.expander("🔑 Ubah Kata Sandi", expanded=False):
                with st.form("sidebar_change_pwd_form", clear_on_submit=True):
                    old_pw = st.text_input("Kata Sandi Lama:", type="password")
                    new_pw = st.text_input("Kata Sandi Baru:", type="password")
                    submitted_pw = st.form_submit_button("Simpan Sandi", width="stretch")
                    if submitted_pw:
                        ok, msg = change_own_password(curr["username"], old_pw, new_pw)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)

            if st.button("Keluar (Logout)", icon=":material/logout:", width="stretch"):
                _clear_auth_session()
                st.rerun()


def require_authentication() -> bool:
    """Gatekeeper function. Syncs active user record; renders login page if unauthenticated."""
    if not AUTH_ENABLED:
        return True

    if not st.session_state.get("authenticated", False):
        render_login_page()
        return False

    # Live-sync user record on every rerun in case Admin updated permissions or deactivated account
    uname = normalize_username(st.session_state.get("username", ""))
    users = load_users()
    user_rec = users.get(uname)
    if not user_rec or not user_rec.get("is_active", True):
        _clear_auth_session()
        st.warning("Sesi Anda telah berakhir atau akun dinonaktifkan oleh Administrator.")
        render_login_page()
        return False

    _populate_auth_session(sanitize_user_public(user_rec))
    return True
