import os
import hmac
import hashlib
import streamlit as st
from typing import Optional, Dict, Any

# Environment-configurable credentials
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")

def _get_registered_users() -> Dict[str, Dict[str, str]]:
    """Returns the dictionary of configured users and their roles/passwords."""
    admin_user = os.getenv("APP_ADMIN_USER", "admin")
    admin_pass = os.getenv("APP_ADMIN_PASSWORD", "admin123")
    
    guru_user = os.getenv("APP_GURU_USER", "guru")
    guru_pass = os.getenv("APP_GURU_PASSWORD", "guru123")

    return {
        admin_user.lower().strip(): {
            "username": admin_user,
            "name": "Administrator / Peneliti S2",
            "role": "Peneliti (Full Access)",
            "password": admin_pass,
            "badge": "🛡️ Admin / Peneliti"
        },
        guru_user.lower().strip(): {
            "username": guru_user,
            "name": "Guru Pamong / Pendidik Fase E",
            "role": "Pendidik (Modul & Asesmen)",
            "password": guru_pass,
            "badge": "👨‍🏫 Guru / Pendidik"
        }
    }


def verify_credentials(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verifies username and password using secure constant-time comparison."""
    if not username or not password:
        return None

    users = _get_registered_users()
    uname = username.lower().strip()
    if uname in users:
        user_info = users[uname]
        # Secure string comparison to prevent timing attacks
        valid_password = user_info["password"]
        if hmac.compare_digest(password.encode("utf-8"), valid_password.encode("utf-8")):
            return user_info

    return None


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
                "Sistem RAG Generatif Pembelajaran Fase E (Kelas X) Kurikulum Merdeka — "
                "Penelitian Pascasarjana / S2 Pendidikan & Ilmu Komputer."
            )
            st.divider()

            with st.form("login_form", clear_on_submit=False):
                username = st.text_input(
                    "Nama Pengguna (Username):",
                    placeholder="Contoh: admin atau guru",
                    autocomplete="username"
                )
                password = st.text_input(
                    "Kata Sandi (Password):",
                    type="password",
                    placeholder="Masukkan kata sandi Anda",
                    autocomplete="current-password"
                )
                
                submitted = st.form_submit_button(
                    "Masuk ke Sistem",
                    icon=":material/login:",
                    width="stretch"
                )

                if submitted:
                    user_info = verify_credentials(username, password)
                    if user_info:
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = user_info["username"]
                        st.session_state["user_name"] = user_info["name"]
                        st.session_state["user_role"] = user_info["role"]
                        st.session_state["user_badge"] = user_info["badge"]
                        st.toast("Autentikasi berhasil! Selamat datang.", icon="✅")
                        st.rerun()
                    else:
                        st.error(
                            "Nama pengguna atau kata sandi yang Anda masukkan salah.",
                            icon=":material/error:"
                        )

            # Quick credentials hint for evaluation & thesis testing
            with st.expander("ℹ️ Akun Uji Coba Demo Penelitian", expanded=False):
                st.markdown(
                    """
                    Gunakan kredensial pengujian berikut untuk mengakses sistem:
                    - **Akun Peneliti**: `admin` | Kata Sandi: `admin123`  
                      *(Akses penuh konfigurasi model, indeksing dokumen, dan evaluasi ablation study)*
                    - **Akun Guru**: `guru` | Kata Sandi: `guru123`  
                      *(Akses praktis pembuatan modul ajar, bank soal HOTS, dan ringkasan materi)*
                    """
                )


def render_user_profile_sidebar() -> None:
    """Renders logged-in user details and a logout button at the top of the sidebar."""
    if not st.session_state.get("authenticated", False):
        return

    user_name = st.session_state.get("user_name", "Pengguna")
    user_badge = st.session_state.get("user_badge", "👤 Pengguna")
    user_role = st.session_state.get("user_role", "Pendidik")

    with st.sidebar.container(border=True):
        st.markdown(f"**{user_badge}**")
        st.caption(f"{user_name} ({user_role})")
        
        if st.button("Keluar (Logout)", icon=":material/logout:", width="stretch"):
            # Clear authentication session state
            for key in ["authenticated", "username", "user_name", "user_role", "user_badge"]:
                st.session_state.pop(key, None)
            st.rerun()


def require_authentication() -> bool:
    """Gatekeeper function. Returns True if authenticated, else renders login page and returns False."""
    if not AUTH_ENABLED:
        return True

    if not st.session_state.get("authenticated", False):
        render_login_page()
        return False

    return True
