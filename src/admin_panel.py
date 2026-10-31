from typing import Any, Dict
import pandas as pd
import streamlit as st

from src.config import RAW_PDF_DIR
from src.subject_manager import delete_subject, load_subjects, upsert_subject
from src.user_manager import (
    BADGE_OPTIONS,
    create_user,
    delete_user,
    load_users,
    remove_subject_from_users,
    update_user,
)


def render_admin_management_tab(
    current_user: Dict[str, Any],
    vector_store: Any,
    bm25_store: Any,
    embedder: Any = None,
) -> None:
    """Renders the dynamic Subject (Mapel) and Teacher/User (Guru) management dashboard for Admins."""
    if current_user.get("account_type") != "admin":
        st.error("⛔ Akses ditolak. Menu manajemen ini khusus untuk Administrator.")
        return

    all_subjects = load_subjects()
    all_users = load_users()

    try:
        vec_stats = vector_store.get_stats_by_subject(embedder=embedder)
    except TypeError:
        vec_stats = vector_store.get_stats_by_subject()

    bm25_stats = (
        bm25_store.get_stats_by_subject()
        if hasattr(bm25_store, "get_stats_by_subject")
        else {}
    )

    total_pdfs = 0
    pdf_counts_by_sub: Dict[str, int] = {}
    for sk in all_subjects.keys():
        sdir = RAW_PDF_DIR / sk
        cnt = len(list(sdir.glob("*.pdf"))) if sdir.exists() else 0
        pdf_counts_by_sub[sk] = cnt
        total_pdfs += cnt

    guru_accounts = [u for u in all_users.values() if u.get("account_type") == "guru"]
    admin_accounts = [u for u in all_users.values() if u.get("account_type") == "admin"]

    st.subheader("🛠️ Pusat Manajemen Dinamis Mata Pelajaran & Akun Guru")
    st.caption(
        "Kelola daftar mata pelajaran, cakupan kurikulum, akun pendidik (guru), serta pembagian hak akses mapel secara langsung tanpa menyentuh kode."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📚 Mata Pelajaran Aktif", len(all_subjects))
    m2.metric("👨‍🏫 Akun Guru Terdaftar", len(guru_accounts))
    m3.metric("🛡️ Akun Admin", len(admin_accounts))
    m4.metric("📄 Total Dokumen PDF", total_pdfs)

    subtab_mapel, subtab_guru = st.tabs(
        [
            "📚 Kelola Mata Pelajaran (Mapel)",
            "👨‍🏫 Kelola Akun Guru & Pengguna",
        ]
    )

    # =========================================================================
    # SUB-TAB 1: MANAJEMEN MATA PELAJARAN
    # =========================================================================
    with subtab_mapel:
        st.markdown("#### 📋 Daftar Mata Pelajaran Terdaftar")
        sub_rows = []
        for sk, sinfo in all_subjects.items():
            assigned_gurus = []
            for g in guru_accounts:
                if not g.get("is_active", True):
                    continue
                g_allowed = g.get("allowed_subjects", ["*"])
                if "*" in g_allowed:
                    assigned_gurus.append(f"{g['name']} (Semua)")
                elif sk in g_allowed:
                    assigned_gurus.append(g["name"])

            sub_rows.append(
                {
                    "Kode Mapel": sk,
                    "Nama Mata Pelajaran": sinfo.get("display_name", sk),
                    "Jenjang / Fase": sinfo.get("grade_level", "Kelas X"),
                    "File PDF": pdf_counts_by_sub.get(sk, 0),
                    "Vector Chunks": vec_stats.get(sk, 0),
                    "BM25 Chunks": bm25_stats.get(sk, 0),
                    "Guru Pengampu": ", ".join(assigned_gurus) if assigned_gurus else "-",
                    "Deskripsi Cakupan": sinfo.get("description", ""),
                }
            )
        st.dataframe(pd.DataFrame(sub_rows), width="stretch", hide_index=True)

        st.divider()
        col_add_sub, col_edit_sub = st.columns([1, 1])

        # --- TAMBAH MAPEL BARU ---
        with col_add_sub:
            with st.container(border=True):
                st.markdown("#### ➕ Tambah Mata Pelajaran Baru")
                with st.form("form_add_subject", clear_on_submit=True):
                    new_sub_display = st.text_input(
                        "Nama Mata Pelajaran *:",
                        placeholder="Contoh: Matematika (Kelas X) atau Fisika",
                    )
                    new_sub_grade = st.text_input(
                        "Jenjang / Fase:",
                        value="Kelas X",
                        placeholder="Contoh: Kelas X, Kelas XI, Fase E, Fase F",
                    )
                    new_sub_slug = st.text_input(
                        "Kode Unik Mapel (Opsional):",
                        placeholder="Contoh: matematika (kosongkan agar dibuat otomatis)",
                        help="Digunakan sebagai nama folder penyimpanan PDF dan filter metadata ChromaDB/BM25.",
                    )
                    new_sub_desc = st.text_area(
                        "Deskripsi Materi / Elemen CP:",
                        placeholder="Contoh: Fase E SMA/SMK: Eksponen, Logaritma, Barisan & Deret, Trigonometri",
                        height=85,
                    )
                    new_sub_sample = st.text_area(
                        "Contoh Prompt Default (Opsional):",
                        placeholder="Contoh: Buat modul ajar eksponen dan logaritma beserta asesmen HOTS kontekstual",
                        height=75,
                    )
                    submitted_add_sub = st.form_submit_button(
                        "➕ Tambahkan Mata Pelajaran",
                        type="primary",
                        width="stretch",
                    )

                    if submitted_add_sub:
                        ok, msg, _ = upsert_subject(
                            subject_key=new_sub_slug,
                            display_name=new_sub_display,
                            description=new_sub_desc,
                            grade_level=new_sub_grade,
                            sample_query=new_sub_sample,
                            is_new=True,
                        )
                        if ok:
                            st.toast(msg, icon="✅")
                            st.rerun()
                        else:
                            st.error(msg)

        # --- EDIT / HAPUS MAPEL ---
        with col_edit_sub:
            with st.container(border=True):
                st.markdown("#### ✏️ Edit / Hapus Mata Pelajaran")
                sub_keys_list = list(all_subjects.keys())
                edit_sub_key = st.selectbox(
                    "Pilih Mata Pelajaran:",
                    sub_keys_list,
                    format_func=lambda k: f"{all_subjects[k]['display_name']} [{k}]",
                    key="admin_edit_subject_selector",
                )
                curr_sub_info = all_subjects[edit_sub_key]

                with st.form(f"form_edit_subject_{edit_sub_key}"):
                    ed_display = st.text_input(
                        "Nama Mata Pelajaran *:",
                        value=curr_sub_info.get("display_name", ""),
                    )
                    ed_grade = st.text_input(
                        "Jenjang / Fase:",
                        value=curr_sub_info.get("grade_level", "Kelas X"),
                    )
                    ed_desc = st.text_area(
                        "Deskripsi Materi / Elemen CP:",
                        value=curr_sub_info.get("description", ""),
                        height=85,
                    )
                    ed_sample = st.text_area(
                        "Contoh Prompt Default:",
                        value=curr_sub_info.get("sample_query", ""),
                        height=75,
                    )
                    submitted_edit_sub = st.form_submit_button(
                        "💾 Simpan Perubahan Mapel",
                        width="stretch",
                    )
                    if submitted_edit_sub:
                        ok, msg, _ = upsert_subject(
                            subject_key=edit_sub_key,
                            display_name=ed_display,
                            description=ed_desc,
                            grade_level=ed_grade,
                            sample_query=ed_sample,
                            is_new=False,
                        )
                        if ok:
                            st.toast(msg, icon="✅")
                            st.rerun()
                        else:
                            st.error(msg)

                with st.expander("🗑️ Hapus Mata Pelajaran Ini", expanded=False):
                    if len(sub_keys_list) <= 1:
                        st.info("Minimal harus terdapat 1 mata pelajaran aktif di dalam sistem.")
                    else:
                        purge_sub_data = st.checkbox(
                            "Hapus juga seluruh file PDF fisik serta indeks Vector & BM25 mapel ini",
                            value=False,
                            key=f"purge_sub_{edit_sub_key}",
                        )
                        confirm_sub_del = st.checkbox(
                            f"Saya yakin ingin menghapus mapel '{curr_sub_info.get('display_name', edit_sub_key)}'",
                            value=False,
                            key=f"confirm_del_sub_{edit_sub_key}",
                        )
                        if st.button(
                            "🗑️ Hapus Mata Pelajaran",
                            key=f"btn_del_sub_{edit_sub_key}",
                            width="stretch",
                        ):
                            if not confirm_sub_del:
                                st.warning("Centang konfirmasi penghapusan terlebih dahulu.")
                            else:
                                ok, msg = delete_subject(
                                    subject_key=edit_sub_key,
                                    purge_data=purge_sub_data,
                                    vector_store=vector_store,
                                    bm25_store=bm25_store,
                                )
                                if ok:
                                    remove_subject_from_users(edit_sub_key)
                                    st.toast(msg, icon="🗑️")
                                    st.rerun()
                                else:
                                    st.error(msg)

    # =========================================================================
    # SUB-TAB 2: MANAJEMEN AKUN GURU & PENGGUNA
    # =========================================================================
    with subtab_guru:
        st.markdown("#### 👥 Daftar Akun Guru & Administrator")
        user_rows = []
        for uname, uinfo in all_users.items():
            acc_t = uinfo.get("account_type", "guru")
            allowed_s = uinfo.get("allowed_subjects", ["*"])
            if acc_t == "admin" or "*" in allowed_s:
                allowed_str = "Semua Mapel (*)"
            else:
                mapped = [all_subjects[s]["display_name"] for s in allowed_s if s in all_subjects]
                allowed_str = ", ".join(mapped) if mapped else "⚠️ Belum ada mapel aktif"

            user_rows.append(
                {
                    "Username": uname,
                    "Nama Lengkap": uinfo.get("name", uname),
                    "Tipe Akun": "🛡️ Admin" if acc_t == "admin" else "👨‍🏫 Guru",
                    "Jabatan / Peran": uinfo.get("role", ""),
                    "Akses Mata Pelajaran": allowed_str,
                    "Status": "✅ Aktif" if uinfo.get("is_active", True) else "⛔ Nonaktif",
                }
            )
        st.dataframe(pd.DataFrame(user_rows), width="stretch", hide_index=True)

        st.divider()
        col_add_usr, col_edit_usr = st.columns([1, 1])

        # --- TAMBAH AKUN GURU / PENGGUNA BARU ---
        with col_add_usr:
            with st.container(border=True):
                st.markdown("#### ➕ Tambah Akun Guru / Pengguna Baru")
                new_acc_type = st.radio(
                    "Tipe Akun:",
                    ["guru", "admin"],
                    format_func=lambda x: "👨‍🏫 Guru / Pendidik" if x == "guru" else "🛡️ Admin / Peneliti",
                    horizontal=True,
                    key="add_usr_acc_type",
                )

                new_scope_all = True
                if new_acc_type == "guru":
                    scope_choice = st.radio(
                        "Hak Akses Mata Pelajaran:",
                        ["specific", "all"],
                        format_func=lambda x: "🎯 Pilih Mapel Tertentu" if x == "specific" else "🌐 Semua Mata Pelajaran",
                        horizontal=True,
                        key="add_usr_scope_choice",
                    )
                    new_scope_all = scope_choice == "all"

                with st.form("form_create_user", clear_on_submit=True):
                    c_uname = st.text_input(
                        "Username Login *:",
                        placeholder="Contoh: guru_matematika atau budi.santoso",
                    )
                    c_name = st.text_input(
                        "Nama Lengkap & Gelar *:",
                        placeholder="Contoh: Budi Santoso, S.Pd.",
                    )
                    c_pass = st.text_input(
                        "Kata Sandi Awal *:",
                        type="password",
                        placeholder="Minimal 4 karakter",
                    )

                    selected_new_subs = ["*"]
                    if new_acc_type == "guru" and not new_scope_all:
                        selected_new_subs = st.multiselect(
                            "Pilih Mata Pelajaran yang Diampu *:",
                            options=list(all_subjects.keys()),
                            default=[list(all_subjects.keys())[0]] if all_subjects else [],
                            format_func=lambda k: all_subjects[k]["display_name"],
                        )

                    default_role_val = (
                        "Pendidik (Modul & Asesmen)"
                        if new_acc_type == "guru"
                        else "Peneliti (Full Access)"
                    )
                    c_role = st.text_input(
                        "Label Jabatan / Peran:",
                        value=default_role_val,
                    )

                    default_badge_idx = 0 if new_acc_type == "guru" else 2
                    c_badge = st.selectbox(
                        "Badge Tampilan:",
                        BADGE_OPTIONS,
                        index=default_badge_idx,
                    )

                    submitted_create_usr = st.form_submit_button(
                        "➕ Buat Akun Pengguna",
                        type="primary",
                        width="stretch",
                    )
                    if submitted_create_usr:
                        final_allowed = ["*"] if (new_acc_type == "admin" or new_scope_all) else selected_new_subs
                        ok, msg = create_user(
                            username=c_uname,
                            name=c_name,
                            password=c_pass,
                            account_type=new_acc_type,
                            role=c_role,
                            badge=c_badge,
                            allowed_subjects=final_allowed,
                            is_active=True,
                        )
                        if ok:
                            st.toast(msg, icon="✅")
                            st.rerun()
                        else:
                            st.error(msg)

        # --- EDIT / RESET PASSWORD / HAPUS PENGGUNA ---
        with col_edit_usr:
            with st.container(border=True):
                st.markdown("#### ✏️ Edit Akun, Penugasan Mapel & Reset Sandi")
                user_keys_list = list(all_users.keys())
                edit_uname = st.selectbox(
                    "Pilih Akun Pengguna:",
                    user_keys_list,
                    format_func=lambda u: f"{all_users[u]['name']} (@{u}) — {'Admin' if all_users[u].get('account_type') == 'admin' else 'Guru'}",
                    key="admin_edit_user_selector",
                )
                target_u = all_users[edit_uname]
                is_self = edit_uname == current_user.get("username")

                curr_acc_type = target_u.get("account_type", "guru")
                ed_acc_type = st.radio(
                    "Tipe Akun:",
                    ["guru", "admin"],
                    index=0 if curr_acc_type == "guru" else 1,
                    format_func=lambda x: "👨‍🏫 Guru / Pendidik" if x == "guru" else "🛡️ Admin / Peneliti",
                    horizontal=True,
                    disabled=is_self,
                    key=f"ed_acc_type_{edit_uname}",
                    help="Anda tidak dapat menurunkan tipe akun Anda sendiri." if is_self else None,
                )

                curr_allowed = target_u.get("allowed_subjects", ["*"])
                curr_is_all = "*" in curr_allowed
                ed_scope_all = True
                if ed_acc_type == "guru":
                    ed_scope_mode = st.radio(
                        "Cakupan Hak Akses Mapel:",
                        ["specific", "all"],
                        index=1 if curr_is_all else 0,
                        format_func=lambda x: "🎯 Pilih Mapel Tertentu" if x == "specific" else "🌐 Semua Mata Pelajaran",
                        horizontal=True,
                        key=f"ed_scope_mode_{edit_uname}",
                    )
                    ed_scope_all = ed_scope_mode == "all"

                with st.form(f"form_edit_user_{edit_uname}"):
                    ed_name = st.text_input(
                        "Nama Lengkap & Gelar *:",
                        value=target_u.get("name", ""),
                    )

                    ed_selected_subs = ["*"]
                    if ed_acc_type == "guru" and not ed_scope_all:
                        valid_defaults = [s for s in curr_allowed if s in all_subjects]
                        ed_selected_subs = st.multiselect(
                            "Daftar Mata Pelajaran yang Diampu *:",
                            options=list(all_subjects.keys()),
                            default=valid_defaults,
                            format_func=lambda k: all_subjects[k]["display_name"],
                        )

                    ed_role = st.text_input(
                        "Label Jabatan / Peran:",
                        value=target_u.get("role", ""),
                    )

                    curr_badge = target_u.get("badge", BADGE_OPTIONS[0])
                    badge_options_ed = (
                        BADGE_OPTIONS
                        if curr_badge in BADGE_OPTIONS
                        else [curr_badge] + BADGE_OPTIONS
                    )
                    ed_badge = st.selectbox(
                        "Badge Tampilan:",
                        badge_options_ed,
                        index=badge_options_ed.index(curr_badge),
                    )

                    ed_active = st.checkbox(
                        "Akun Aktif (Izinkan Login)",
                        value=target_u.get("is_active", True),
                        disabled=is_self,
                    )

                    ed_new_pass = st.text_input(
                        "Reset Kata Sandi Baru (Opsional):",
                        type="password",
                        placeholder="Kosongkan jika tidak ingin mengubah kata sandi",
                    )

                    submitted_edit_usr = st.form_submit_button(
                        "💾 Simpan Perubahan Akun",
                        width="stretch",
                    )
                    if submitted_edit_usr:
                        final_ed_subs = (
                            ["*"]
                            if (ed_acc_type == "admin" or ed_scope_all)
                            else ed_selected_subs
                        )
                        ok, msg = update_user(
                            username=edit_uname,
                            name=ed_name,
                            account_type=ed_acc_type,
                            role=ed_role,
                            badge=ed_badge,
                            allowed_subjects=final_ed_subs,
                            is_active=ed_active,
                            new_password=ed_new_pass,
                            actor_username=current_user.get("username"),
                        )
                        if ok:
                            st.toast(msg, icon="✅")
                            st.rerun()
                        else:
                            st.error(msg)

                with st.expander("🗑️ Hapus Akun Pengguna", expanded=False):
                    if is_self:
                        st.info("🔒 Anda tidak dapat menghapus akun Anda sendiri yang sedang login.")
                    else:
                        conf_del_u = st.checkbox(
                            f"Saya yakin ingin menghapus akun '{target_u.get('name', edit_uname)}' (@{edit_uname})",
                            value=False,
                            key=f"conf_del_usr_{edit_uname}",
                        )
                        if st.button(
                            "🗑️ Hapus Akun Permanen",
                            key=f"btn_del_usr_{edit_uname}",
                            width="stretch",
                        ):
                            if not conf_del_u:
                                st.warning("Centang konfirmasi penghapusan terlebih dahulu.")
                            else:
                                ok, msg = delete_user(
                                    username=edit_uname,
                                    actor_username=current_user.get("username"),
                                )
                                if ok:
                                    st.toast(msg, icon="🗑️")
                                    st.rerun()
                                else:
                                    st.error(msg)
