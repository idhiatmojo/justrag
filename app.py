import os
import importlib
from pathlib import Path
import pandas as pd
import streamlit as st

import src.config
import src.subject_manager
import src.user_manager
import src.storage.vector_store
import src.storage.bm25_store
import src.llm_factory
import src.auth
import src.ingestion.pipeline
import src.generation.generator
import src.admin_panel

importlib.reload(src.config)
importlib.reload(src.subject_manager)
importlib.reload(src.user_manager)
importlib.reload(src.storage.vector_store)
importlib.reload(src.storage.bm25_store)
importlib.reload(src.llm_factory)
importlib.reload(src.auth)
importlib.reload(src.ingestion.pipeline)
importlib.reload(src.generation.generator)
importlib.reload(src.admin_panel)

from src.config import RAW_PDF_DIR
from src.subject_manager import load_subjects
from src.user_manager import get_accessible_subjects_for_user
from src.auth import (
    get_current_user,
    is_current_user_admin,
    render_user_profile_sidebar,
    require_authentication,
)
from src.admin_panel import render_admin_management_tab
from src.llm_factory import get_llm_client
from src.storage.vector_store import ChromaVectorStore
from src.storage.bm25_store import BM25SubjectStore
from src.ingestion.pipeline import DynamicIngestionPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import LexicalSemanticReranker
from src.generation.generator import PedagogicalGenerator
from src.evaluation.evaluator import SystematicRAGEvaluator
from src.evaluation.synthetic_data import EvaluationDatasetManager
from src.custom_providers import (
    delete_custom_preset,
    fetch_available_models,
    load_custom_presets,
    save_custom_preset,
    test_custom_endpoint,
)

st.set_page_config(
    page_title="RAG Generatif Mapel Fase E - S2 Project",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Authentication Gate
if not require_authentication():
    st.stop()

current_user = get_current_user()
is_admin = is_current_user_admin()
all_subjects = load_subjects()
accessible_subjects = get_accessible_subjects_for_user(current_user, all_subjects)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.title("🎓 Konfigurasi Sistem RAG")
st.sidebar.caption("Prototipe Penelitian S2: Generatif Bahan Ajar & Asesmen Terisolasi Domain")

# User Profile & Logout Section
render_user_profile_sidebar()

if not accessible_subjects:
    st.title("📚 Pipeline RAG Generatif Kurikulum Merdeka")
    st.warning(
        "⚠️ Akun Anda saat ini belum ditugaskan ke mata pelajaran aktif mana pun. "
        "Silakan hubungi Administrator untuk mengatur penugasan mata pelajaran pada akun Anda."
    )
    st.stop()

# 1. Subject Selection (Filtered by User Role & Assignment)
st.sidebar.subheader("1. Pilih Mata Pelajaran")
subject_keys = list(accessible_subjects.keys())
subject_labels = [accessible_subjects[k]["display_name"] for k in subject_keys]

selected_idx = st.sidebar.selectbox(
    "Mata Pelajaran Aktif:",
    range(len(subject_keys)),
    format_func=lambda i: subject_labels[i],
)
selected_subject = subject_keys[selected_idx]
selected_subject_info = accessible_subjects[selected_subject]
st.sidebar.info(selected_subject_info["description"])

# 2. LLM Provider & Model Selection
st.sidebar.subheader("2. Pilihan Model LLM")
providers_list = ["custom", "gemini", "openai", "grok", "mock"]
saved_prov = st.query_params.get("provider", "custom")
def_idx = providers_list.index(saved_prov) if saved_prov in providers_list else 0

provider = st.sidebar.selectbox(
    "LLM Provider:",
    providers_list,
    index=def_idx,
    help="Pilih provider LLM: Custom (OpenAI-compatible / TamanData / JuanRouter), Gemini, OpenAI, Grok (xAI), atau simulasi Mock.",
    key="sb_llm_provider",
)
st.query_params["provider"] = provider

embedding_provider_input: str | None = None
embedding_api_key_input: str | None = None
base_url_input: str | None = None

if provider == "gemini":
    model_name = st.sidebar.selectbox(
        "Varian Model Gemini:",
        ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-flash-latest"],
        index=0,
    )
    if is_admin:
        api_key_input = st.sidebar.text_input(
            "Google Gemini API Key:",
            value=os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
            type="password",
            help="Masukkan Gemini API Key Anda. Jika kosong, sistem otomatis beralih ke mode simulasi aman.",
        )
    else:
        api_key_input = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    st.sidebar.caption("⚙️ Embedding Provider (untuk Vector DB):")
    embedding_provider_input = st.sidebar.selectbox(
        "Pilih Embedding Provider:",
        ["local", "gemini"],
        index=0,
        help="Pilihan: 'local' (FastEmbed Neural 384-dim, mengakses korpus lokal tanpa API key), atau 'gemini' (Google Embedding API 3072-dim).",
        key="gemini_emb_provider",
    )
    if embedding_provider_input == "local":
        st.sidebar.success("⚡ FastEmbed Neural Lokal aktif (384 dim). Mengakses korpus offline yang tersimpan!")
elif provider == "openai":
    model_name = st.sidebar.selectbox(
        "Varian Model OpenAI:",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
    )
    if is_admin:
        api_key_input = st.sidebar.text_input(
            "OpenAI API Key:",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
    else:
        api_key_input = os.getenv("OPENAI_API_KEY", "")
    st.sidebar.caption("⚙️ Embedding Provider (untuk Vector DB):")
    embedding_provider_input = st.sidebar.selectbox(
        "Pilih Embedding Provider:",
        ["local", "openai"],
        index=0,
        help="Pilihan: 'local' (FastEmbed Neural 384-dim, mengakses korpus lokal tanpa API key), atau 'openai' (text-embedding-3-small).",
        key="openai_emb_provider",
    )
    if embedding_provider_input == "local":
        st.sidebar.success("⚡ FastEmbed Neural Lokal aktif (384 dim). Mengakses korpus offline yang tersimpan!")
elif provider == "grok":
    model_name = st.sidebar.selectbox(
        "Varian Model Grok (xAI):",
        ["grok-3-mini", "grok-3", "grok-2"],
        index=0,
        help="grok-3-mini lebih hemat; grok-3 untuk respons terbaik.",
    )
    if is_admin:
        api_key_input = st.sidebar.text_input(
            "xAI API Key:",
            value=os.getenv("XAI_API_KEY", "") or os.getenv("GROK_API_KEY", ""),
            type="password",
            help="Dapatkan di https://console.x.ai",
        )
    else:
        api_key_input = os.getenv("XAI_API_KEY", "") or os.getenv("GROK_API_KEY", "")
    st.sidebar.caption("⚙️ Embedding Provider (Grok tidak punya model embedding sendiri):")
    embedding_provider_input = st.sidebar.selectbox(
        "Pilih Embedding Provider:",
        ["gemini", "openai", "local"],
        index=2 if not is_admin else 0,
        help="Pilihan: 'gemini' (cloud), 'openai' (cloud), atau 'local' (FastEmbed Neural all-MiniLM-L6-v2, 100% offline & tanpa API key).",
        key="grok_emb_provider",
    )
    if embedding_provider_input == "gemini":
        embedding_api_key_input = (
            st.sidebar.text_input(
                "Gemini API Key (untuk embedding):",
                value=os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
                type="password",
                key="grok_emb_key",
            )
            if is_admin
            else (os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""))
        )
    elif embedding_provider_input == "openai":
        embedding_api_key_input = (
            st.sidebar.text_input(
                "OpenAI API Key (untuk embedding):",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
                key="grok_emb_key_oai",
            )
            if is_admin
            else os.getenv("OPENAI_API_KEY", "")
        )
    else:
        st.sidebar.success("⚡ FastEmbed Neural Lokal aktif (all-MiniLM-L6-v2, 384 dim ONNX). 100% offline tanpa API key.")
elif provider == "custom":
    presets = load_custom_presets()
    preset_names = list(presets.keys())
    preset_options = (preset_names + ["➕ Buat Preset Baru..."]) if is_admin else (preset_names or ["Default"])

    if "custom_preset_choice" not in st.session_state or st.session_state.custom_preset_choice not in preset_options:
        st.session_state.custom_preset_choice = preset_options[0]

    selected_preset = st.sidebar.selectbox(
        "Pilih Preset Endpoint Tersimpan:",
        preset_options,
        index=preset_options.index(st.session_state.custom_preset_choice)
        if st.session_state.custom_preset_choice in preset_options
        else 0,
        help="Preset disimpan permanen di data/custom_providers.json",
        key="sb_preset_selector",
    )
    st.session_state.custom_preset_choice = selected_preset
    is_new = is_admin and (selected_preset == "➕ Buat Preset Baru...")

    curr_preset = {}
    if not is_new:
        curr_preset = presets.get(selected_preset, {})
        def_url = curr_preset.get("base_url", "https://api.juanrouter.com/v1")
        def_key = curr_preset.get("api_key", "")
        def_model = curr_preset.get("model_name", "qwen3.8-max")
        def_emb = curr_preset.get("embedding_provider", "gemini")
    else:
        def_url = "https://api.juanrouter.com/v1"
        def_key = ""
        def_model = "qwen3.8-max"
        def_emb = "gemini"

    if is_admin:
        if is_new:
            preset_name_input = st.sidebar.text_input(
                "Nama Preset Baru:",
                value="Custom Gateway",
                help="Nama pengenal untuk disimpan ke daftar preset Anda",
            )
        else:
            preset_name_input = selected_preset

        base_url_input = st.sidebar.text_input(
            "API Endpoint (Base URL):",
            value=def_url,
            key=f"url_input_{selected_preset}",
            help="Endpoint OpenAI-compatible (contoh: https://api.juanrouter.com/v1, https://api.deepseek.com/v1, http://localhost:11434/v1)",
        )

        api_key_input = st.sidebar.text_input(
            "API Key:",
            value=def_key,
            type="password",
            key=f"key_input_{selected_preset}",
            help="API Key provider custom Anda",
        )

        # Tarik model dari server
        col_fetch, col_test = st.sidebar.columns([1, 1])
        with col_fetch:
            if st.button("🔄 Tarik Model", width="stretch", help="Tarik daftar model aktif langsung dari endpoint"):
                with st.spinner("Mengambil..."):
                    found_models = fetch_available_models(base_url_input, api_key_input)
                    if found_models:
                        st.session_state[f"fetched_models_{selected_preset}"] = found_models
                        st.sidebar.success(f"Ditemukan {len(found_models)} model!")
                    else:
                        st.sidebar.warning(
                            "Gagal menarik daftar model otomatis (atau endpoint tidak mendukung /v1/models). Silakan ketik nama model secara manual."
                        )
    else:
        base_url_input = def_url
        api_key_input = def_key
        col_test = None

    available_preset_models = st.session_state.get(f"fetched_models_{selected_preset}") or curr_preset.get(
        "available_models", []
    )
    if available_preset_models:
        extra_opts = ["Ketik manual..."] if is_admin else []
        model_choice_list = list(dict.fromkeys(available_preset_models + extra_opts))
        m_idx = model_choice_list.index(def_model) if def_model in model_choice_list else 0
        chosen_m = st.sidebar.selectbox("Pilih Model:", model_choice_list, index=m_idx, key=f"sel_m_{selected_preset}")
        if chosen_m == "Ketik manual...":
            model_name = st.sidebar.text_input("Nama Model Manual:", value=def_model, key=f"manual_m_{selected_preset}")
        else:
            model_name = chosen_m
    else:
        model_name = st.sidebar.text_input(
            "Nama Model:",
            value=def_model,
            key=f"model_input_{selected_preset}",
            disabled=not is_admin,
            help="Contoh: qwen3.8-max, deepseek-v4-pro, llama3.2",
        )

    if is_admin and col_test is not None:
        with col_test:
            if st.button("🔌 Tes Koneksi", width="stretch", help="Kirim ping 1 token ke model"):
                with st.spinner("Menguji..."):
                    ok, msg = test_custom_endpoint(base_url_input, api_key_input, model_name)
                    if ok:
                        st.sidebar.success(msg)
                    else:
                        st.sidebar.error(msg)

    # Embedding Provider
    st.sidebar.caption("⚙️ Embedding Provider (untuk Vector DB):")
    emb_opts = ["gemini", "openai", "local"]
    emb_idx = emb_opts.index(def_emb) if def_emb in emb_opts else 0
    embedding_provider_input = st.sidebar.selectbox(
        "Pilih Embedding Provider:",
        emb_opts,
        index=emb_idx,
        key=f"emb_prov_{selected_preset}",
        help="Pilihan: 'gemini' (cloud), 'openai' (cloud), atau 'local' (FastEmbed Neural all-MiniLM-L6-v2, 100% offline & tanpa API key).",
    )

    if embedding_provider_input == "gemini":
        embedding_api_key_input = (
            st.sidebar.text_input(
                "Gemini API Key (untuk embedding):",
                value=os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
                type="password",
                key=f"custom_emb_key_gemini_{selected_preset}",
            )
            if is_admin
            else (os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""))
        )
    elif embedding_provider_input == "openai":
        embedding_api_key_input = (
            st.sidebar.text_input(
                "OpenAI API Key (untuk embedding):",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
                key=f"custom_emb_key_oai_{selected_preset}",
            )
            if is_admin
            else os.getenv("OPENAI_API_KEY", "")
        )
    else:
        st.sidebar.success("⚡ FastEmbed Neural Lokal aktif (all-MiniLM-L6-v2, 384 dim ONNX). 100% offline tanpa API key.")

    # Tombol Simpan & Hapus Preset (Khusus Admin)
    if is_admin:
        col_save, col_del = st.sidebar.columns([1, 1])
        with col_save:
            if st.button("💾 Simpan", width="stretch", help="Simpan preset ini ke file data/custom_providers.json"):
                target_save_name = preset_name_input.strip()
                if not target_save_name:
                    st.sidebar.error("Nama preset tidak boleh kosong!")
                else:
                    save_custom_preset(
                        target_save_name,
                        {
                            "base_url": base_url_input,
                            "api_key": api_key_input,
                            "model_name": model_name,
                            "embedding_provider": embedding_provider_input,
                            "available_models": available_preset_models,
                        },
                    )
                    st.session_state.custom_preset_choice = target_save_name
                    st.sidebar.success(f"Preset '{target_save_name}' tersimpan permanen!")
                    st.rerun()

        with col_del:
            if not is_new:
                if st.button("🗑️ Hapus", width="stretch", help="Hapus preset ini dari penyimpanan"):
                    delete_custom_preset(selected_preset)
                    remaining = list(load_custom_presets().keys())
                    st.session_state.custom_preset_choice = remaining[0] if remaining else "➕ Buat Preset Baru..."
                    st.sidebar.warning(f"Preset '{selected_preset}' dihapus.")
                    st.rerun()
else:
    model_name = "mock-pedagogy-v1"
    api_key_input = "mock-key"
    st.sidebar.warning("Mode simulasi lokal aktif. Hasil respons bersifat representatif offline.")

# Instantiate LLM Client
llm_client = get_llm_client(
    provider=provider,
    model_name=model_name,
    api_key=api_key_input if api_key_input else None,
    base_url=base_url_input if base_url_input else None,
    embedding_provider=embedding_provider_input,
    embedding_api_key=embedding_api_key_input if embedding_api_key_input else None,
)

# Initialize Storage & Engines
vector_store = ChromaVectorStore()
bm25_store = BM25SubjectStore()
ingestion_pipeline = DynamicIngestionPipeline(vector_store=vector_store, bm25_store=bm25_store)
retriever = HybridRetriever(vector_store=vector_store, bm25_store=bm25_store, embedder=llm_client)
reranker = LexicalSemanticReranker()
generator = PedagogicalGenerator(llm_client=llm_client)

# 3. Dynamic PDF Ingestion Section (Tersedia untuk Admin & Guru pada mapel yang diizinkan)
st.sidebar.subheader("3. Dynamic Ingestion PDF")
uploaded_files = st.sidebar.file_uploader(
    f"Upload PDF Baru untuk {selected_subject_info['display_name']}:",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.sidebar.button("📥 Proses & Indeks PDF ke Vector DB", width="stretch"):
        with st.sidebar.status("Memproses dokumen...", expanded=True) as status:
            target_dir = RAW_PDF_DIR / selected_subject
            target_dir.mkdir(parents=True, exist_ok=True)
            subject_grade = selected_subject_info.get("grade_level", "Kelas X")

            for uf in uploaded_files:
                clean_name = Path(uf.name).name
                save_path = target_dir / clean_name
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())

                st.write(f"Indexing `{clean_name}`...")
                try:
                    res = ingestion_pipeline.ingest_single_pdf(
                        pdf_path=save_path,
                        subject=selected_subject,
                        embedder=llm_client,
                        grade_level=subject_grade,
                    )
                    st.write(f"✅ Selesai: {res['pages']} hal, {res['chunks']} chunks.")
                except Exception as e:
                    st.error(f"❌ Gagal memproses `{clean_name}`: {e}")
            status.update(label="Proses indeks selesai!", state="complete")

# Stats per subject & active collection info
try:
    stats = vector_store.get_stats_by_subject(embedder=llm_client)
except TypeError:
    stats = vector_store.get_stats_by_subject()

active_col_info = vector_store.get_active_collection_info(embedder=llm_client)
col_name = active_col_info["name"]
col_dim = active_col_info["dimension"]
col_label = (
    "FastEmbed Local (384-dim)"
    if col_dim == 384
    else ("Gemini Cloud (3072-dim)" if col_dim == 3072 else f"{col_dim}-dim")
)

st.sidebar.caption(f"📊 Status Korpus Aktif ({col_label}):")
for sk in accessible_subjects.keys():
    st.sidebar.text(f"• {sk.upper()}: {stats.get(sk, 0)} chunks")

# BM25 Lexical Stats
if hasattr(bm25_store, "get_stats_by_subject"):
    bm25_stats = bm25_store.get_stats_by_subject()
else:
    bm25_stats = {sub: len(chunks) for sub, chunks in getattr(bm25_store, "chunk_stores", {}).items()}
bm25_curr = bm25_stats.get(selected_subject, 0)
st.sidebar.caption(f"🗂️ Indeks Leksikal BM25 ({selected_subject.upper()}): {bm25_curr} chunks")

# Show other collections if they exist (Admin view)
if is_admin:
    all_summaries = vector_store.get_all_collections_summary()
    other_cols = [s for s in all_summaries if s["name"] != col_name and s["count"] > 0]
    if other_cols:
        with st.sidebar.expander("ℹ️ Koleksi Lain di ChromaDB", expanded=False):
            for oc in other_cols:
                st.caption(f"• **`{oc['name']}`**: {oc['count']} chunks")
            st.caption("Ubah Embedding Provider di atas untuk mengaktifkan koleksi terkait.")

# ----------------- MAIN UI TABS (ROLE-BASED) -----------------
st.title("📚 Pipeline RAG Generatif Kurikulum Merdeka")
st.caption("Implementasi End-to-End dengan Isolasi Metadata Domain, Manajemen Multi-Mapel & Evaluasi Saintifik")

if is_admin:
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🚀 Generatif Bahan Ajar & Asesmen",
            "📊 Metodologi Evaluasi & Ablation Study",
            "📂 Manajemen Dokumen & Analisis Korpus",
            "🛠️ Manajemen Mapel & Guru",
        ]
    )
else:
    tab1, tab3 = st.tabs(
        [
            "🚀 Generatif Bahan Ajar & Asesmen",
            "📂 Dokumen Materi Mapel",
        ]
    )
    tab2 = None
    tab4 = None

# ----------------- TAB 1: GENERATIF -----------------
with tab1:
    col_mode, col_task = st.columns([1, 1])
    with col_mode:
        task_mode = st.selectbox(
            "Pilih Mode Generasi Pedagogis:",
            [
                ("modul_ajar", "📝 Modul Ajar (Kurikulum Merdeka)"),
                ("soal_hots", "🎯 Generator Paket Soal HOTS (Bloom C4-C6)"),
                ("ringkasan", "📖 Ringkasan Konsep & Peta Materi"),
                ("qa_chat", "💬 Tanya Jawab Terpandu (Q&A)"),
            ],
            format_func=lambda x: x[1],
        )[0]

    with col_task:
        retrieval_strategy = st.selectbox(
            "Strategi Retrieval:",
            [
                ("hybrid", "Hybrid Search (Dense + BM25 dengan RRF)"),
                ("dense_only", "Dense Vector Search Only"),
                ("bm25_only", "BM25 Sparse Lexical Only"),
            ],
            format_func=lambda x: x[1],
        )[0]

    with st.expander("⚙️ Pengaturan Parameter Pipeline (Ablation Toggles)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            enable_filter = st.checkbox(
                "Isolasi Filter Mapel",
                value=True,
                disabled=not is_admin,
                help="Mencegah kontaminasi dokumen mapel lain.",
            )
        with c2:
            enable_rerank = st.checkbox(
                "Aktifkan Re-ranking (Cross-Scoring)",
                value=True,
                help="Menyaring Top-K chunk sebelum masuk konteks.",
            )
        with c3:
            temperature = st.slider("Temperature Generasi:", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    default_sample_prompt = selected_subject_info.get(
        "sample_query",
        f"Buat modul ajar {selected_subject_info['display_name']} beserta asesmen soal HOTS kontekstual",
    )

    user_query = st.text_area(
        "Instruksi / Topik yang Ingin Dihasilkan:",
        value=default_sample_prompt,
        height=100,
        key=f"user_query_input_{selected_subject}",
    )

    if st.button("✨ Hasilkan Konten dengan RAG", type="primary", width="stretch"):
        if not user_query.strip():
            st.warning("Silakan masukkan topik atau instruksi.")
        else:
            with st.spinner("Menjalankan pipeline retrieval dan inferensi LLM..."):
                # 1. Retrieve (Enforce subject isolation for non-admin accounts)
                effective_filter = enable_filter if is_admin else True
                candidate_k = 6 if enable_rerank else 4
                raw_chunks = retriever.retrieve(
                    query=user_query,
                    subject=selected_subject,
                    mode=retrieval_strategy,
                    use_subject_filter=effective_filter,
                    top_k=candidate_k,
                )

                # 2. Rerank
                if enable_rerank and raw_chunks:
                    final_chunks = reranker.rerank(query=user_query, candidates=raw_chunks, top_k=4)
                else:
                    final_chunks = raw_chunks[:4]

                # 3. Generate
                try:
                    output = generator.generate(
                        query=user_query,
                        subject=selected_subject,
                        retrieved_chunks=final_chunks,
                        task_type=task_mode,
                        temperature=temperature,
                    )
                    # Display Output
                    st.subheader("📋 Hasil Generasi Pedagogis:")
                    st.markdown(output["response"])
                except Exception as e:
                    st.error(f"❌ Terjadi kendala dari provider LLM: {e}")
                    st.info(
                        "💡 **Solusi Gateway Timeout (504) / Gangguan Server:**\n"
                        "- Router gateway publik (`ai.tamandata.com`) sempat melebihi batas waktu respon.\n"
                        "- Silakan klik kembali tombol **✨ Hasilkan Konten dengan RAG** (otomatis melakukan retry).\n"
                        "- Atau ganti ke model yang lebih cepat/stabil di dropdown sidebar, misalnya **`tamandata`**, **`cx/gpt-5.6-sol`**, atau **`solar-pro4`**."
                    )
                    output = None

                # Display Citations / Grounding
                st.divider()
                st.subheader("🔍 Bukti Grounding & Konteks Dokumen Terambil:")
                if final_chunks:
                    for i, chunk in enumerate(final_chunks, 1):
                        meta = chunk.get("metadata", {})
                        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0.0)))
                        with st.expander(
                            f"Kutipan #{i} | Sumber: `{meta.get('source_file')}` (Hal. {meta.get('page_number')}) | Skor: {score:.4f}"
                        ):
                            st.write(chunk.get("text"))
                else:
                    st.info("Tidak ada konteks yang terambil. Pastikan PDF materi mapel ini sudah diunggah & diindeks.")

# ----------------- TAB 2: EVALUATION (KHUSUS ADMIN) -----------------
if is_admin and tab2 is not None:
    with tab2:
        st.subheader("🔬 Metodologi Evaluasi Saintifik & Studi Komparatif (Ablation Benchmark)")
        st.markdown(
            """
        Sesuai standar riset S2, pengujian dilakukan dengan membandingkan 4 konfigurasi pipeline pada **Golden Testset** terstandarisasi:
        1. **Baseline (Naive RAG)**: Fixed Dense Retrieval tanpa filter mapel (menguji terjadinya domain cross-contamination).
        2. **Model A (Metadata Filter)**: Dense Retrieval dengan filter mapel aktif.
        3. **Model B (Hybrid Search)**: Dense + BM25 Lexical dengan fusi peringkat RRF.
        4. **Model C (Proposed Full Pipeline)**: Hybrid + Filter + Re-ranking + Contextual Grounded Prompting.
        """
        )

        dataset_mgr = EvaluationDatasetManager()
        all_test_cases = dataset_mgr.load_dataset()

        with st.expander(f"📝 Kelola Golden Testset Evaluasi ({len(all_test_cases)} Butir Uji)", expanded=False):
            if all_test_cases:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "ID": tc.get("id", ""),
                                "Mapel": tc.get("subject", ""),
                                "Topik": tc.get("topic", ""),
                                "Bloom": tc.get("bloom_level", "C4"),
                                "Pertanyaan": tc.get("question", ""),
                            }
                            for tc in all_test_cases
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

            st.markdown("##### ➕ Tambah Butir Soal Uji Baru (Untuk Mapel Baru / Tambahan)")
            with st.form("form_add_golden_testcase", clear_on_submit=True):
                tc_c1, tc_c2, tc_c3 = st.columns([2, 2, 1])
                with tc_c1:
                    tc_sub = st.selectbox(
                        "Mata Pelajaran:",
                        list(all_subjects.keys()),
                        format_func=lambda k: f"{all_subjects[k]['display_name']} [{k}]",
                    )
                with tc_c2:
                    tc_topic = st.text_input("Topik / Bab:", placeholder="Contoh: Logaritma & Eksponen")
                with tc_c3:
                    tc_bloom = st.selectbox("Level Bloom:", ["C4", "C5", "C6"])

                tc_question = st.text_area("Pertanyaan Uji:", height=70)
                tc_gt = st.text_area("Jawaban Referensi (Ground Truth):", height=70)
                tc_keywords = st.text_input(
                    "Kata Kunci Ekspektasi (pisahkan dengan koma):",
                    placeholder="Contoh: eksponen, basis, logaritma, pangkat",
                )
                if st.form_submit_button("➕ Simpan ke Golden Testset", width="stretch"):
                    kw_list = [k.strip().lower() for k in tc_keywords.split(",") if k.strip()]
                    if not tc_question.strip() or not kw_list:
                        st.error("Pertanyaan uji dan minimal 1 kata kunci ekspektasi wajib diisi.")
                    else:
                        existing_for_sub = [x for x in all_test_cases if x.get("subject") == tc_sub]
                        new_id = f"{tc_sub}_{len(existing_for_sub) + 1:02d}"
                        dataset_mgr.add_custom_test_case(
                            {
                                "id": new_id,
                                "subject": tc_sub,
                                "topic": tc_topic.strip() or all_subjects[tc_sub]["display_name"],
                                "question": tc_question.strip(),
                                "ground_truth_answer": tc_gt.strip(),
                                "expected_keywords": kw_list,
                                "bloom_level": tc_bloom,
                            }
                        )
                        st.toast(f"Butir uji '{new_id}' berhasil ditambahkan ke Golden Testset!", icon="✅")
                        st.rerun()

        col_btn, col_scope = st.columns([1, 1])
        with col_scope:
            eval_scope = st.selectbox(
                "Cakupan Evaluasi:",
                ["Semua Mapel (Cross-Domain)"] + list(all_subjects.keys()),
                index=0,
                format_func=lambda x: x
                if x == "Semua Mapel (Cross-Domain)"
                else f"Mapel: {all_subjects[x]['display_name']}",
            )
        with col_btn:
            st.write("")
            st.write("")
            run_eval_btn = st.button("⚡ Jalankan Eksperimen Evaluasi Lengkap", type="primary", width="stretch")

        if run_eval_btn:
            target_sub = None if eval_scope == "Semua Mapel (Cross-Domain)" else eval_scope
            try:
                with st.spinner("Menjalankan 4 varian pipeline terhadap dataset uji..."):
                    evaluator = SystematicRAGEvaluator(retriever=retriever, generator=generator)
                    summary_df = evaluator.run_ablation_benchmark(subject=target_sub)
                    st.session_state.eval_results = summary_df
            except ValueError as ve:
                st.warning(f"⚠️ {ve} Buka panel 'Kelola Golden Testset Evaluasi' di atas untuk menambahkan soal uji mapel tersebut.")

        if st.session_state.eval_results is not None:
            res_df = st.session_state.eval_results
            st.success("Eksperimen selesai! Berikut matriks perbandingan performa:")

            st.dataframe(res_df, width="stretch")

            # Graphical Visualization
            st.subheader("📈 Visualisasi Metrik Kunci:")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.caption("Akurasi Retrieval: Hit Rate@4 & Context Precision")
                st.bar_chart(res_df.set_index("config")[["hit_rate@4", "context_precision"]])
            with chart_col2:
                st.caption("Kualitas Generasi: Faithfulness (Anti-Halusinasi) & Relevance")
                st.bar_chart(res_df.set_index("config")[["faithfulness", "answer_relevance"]])

            # Download Report
            csv_data = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Unduh Hasil Evaluasi (CSV untuk Laporan Tesis)",
                data=csv_data,
                file_name="rag_ablation_study_results.csv",
                mime="text/csv",
            )

# ----------------- TAB 3: DOKUMEN -----------------
with tab3:
    st.subheader(f"📂 Dokumen & Indeks Materi: {selected_subject_info['display_name']}")
    if is_admin:
        st.caption(f"Direktori Dokumen Fisik: `{RAW_PDF_DIR / selected_subject}`")

    sub_dir = RAW_PDF_DIR / selected_subject
    files = sorted(list(sub_dir.glob("*.pdf"))) if sub_dir.exists() else []

    if is_admin:
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🔄 Generate Dokumen Contoh Awal (Fase E)", width="stretch"):
                with st.spinner("Membuat PDF modul ajar Bahasa Inggris, Informatika, dan KKA..."):
                    from scripts.generate_sample_data import (
                        generate_bahasa_inggris_sample,
                        generate_informatika_sample,
                        generate_kka_sample,
                    )

                    generate_bahasa_inggris_sample()
                    generate_informatika_sample()
                    generate_kka_sample()
                    ingestion_pipeline.ingest_all_in_folder(embedder=llm_client)
                    st.success("Dokumen contoh berhasil dibuat dan langsung diindeks!")
                    st.rerun()

        with col_act2:
            if st.button(f"🗑️ Bersihkan Indeks Mapel '{selected_subject.upper()}'", width="stretch"):
                vector_store.clear_subject(selected_subject)
                bm25_store.clear_subject(selected_subject)
                st.warning(f"Indeks Vector & BM25 untuk {selected_subject} telah dikosongkan.")
                st.rerun()
    else:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if st.button(
                f"🔄 Indeks Ulang Seluruh PDF '{selected_subject_info['display_name']}'",
                width="stretch",
                disabled=len(files) == 0,
            ):
                with st.spinner("Mengindeks ulang dokumen mapel..."):
                    subject_grade = selected_subject_info.get("grade_level", "Kelas X")
                    for pdf_f in files:
                        ingestion_pipeline.ingest_single_pdf(
                            pdf_path=pdf_f,
                            subject=selected_subject,
                            embedder=llm_client,
                            grade_level=subject_grade,
                        )
                    st.toast("Indeks dokumen mapel berhasil diperbarui!", icon="✅")
                    st.rerun()
        with col_g2:
            if st.button(f"🗑️ Bersihkan Indeks Mapel '{selected_subject.upper()}'", width="stretch"):
                vector_store.clear_subject(selected_subject)
                bm25_store.clear_subject(selected_subject)
                st.warning(f"Indeks Vector & BM25 untuk {selected_subject} telah dikosongkan.")
                st.rerun()

    st.write(
        f"Daftar File PDF tersimpan pada mapel **{selected_subject_info['display_name']}** ({len(files)} file):"
    )
    if files:
        for f in files:
            fc1, fc2 = st.columns([5, 1])
            with fc1:
                st.text(f"• {f.name} ({round(f.stat().st_size / 1024, 1)} KB)")
            with fc2:
                if st.button("🗑️ Hapus", key=f"del_pdf_{selected_subject}_{f.name}", width="stretch"):
                    vector_store.delete_source_file(selected_subject, f.name)
                    bm25_store.delete_source_file(selected_subject, f.name)
                    try:
                        f.unlink()
                    except OSError:
                        pass
                    st.toast(f"Dokumen '{f.name}' beserta indeksnya berhasil dihapus.", icon="🗑️")
                    st.rerun()
    else:
        st.info("Belum ada file PDF pada mata pelajaran ini. Silakan unggah melalui menu '3. Dynamic Ingestion PDF' di sidebar kiri.")

    if is_admin:
        st.divider()
        st.subheader("📊 Status Penyimpanan Vektor (ChromaDB) & Leksikal (BM25)")
        st.caption("Ringkasan seluruh koleksi chunk yang tersimpan permanen di disk `data/chroma_db` dan `data/bm25_db`.")

        col_rows = []
        for col_info in vector_store.get_all_collections_summary():
            c_name = col_info["name"]
            dim_label = (
                "384-dim (FastEmbed Local)"
                if "384" in c_name
                else ("3072-dim (Gemini)" if c_name == "curriculum_rag_docs" else "Vector")
            )
            sub_str = ", ".join([f"{k.upper()}: {v}" for k, v in col_info.get("subjects", {}).items()])
            col_rows.append(
                {
                    "Nama Koleksi": c_name,
                    "Tipe / Dimensi": dim_label,
                    "Total Chunk": col_info["count"],
                    "Rincian Mapel": sub_str or "(kosong)",
                }
            )
        if col_rows:
            st.dataframe(pd.DataFrame(col_rows), width="stretch")

        if hasattr(bm25_store, "get_stats_by_subject"):
            bm25_all = bm25_store.get_stats_by_subject()
        else:
            bm25_all = {sub: len(chunks) for sub, chunks in getattr(bm25_store, "chunk_stores", {}).items()}
        if bm25_all:
            bm25_str = " | ".join([f"{k.upper()}: {v} chunks" for k, v in bm25_all.items()])
            st.info(f"📚 **Indeks Leksikal BM25 (Offline Terpadu):** {bm25_str}")

        st.divider()
        st.subheader("⚙️ Preset Endpoint LLM Custom (Tersimpan Permanen)")
        st.caption(
            "Daftar konfigurasi endpoint OpenAI-compatible yang tersimpan di `data/custom_providers.json`. Dapat digunakan langsung tanpa menyentuh file `.env`."
        )

        custom_presets_data = load_custom_presets()
        if custom_presets_data:
            rows = []
            for pname, pval in custom_presets_data.items():
                key_raw = pval.get("api_key", "")
                masked_key = (
                    f"{key_raw[:8]}...{key_raw[-4:]}"
                    if len(key_raw) > 12
                    else ("(kosong)" if not key_raw else "****")
                )
                rows.append(
                    {
                        "Nama Preset": pname,
                        "Base URL": pval.get("base_url", ""),
                        "Model Default": pval.get("model_name", ""),
                        "Embedding Provider": pval.get("embedding_provider", ""),
                        "API Key": masked_key,
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("Belum ada preset custom yang tersimpan.")

# ----------------- TAB 4: MANAJEMEN MAPEL & GURU (KHUSUS ADMIN) -----------------
if is_admin and tab4 is not None:
    with tab4:
        render_admin_management_tab(
            current_user=current_user,
            vector_store=vector_store,
            bm25_store=bm25_store,
            embedder=llm_client,
        )
