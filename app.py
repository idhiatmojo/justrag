import os
import streamlit as st
import pandas as pd
from pathlib import Path

from src.config import ALLOWED_SUBJECTS, RAW_PDF_DIR
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
    load_custom_presets,
    save_custom_preset,
    delete_custom_preset,
    fetch_available_models,
    test_custom_endpoint,
)

st.set_page_config(
    page_title="RAG Generatif Mapel Fase E - S2 Project",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.title("🎓 Konfigurasi Sistem RAG")
st.sidebar.caption("Prototipe Penelitian S2: Generatif Bahan Ajar & Asesmen Terisolasi Domain")

# 1. Subject Selection
st.sidebar.subheader("1. Pilih Mata Pelajaran (Kelas X)")
subject_keys = list(ALLOWED_SUBJECTS.keys())
subject_labels = [ALLOWED_SUBJECTS[k]["display_name"] for k in subject_keys]

selected_idx = st.sidebar.selectbox(
    "Mata Pelajaran Aktif:",
    range(len(subject_keys)),
    format_func=lambda i: subject_labels[i]
)
selected_subject = subject_keys[selected_idx]
st.sidebar.info(ALLOWED_SUBJECTS[selected_subject]["description"])

# 2. LLM Provider & Model Selection
st.sidebar.subheader("2. Pilihan Model LLM")
provider = st.sidebar.selectbox(
    "LLM Provider:",
    ["gemini", "openai", "grok", "custom", "mock"],
    index=0,
    help="Pilih provider LLM: Gemini, OpenAI, Grok (xAI), Custom (OpenAI-compatible), atau simulasi Mock."
)

embedding_provider_input: str | None = None
embedding_api_key_input: str | None = None
base_url_input: str | None = None

if provider == "gemini":
    model_name = st.sidebar.selectbox(
        "Varian Model Gemini:",
        ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-flash-latest"],
        index=0
    )
    api_key_input = st.sidebar.text_input(
        "Google Gemini API Key:",
        value=os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Masukkan Gemini API Key Anda. Jika kosong, sistem otomatis beralih ke mode simulasi aman."
    )
elif provider == "openai":
    model_name = st.sidebar.selectbox(
        "Varian Model OpenAI:",
        ["gpt-4o-mini", "gpt-4o"],
        index=0
    )
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key:",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password"
    )
elif provider == "grok":
    model_name = st.sidebar.selectbox(
        "Varian Model Grok (xAI):",
        ["grok-3-mini", "grok-3", "grok-2"],
        index=0,
        help="grok-3-mini lebih hemat; grok-3 untuk respons terbaik."
    )
    api_key_input = st.sidebar.text_input(
        "xAI API Key:",
        value=os.getenv("XAI_API_KEY", "") or os.getenv("GROK_API_KEY", ""),
        type="password",
        help="Dapatkan di https://console.x.ai"
    )
    st.sidebar.caption("⚙️ Embedding Provider (Grok tidak punya model embedding sendiri):")
    embedding_provider_input = st.sidebar.selectbox(
        "Pilih Embedding Provider:",
        ["gemini", "openai", "local"],
        index=0,
        help="'gemini' direkomendasikan (gratis & berkualitas). 'local' tidak butuh API key.",
        key="grok_emb_provider"
    )
    if embedding_provider_input == "gemini":
        embedding_api_key_input = st.sidebar.text_input(
            "Gemini API Key (untuk embedding):",
            value=os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
            type="password",
            key="grok_emb_key"
        )
    elif embedding_provider_input == "openai":
        embedding_api_key_input = st.sidebar.text_input(
            "OpenAI API Key (untuk embedding):",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            key="grok_emb_key_oai"
        )
    else:
        st.sidebar.info("Mode embedding lokal aktif (hash-based, tanpa API). Akurasi semantic retrieval terbatas.")
elif provider == "custom":
    presets = load_custom_presets()
    preset_names = list(presets.keys())
    preset_options = preset_names + ["➕ Buat Preset Baru..."]

    if "custom_preset_choice" not in st.session_state or st.session_state.custom_preset_choice not in preset_options:
        st.session_state.custom_preset_choice = preset_names[0] if preset_names else "➕ Buat Preset Baru..."

    selected_preset = st.sidebar.selectbox(
        "Pilih Preset Endpoint Tersimpan:",
        preset_options,
        index=preset_options.index(st.session_state.custom_preset_choice) if st.session_state.custom_preset_choice in preset_options else 0,
        help="Preset disimpan permanen di data/custom_providers.json (tidak butuh konfigurasi .env)",
        key="sb_preset_selector"
    )
    st.session_state.custom_preset_choice = selected_preset
    is_new = (selected_preset == "➕ Buat Preset Baru...")

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

    if is_new:
        preset_name_input = st.sidebar.text_input(
            "Nama Preset Baru:",
            value="Custom Gateway",
            help="Nama pengenal untuk disimpan ke daftar preset Anda"
        )
    else:
        preset_name_input = selected_preset

    base_url_input = st.sidebar.text_input(
        "API Endpoint (Base URL):",
        value=def_url,
        key=f"url_input_{selected_preset}",
        help="Endpoint OpenAI-compatible (contoh: https://api.juanrouter.com/v1, https://api.deepseek.com/v1, http://localhost:11434/v1)"
    )

    api_key_input = st.sidebar.text_input(
        "API Key:",
        value=def_key,
        type="password",
        key=f"key_input_{selected_preset}",
        help="API Key provider custom Anda"
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
                    st.sidebar.warning("Gagal menarik daftar model otomatis (atau endpoint tidak mendukung /v1/models). Silakan ketik nama model secara manual.")

    available_preset_models = st.session_state.get(f"fetched_models_{selected_preset}") or curr_preset.get("available_models", [])
    if available_preset_models:
        model_choice_list = list(dict.fromkeys(available_preset_models + ["Ketik manual..."]))
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
            help="Contoh: qwen3.8-max, deepseek-v4-pro, llama3.2"
        )

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
        help="'gemini' direkomendasikan (gratis & berkualitas). 'local' tidak butuh API key tambahan."
    )

    if embedding_provider_input == "gemini":
        embedding_api_key_input = st.sidebar.text_input(
            "Gemini API Key (untuk embedding):",
            value=os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
            type="password",
            key=f"custom_emb_key_gemini_{selected_preset}"
        )
    elif embedding_provider_input == "openai":
        embedding_api_key_input = st.sidebar.text_input(
            "OpenAI API Key (untuk embedding):",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            key=f"custom_emb_key_oai_{selected_preset}"
        )
    else:
        st.sidebar.info("Mode embedding lokal aktif (hash-based, tanpa API). Akurasi semantic retrieval terbatas.")

    # Tombol Simpan & Hapus Preset
    col_save, col_del = st.sidebar.columns([1, 1])
    with col_save:
        if st.button("💾 Simpan", width="stretch", help="Simpan preset ini ke file data/custom_providers.json"):
            target_save_name = preset_name_input.strip()
            if not target_save_name:
                st.sidebar.error("Nama preset tidak boleh kosong!")
            else:
                save_custom_preset(target_save_name, {
                    "base_url": base_url_input,
                    "api_key": api_key_input,
                    "model_name": model_name,
                    "embedding_provider": embedding_provider_input,
                    "available_models": available_preset_models
                })
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

# 3. Dynamic PDF Ingestion Section
st.sidebar.subheader("3. Dynamic Ingestion PDF")
uploaded_files = st.sidebar.file_uploader(
    f"Upload PDF Baru untuk {ALLOWED_SUBJECTS[selected_subject]['display_name']}:",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("📥 Proses & Indeks PDF ke Vector DB", width="stretch"):
        with st.sidebar.status("Memproses dokumen...", expanded=True) as status:
            target_dir = RAW_PDF_DIR / selected_subject
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for uf in uploaded_files:
                save_path = target_dir / uf.name
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
                
                st.write(f"Indexing `{uf.name}`...")
                res = ingestion_pipeline.ingest_single_pdf(
                    pdf_path=save_path,
                    subject=selected_subject,
                    embedder=llm_client
                )
                st.write(f"✅ Selesai: {res['pages']} hal, {res['chunks']} chunks.")
            status.update(label="Semua PDF berhasil diindeks!", state="complete")

# Stats per subject
stats = vector_store.get_stats_by_subject()
st.sidebar.caption(f"📊 Status Korpus Tersimpan:")
for sk, sinfo in ALLOWED_SUBJECTS.items():
    st.sidebar.text(f"• {sk.upper()}: {stats.get(sk, 0)} chunks")

# ----------------- MAIN UI TABS -----------------
st.title("📚 Pipeline RAG Generatif Mapel Fase E (Kelas X)")
st.caption("Implementasi End-to-End dengan Isolasi Metadata Domain & Evaluasi Saintifik (Tugas Riset Pascasarjana / S2)")

tab1, tab2, tab3 = st.tabs([
    "🚀 Generatif Bahan Ajar & Asesmen",
    "📊 Metodologi Evaluasi & Ablation Study",
    "📂 Manajemen Dokumen & Analisis Korpus"
])

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
                ("qa_chat", "💬 Tanya Jawab Terpandu (Q&A)")
            ],
            format_func=lambda x: x[1]
        )[0]

    with col_task:
        retrieval_strategy = st.selectbox(
            "Strategi Retrieval:",
            [
                ("hybrid", "Hybrid Search (Dense + BM25 dengan RRF)"),
                ("dense_only", "Dense Vector Search Only"),
                ("bm25_only", "BM25 Sparse Lexical Only")
            ],
            format_func=lambda x: x[1]
        )[0]

    with st.expander("⚙️ Pengaturan Parameter Pipeline (Ablation Toggles)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            enable_filter = st.checkbox("Isolasi Filter Mapel", value=True, help="Mencegah kontaminasi dokumen mapel lain.")
        with c2:
            enable_rerank = st.checkbox("Aktifkan Re-ranking (Cross-Scoring)", value=True, help="Menyaring Top-K chunk sebelum masuk konteks.")
        with c3:
            temperature = st.slider("Temperature Generasi:", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    # Prompt suggestions per subject
    sample_queries = {
        "bahasa_inggris": "Buat modul ajar perbandingan Simple Past dan Present Perfect beserta asesmen kontekstual",
        "informatika": "Buat 3 butir soal HOTS level C4-C5 tentang empat pilar berpikir komputasional dan topologi star",
        "kka": "Rangkum konsep Supervised vs Unsupervised Learning serta etika bias algoritma untuk siswa SMK"
    }

    user_query = st.text_area(
        "Instruksi / Topik yang Ingin Dihasilkan:",
        value=sample_queries.get(selected_subject, ""),
        height=100
    )

    if st.button("✨ Hasilkan Konten dengan RAG", type="primary", width="stretch"):
        if not user_query.strip():
            st.warning("Silakan masukkan topik atau instruksi.")
        else:
            with st.spinner("Menjalankan pipeline retrieval dan inferensi LLM..."):
                # 1. Retrieve
                candidate_k = 6 if enable_rerank else 4
                raw_chunks = retriever.retrieve(
                    query=user_query,
                    subject=selected_subject,
                    mode=retrieval_strategy,
                    use_subject_filter=enable_filter,
                    top_k=candidate_k
                )

                # 2. Rerank
                if enable_rerank and raw_chunks:
                    final_chunks = reranker.rerank(query=user_query, candidates=raw_chunks, top_k=4)
                else:
                    final_chunks = raw_chunks[:4]

                # 3. Generate
                output = generator.generate(
                    query=user_query,
                    subject=selected_subject,
                    retrieved_chunks=final_chunks,
                    task_type=task_mode,
                    temperature=temperature
                )

                # Display Output
                st.subheader("📋 Hasil Generasi Pedagogis:")
                st.markdown(output["response"])

                # Display Citations / Grounding
                st.divider()
                st.subheader("🔍 Bukti Grounding & Konteks Dokumen Terambil:")
                if final_chunks:
                    for i, chunk in enumerate(final_chunks, 1):
                        meta = chunk.get("metadata", {})
                        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0.0)))
                        with st.expander(f"Kutipan #{i} | Sumber: `{meta.get('source_file')}` (Hal. {meta.get('page_number')}) | Skor: {score:.4f}"):
                            st.write(chunk.get("text"))
                else:
                    st.info("Tidak ada konteks yang terambil.")

# ----------------- TAB 2: EVALUATION -----------------
with tab2:
    st.subheader("🔬 Metodologi Evaluasi Saintifik & Studi Komparatif (Ablation Benchmark)")
    st.markdown("""
    Sesuai standar riset S2, pengujian dilakukan dengan membandingkan 4 konfigurasi pipeline pada **Golden Testset** terstandarisasi:
    1. **Baseline (Naive RAG)**: Fixed Dense Retrieval tanpa filter mapel (menguji terjadinya domain cross-contamination).
    2. **Model A (Metadata Filter)**: Dense Retrieval dengan filter mapel aktif.
    3. **Model B (Hybrid Search)**: Dense + BM25 Lexical dengan fusi peringkat RRF.
    4. **Model C (Proposed Full Pipeline)**: Hybrid + Filter + Re-ranking + Contextual Grounded Prompting.
    """)

    col_btn, col_scope = st.columns([1, 1])
    with col_scope:
        eval_scope = st.selectbox(
            "Cakupan Evaluasi:",
            ["Semua Mapel (Cross-Domain)", selected_subject],
            index=0
        )
    with col_btn:
        st.write("")
        st.write("")
        run_eval_btn = st.button("⚡ Jalankan Eksperimen Evaluasi Lengkap", type="primary", width="stretch")

    if run_eval_btn:
        target_sub = None if "Semua" in eval_scope else selected_subject
        with st.spinner("Menjalankan 4 varian pipeline terhadap dataset uji..."):
            evaluator = SystematicRAGEvaluator(retriever=retriever, generator=generator)
            summary_df = evaluator.run_ablation_benchmark(subject=target_sub)
            st.session_state.eval_results = summary_df

    if st.session_state.eval_results is not None:
        res_df = st.session_state.eval_results
        st.success("Eksperimen selesai! Berikut matriks perbandingan performa:")
        
        st.dataframe(res_df)

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
        csv_data = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Unduh Hasil Evaluasi (CSV untuk Laporan Tesis)",
            data=csv_data,
            file_name="rag_ablation_study_results.csv",
            mime="text/csv"
        )

# ----------------- TAB 3: DOKUMEN -----------------
with tab3:
    st.subheader("📂 Manajemen Korpus & Eksplorasi Data")
    st.write(f"Direktori Dokumen Fisik: `{RAW_PDF_DIR}`")
    
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("🔄 Generate Dokumen Contoh Awal (Fase E)", width="stretch"):
            with st.spinner("Membuat PDF modul ajar Bahasa Inggris, Informatika, dan KKA..."):
                from scripts.generate_sample_data import generate_bahasa_inggris_sample, generate_informatika_sample, generate_kka_sample
                generate_bahasa_inggris_sample()
                generate_informatika_sample()
                generate_kka_sample()
                # Ingest immediately
                ingestion_pipeline.ingest_all_in_folder(embedder=llm_client)
                st.success("Dokumen contoh berhasil dibuat dan langsung diindeks!")
                st.rerun()

    with col_act2:
        if st.button(f"🗑️ Bersihkan Indeks Mapel '{selected_subject.upper()}'", width="stretch"):
            vector_store.clear_subject(selected_subject)
            st.warning(f"Indeks untuk {selected_subject} telah dikosongkan.")
            st.rerun()

    # Show files on disk
    sub_dir = RAW_PDF_DIR / selected_subject
    files = list(sub_dir.glob("*.pdf")) if sub_dir.exists() else []
    st.write(f"Daftar File PDF tersimpan pada mapel **{selected_subject}** ({len(files)} file):")
    for f in files:
        st.text(f"• {f.name} ({round(f.stat().st_size / 1024, 1)} KB)")

    st.divider()
    st.subheader("⚙️ Preset Endpoint LLM Custom (Tersimpan Permanen)")
    st.caption("Daftar konfigurasi endpoint OpenAI-compatible yang tersimpan di `data/custom_providers.json`. Dapat digunakan langsung tanpa menyentuh file `.env`.")

    custom_presets_data = load_custom_presets()
    if custom_presets_data:
        rows = []
        for pname, pval in custom_presets_data.items():
            key_raw = pval.get("api_key", "")
            masked_key = f"{key_raw[:8]}...{key_raw[-4:]}" if len(key_raw) > 12 else ("(kosong)" if not key_raw else "****")
            rows.append({
                "Nama Preset": pname,
                "Base URL": pval.get("base_url", ""),
                "Model Default": pval.get("model_name", ""),
                "Embedding Provider": pval.get("embedding_provider", ""),
                "API Key": masked_key
            })
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("Belum ada preset custom yang tersimpan.")
