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
    ["gemini", "openai", "mock"],
    index=0,
    help="Default: Google Gemini. Bisa beralih ke OpenAI atau simulasi Mock."
)

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
else:
    model_name = "mock-pedagogy-v1"
    api_key_input = "mock-key"
    st.sidebar.warning("Mode simulasi lokal aktif. Hasil respons bersifat representatif offline.")

# Instantiate LLM Client
llm_client = get_llm_client(provider=provider, model_name=model_name, api_key=api_key_input if api_key_input else None)

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
    if st.sidebar.button("📥 Proses & Indeks PDF ke Vector DB", use_container_width=True):
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

    if st.button("✨ Hasilkan Konten dengan RAG", type="primary", use_container_width=True):
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
        run_eval_btn = st.button("⚡ Jalankan Eksperimen Evaluasi Lengkap", type="primary", use_container_width=True)

    if run_eval_btn:
        target_sub = None if "Semua" in eval_scope else selected_subject
        with st.spinner("Menjalankan 4 varian pipeline terhadap dataset uji..."):
            evaluator = SystematicRAGEvaluator(retriever=retriever, generator=generator)
            summary_df = evaluator.run_ablation_benchmark(subject=target_sub)
            st.session_state.eval_results = summary_df

    if st.session_state.eval_results is not None:
        res_df = st.session_state.eval_results
        st.success("Eksperimen selesai! Berikut matriks perbandingan performa:")
        
        st.dataframe(res_df, use_container_width=True)

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
        if st.button("🔄 Generate Dokumen Contoh Awal (Fase E)", use_container_width=True):
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
        if st.button(f"🗑️ Bersihkan Indeks Mapel '{selected_subject.upper()}'", use_container_width=True):
            vector_store.clear_subject(selected_subject)
            st.warning(f"Indeks untuk {selected_subject} telah dikosongkan.")
            st.rerun()

    # Show files on disk
    sub_dir = RAW_PDF_DIR / selected_subject
    files = list(sub_dir.glob("*.pdf")) if sub_dir.exists() else []
    st.write(f"Daftar File PDF tersimpan pada mapel **{selected_subject}** ({len(files)} file):")
    for f in files:
        st.text(f"• {f.name} ({round(f.stat().st_size / 1024, 1)} KB)")
