# Dynamic Multi-Subject Pedagogical RAG Pipeline with Systematic Evaluation Framework
### Implementasi Riset Pascasarjana (S2) - Bahan Ajar & Asesmen Fase E (Kelas X)

Proyek ini mengimplementasikan arsitektur **Retrieval-Augmented Generation (RAG) End-to-End** tingkat pascasarjana untuk pembuatan bahan ajar, modul pembelajaran (Kurikulum Merdeka), dan paket soal HOTS (Higher Order Thinking Skills berdasar Taksonomi Bloom) di atas dokumen kurikulum yang ditambahkan secara dinamis.

Pipeline ini mengatasi 3 tantangan kritis:
1. **Isolasi Domain & Pencegahan Kontaminasi Silang (*Cross-Domain Contamination*)**: Memisahkan konteks materi antar mata pelajaran (Bahasa Inggris, Informatika, dan KKA - Koding & Kecerdasan Artifisial) menggunakan *metadata routing & pre-filtering*.
2. **Ingesti Dokumen Dinamis (*Dynamic Incremental Indexing*)**: Mendukung penambahan PDF buku ajar baru tanpa perlu melakukan re-indexing ulang terhadap seluruh korpus.
3. **Metodologi Evaluasi Saintifik & Studi Komparatif (*Ablation Study*)**: Menguji 4 varian konfigurasi pipeline pada *Golden Evaluation Dataset* untuk mengukur dampak isolasi metadata, pencarian hibrida (Dense + BM25 dengan RRF), dan re-ranking secara kuantitatif.

---

## 🏛️ Arsitektur Sistem

```mermaid
flowchart TD
    subgraph Ingestion ["1. Dynamic Ingestion & Indexing"]
        A[Upload PDF Dinamis] --> B[PDF Parser & Normalizer]
        B --> C[Contextual Chunker<br/>Prefix Metadata Injection]
        C --> D[(ChromaDB Vector Store<br/>Metadata-Indexed)]
        C --> E[(BM25 Sparse Store<br/>Partitioned per Subject)]
    end

    subgraph Query ["2. Retrieval & Generation Pipeline"]
        F[Prompt Guru / Siswa] --> G{Pilihan Mapel & Mode}
        G --> H[Hybrid Retriever Engine]
        D -.->|Dense Cosine Search| H
        E -.->|BM25 Sparse Search| H
        H --> I[Reciprocal Rank Fusion - RRF]
        I --> J[Lexical-Semantic Cross Reranker]
        J --> K[Pedagogical Prompt Engine<br/>Modul Ajar / Soal HOTS / Ringkasan]
        K --> L[LLM: Google Gemini (Default) / OpenAI / Mock]
        L --> M[Output Terstruktur + Sitasi Sumber]
    end

    subgraph Evaluation ["3. Systematic Evaluation Suite (S2 Metodologi)"]
        N[Golden Evaluation Dataset<br/>Triplet: Q, Context, GT] --> O[Ablation Runner Engine]
        O --> P[Hit Rate@4 & MRR]
        O --> Q[Context Precision & Recall]
        O --> R[Faithfulness / Groundedness]
        O --> S[Answer Relevance & Bloom Alignment]
        P & Q & R & S --> T[Matriks Komparasi 4 Model & Ekspor CSV]
    end
```

---

## 📚 Cakupan Mata Pelajaran (Fase E - Kelas X)

1. **Bahasa Inggris**:
   - Genre teks: Narrative, Descriptive, Recount, Announcement.
   - Tata Bahasa: Simple Past vs Present Perfect Tense.
   - Reading comprehension level analitis.
2. **Informatika**:
   - 4 Pilar Berpikir Komputasional (Dekomposisi, Pengenalan Pola, Abstraksi, Algoritma).
   - Jaringan Komputer & Internet (Topologi Jaringan, Model TCP/IP, Enkripsi).
   - Algoritma Pemrograman (Flowchart, Pseudocode, Percabangan & Perulangan).
3. **KKA (Koding & Kecerdasan Artifisial)**:
   - Logika Pemrograman AI dengan Python (Struktur data dasar, manipulasi list & dict).
   - Paradigma Machine Learning: Supervised vs Unsupervised Learning.
   - Etika AI: Mitigasi bias algoritma, keadilan representasi, human-in-the-loop.
   - Rekayasa Prompt: Few-shot prompting dan Chain-of-Thought (CoT).

---

## 🤖 Pilihan Model LLM

Pipeline dirancang secara modular (*Provider Pattern* via `src/llm_factory.py`):
- **Google Gemini (Default)**:
  - `gemini-2.0-flash` (Direkomendasikan untuk kecepatan & penalaran tinggi)
  - `gemini-1.5-flash`
  - `gemini-1.5-pro` (Untuk penalaran kurikulum mendalam)
- **OpenAI**:
  - `gpt-4o-mini`
  - `gpt-4o`
- **Offline Mock / Test Simulation**:
  - Memungkinkan pengujian fungsionalitas dan benchmarking tanpa ketergantungan API eksternal.

---

## 🔬 Metodologi Evaluasi Sistematis (Ablation Study)

Untuk memenuhi standar akademik pascasarjana, evaluasi memisahkan kinerja **Retrieval** dari **Generation** menggunakan 4 konfigurasi:

| Konfigurasi | Deskripsi | Tujuan Eksperimen |
| :--- | :--- | :--- |
| **1. Baseline (Naive RAG)** | Dense Only, Tanpa Filter Mapel | Membuktikan terjadinya *domain cross-contamination* |
| **2. Model A** | Dense Only + Metadata Filter Mapel | Mengukur dampak isolasi domain |
| **3. Model B** | Hybrid (Dense + BM25 RRF) + Filter Mapel | Mengukur kontribusi pencarian leksikal untuk istilah teknis |
| **4. Model C (Proposed)** | Hybrid + Filter + Reranker + Pedagogical CoT | Mengukur performa end-to-end terbaik |

### Metrik Kuantitatif Terukur:
- **Retrieval**: `Hit Rate@4`, `MRR (Mean Reciprocal Rank)`, `Context Precision`, `Context Recall`.
- **Generation**: `Faithfulness (Groundedness / Anti-Halusinasi)`, `Answer Relevance`, `Bloom's Taxonomy Compliance`.
- **Efisiensi**: `Latency (detik)`.

---

## 🚀 Petunjuk Penggunaan & Eksekusi

### 1. Persiapan Environment
Gunakan Python 3.10+:
```bash
# Salin konfigurasi environment
cp .env.example .env

# Masukkan API Key Anda di .env atau via UI Streamlit:
# GEMINI_API_KEY=AIzaSy...
```

### 2. Generate Data Dokumen Awal (Otomatis)
Untuk menghasilkan PDF modul kurikulum Fase E Kelas X secara instan:
```bash
/Users/bgxerjoe/miniforge3/envs/ml/bin/python scripts/generate_sample_data.py
```

### 3. Menjalankan Prototipe Web Interaktif (Streamlit)
```bash
/Users/bgxerjoe/miniforge3/envs/ml/bin/streamlit run app.py
```
Akses di browser: `http://localhost:8501`. Fitur UI mencakup:
- Upload dinamis PDF per mapel.
- Pilihan mode generatif: Modul Ajar, Soal HOTS, Rangkuman, Tanya Jawab.
- Panel evaluasi ablasi interaktif dengan grafik perbandingan metrik.

### 4. Menjalankan Evaluasi Sistematis via CLI
```bash
# Menjalankan evaluasi lengkap seluruh mapel
/Users/bgxerjoe/miniforge3/envs/ml/bin/python scripts/run_evaluation.py --provider gemini --model gemini-2.0-flash

# Menjalankan evaluasi khusus mapel Informatika
/Users/bgxerjoe/miniforge3/envs/ml/bin/python scripts/run_evaluation.py --subject informatika --provider gemini
```
Hasil benchmark akan tersimpan di `data/evaluation/ablation_benchmark_results.csv`.
