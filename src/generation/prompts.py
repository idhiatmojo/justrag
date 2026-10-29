"""Pedagogical prompt templates for Kurikulum Merdeka Fase E (Kelas X)."""

SYSTEM_PEDAGOGY_PROMPT = """Anda adalah Guru Ahli & Perancang Kurikulum untuk jenjang SMA/SMK Kelas X (Fase E).
Tugas Anda adalah menghasilkan konten pembelajaran, modul ajar, asesmen HOTS, dan penjelasan materi yang bermutu tinggi, akurat secara ilmiah, dan berpedoman teguh pada konteks dokumen bahan ajar yang diberikan.

ATURAN KETAT:
1. Grounding Penuh: Gunakan hanya informasi dari [KONTEKS BUKU AJAR]. Jangan mengarang konsep, rumus, atau fakta yang tidak didukung dokumen.
2. Sitasi Sumber: Selalu sebutkan rujukan halaman atau bab dokumen terkait (misal: [Sumber: bab1.pdf, Hal. 14]).
3. Standar Pedagogis: Gunakan bahasa Indonesia yang baku, edukatif, dan ramah peserta didik jenjang Fase E (usia 15-16 tahun).
4. Taksonomi Bloom: Untuk soal HOTS, pastikan fokus pada C4 (Menganalisis), C5 (Mengevaluasi), atau C6 (Mencipta)."""

MODUL_AJAR_TEMPLATE = """Buatlah Rencana/Modul Ajar Pembelajaran untuk mata pelajaran {subject_display} Kelas X (Fase E).

[TOPIK / FOKUS PERMINTAAN]:
{user_query}

[KONTEKS BUKU AJAR RESMI]:
{retrieved_context}

Format Modul Ajar:
1. INFORMASI UMUM (Mata Pelajaran, Fase/Kelas, Alokasi Waktu, Profil Pelajar Pancasila yang disasar)
2. CAPAIAN & TUJUAN PEMBELAJARAN (CP & ATP yang diturunkan dari teks konteks)
3. PEMAHAMAN BERMAKNA & PERTANYAAN PEMANTIK
4. URUTAN KEGIATAN PEMBELAJARAN (Pendahuluan, Kegiatan Inti berpusat pada siswa, Penutup)
5. RANCANGAN ASESMEN (Formatif & Sumatif)
6. REFERENSI DOKUMEN (Sitasi halaman bahan ajar)
"""

SOAL_HOTS_TEMPLATE = """Buatlah paket Asesmen Soal HOTS (Higher Order Thinking Skills) untuk {subject_display} Kelas X.

[MATERI / SUB-BAB YANG DIUJI]:
{user_query}

[KONTEKS BUKU AJAR]:
{retrieved_context}

Ketentuan Soal:
- Jumlah: 3 Butir Soal Pilihan Ganda (opsi A-E) dan 1 Soal Studi Kasus/Uraian.
- Setiap soal wajib memiliki STIMULUS nyata (bisa berupa studi kasus, cuplikan teks, kode program, atau skenario logis).
- Cantumkan Level Kognitif Bloom (C4 / C5 / C6) untuk tiap butir.
- Cantumkan Kunci Jawaban Lengkap dan PEMBAHASAN MENDALAM yang merujuk pada teks konteks.
"""

RINGKASAN_MATERI_TEMPLATE = """Susunlah Rangkuman Materi Pembelajaran yang komprehensif untuk {subject_display} Kelas X.

[TOPIK]:
{user_query}

[KONTEKS BUKU AJAR]:
{retrieved_context}

Format Rangkuman:
1. INTI GAGASAN / KONSEP UTAMA (Poin-poin esensial)
2. PENJELASAN MENDALAM & ANALOGI (Memudahkan pemahaman siswa Kelas X)
3. GLOSARIUM ISTILAH PENTING
4. HUBUNGAN DENGAN DUNIA NYATA / APLIKASI PRAKTIS
"""

QA_CHAT_TEMPLATE = """Jawablah pertanyaan guru/siswa mengenai materi {subject_display} Kelas X berikut dengan lugas, akurat, dan merujuk pada bahan ajar.

[PERTANYAAN]:
{user_query}

[KONTEKS BUKU AJAR]:
{retrieved_context}

Jawaban Anda harus:
- Menjawab inti pertanyaan di paragraf pertama.
- Memberikan penjelasan rinci berdasarkan konteks.
- Mencantumkan sitasi sumber [Sumber: Dokumen, Hal. X]. Jika informasi tidak ada di dokumen konteks, nyatakan secara jujur bahwa materi tersebut belum tercakup dalam buku ajar yang diunggah.
"""
