import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from src.config import RAW_PDF_DIR

def build_pdf(filepath: Path, title: str, pages_content: list):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(filepath), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        spaceAfter=8,
        spaceBefore=12
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=8
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 12))

    for page_idx, sections in enumerate(pages_content):
        if page_idx > 0:
            story.append(PageBreak())
        for heading, body in sections:
            story.append(Paragraph(heading, heading_style))
            story.append(Paragraph(body, body_style))
            story.append(Spacer(1, 8))

    doc.build(story)
    print(f"[OK] Generated: {filepath}")

def generate_bahasa_inggris_sample():
    path = RAW_PDF_DIR / "bahasa_inggris" / "modul_ajar_bahasa_inggris_kelas_x.pdf"
    title = "Kementerian Pendidikan - Bahan Ajar Bahasa Inggris Fase E Kelas X"
    pages = [
        [
            ("Unit 1: The Art of Narrative Texts", 
             "A narrative text is an imaginative or factual story that entertains and informs readers. The generic structure of a narrative text comprises: Orientation, Complication, Resolution, and optional Re-orientation. Orientation introduces the characters, setting, and background of the story. Complication introduces conflicts or problems faced by the protagonists. Resolution shows how the characters solve the crisis."),
            ("Language Features of Narrative Texts",
             "Narrative texts predominantly employ the Past Tense, action verbs, temporal connectives (once upon a time, suddenly, then), and direct speech. Students must recognize past forms of irregular and regular verbs to express events sequentially.")
        ],
        [
            ("Unit 2: Grammar Focus - Simple Past vs Present Perfect",
             "Understanding the temporal distinction between Simple Past and Present Perfect is crucial for Grade X students. Simple Past is utilized for events that started and finished at a definite time in the past (e.g., 'The team completed the coding project yesterday'). In contrast, Present Perfect describes actions that occurred at an indefinite time, or past events that have present relevance (e.g., 'We have studied three programming paradigms so far'). Key time signals for Simple Past include 'yesterday', 'last week', and 'ago', while Present Perfect utilizes 'already', 'yet', 'since', and 'for'."),
            ("Unit 3: Functional Texts - School Announcements",
             "School announcements convey critical operational information. An effective announcement text must specify the purpose, target audience, day and date, exact time, venue or location, and the contact person. Omission of venue or schedule causes informational ambiguity.")
        ]
    ]
    build_pdf(path, title, pages)

def generate_informatika_sample():
    path = RAW_PDF_DIR / "informatika" / "buku_informatika_fase_e_kelas_x.pdf"
    title = "Buku Siswa Informatika Kelas X - Kurikulum Merdeka"
    pages = [
        [
            ("Bab 1: Berpikir Komputasional (Computational Thinking)",
             "Berpikir Komputasional (BK) merupakan landasan problem solving dalam Informatika Fase E Kelas X. Terdapat empat pilar utama: 1) Dekomposisi: kemampuan memecah persoalan kompleks menjadi sub-persoalan yang lebih terkelola. 2) Pengenalan Pola: mengidentifikasi keteraturan atau kesamaan ciri dalam kumpulan data atau masalah. 3) Abstraksi: menyaring prinsip inti dan mengabaikan rincian yang tidak esensial. 4) Algoritma: merumuskan langkah-langkah sistematis dan terurut untuk menghasilkan solusi otomatis."),
            ("Penerapan Algoritma dalam Pemrograman",
             "Algoritma diekspresikan melalui pseudocode dan diagram alir (flowchart). Struktur kontrol mencakup percabangan (if-else) untuk pengambilan keputusan logis, serta perulangan (for dan while). Perulangan 'for' digunakan untuk counted loops (jumlah iterasi diketahui), sedangkan 'while' digunakan untuk conditional loops (berulang selama kondisi benar).")
        ],
        [
            ("Bab 2: Jaringan Komputer dan Internet",
             "Jaringan komputer memungkinkan pertukaran data antar-perangkat melalui protokol standar TCP/IP. Topologi jaringan menentukan konfigurasi fisik perangkat. Topologi Star menghubungkan setiap node ke switch sentral; keunggulannya adalah isolasi kerusakan kabel (jika satu kabel putus, koneksi node lain tidak terganggu), namun biayanya lebih mahal dan rentan single point of failure pada switch sentral. Topologi Bus menggunakan kabel tunggal backbone; instalasinya hemat biaya, namun tabrakan paket data (collision) tinggi dan kerusakan kabel utama melumpuhkan seluruh jaringan."),
            ("Keamanan Jaringan & Enkripsi Data",
             "Transmisi data di internet memerlukan enkripsi seperti HTTPS/TLS untuk melindungi integritas dan kerahasiaan informasi pengguna dari serangan man-in-the-middle (MITM).")
        ]
    ]
    build_pdf(path, title, pages)

def generate_kka_sample():
    path = RAW_PDF_DIR / "kka" / "modul_kka_koding_ai_kelas_x.pdf"
    title = "Modul KKA: Koding dan Kecerdasan Artifisial Kelas X Fase E"
    pages = [
        [
            ("Modul 1: Pengantar Logika Pemrograman AI dengan Python",
             "Koding dan Kecerdasan Artifisial (KKA) membekali siswa Kelas X dengan literasi komputasi modern. Python menjadi bahasa utama karena sintaksis ekspresif dan ekosistem pustaka data science. Struktur data fundamental mencakup List, Tuple, dan Dictionary. Logika percabangan dan perulangan digunakan untuk memproses masukan sensor atau data masukan pengguna secara berulang."),
            ("Modul 2: Paradigma Machine Learning - Supervised vs Unsupervised",
             "Machine Learning dibagi menjadi beberapa paradigma utama. Supervised Learning melatih algoritma menggunakan kumpulan data berlabel (labeled dataset) di mana setiap fitur input dipasangkan dengan target output yang valid; contoh penerapannya meliputi klasifikasi spam email dan deteksi citra. Sebaliknya, Unsupervised Learning menganalisis kumpulan data tanpa label untuk mengungkap pola laten atau pengelompokan alami, seperti clustering segmentasi profil pengguna.")
        ],
        [
            ("Modul 3: Etika Algoritma & Bias Kecerdasan Artifisial",
             "Sistem AI dilatih di atas data buatan manusia yang berpotensi memuat bias historis, bias representasi, atau bias seleksi. Apabila dataset pelatihan diskriminatif, model AI akan mereproduksi dan memperkuat ketidakadilan sosial tersebut. Oleh karena itu, prinsip AI beretika mewajibkan transparansi, audit keadilan (fairness metrics), akuntabilitas, serta pelibatan manusia sebagai pengawas (human-in-the-loop)."),
            ("Modul 4: Prompt Engineering & Large Language Models",
             "Teknik rekayasa prompt (Prompt Engineering) esensial untuk memaksimalkan kinerja model generatif. Metode Few-Shot Prompting menyertakan contoh masukan-keluaran demonstratif sebelum mengajukan tugas baru. Pendekatan Chain-of-Thought (CoT) menginstruksikan model bernalar langkah-demi-langkah, secara signifikan mereduksi halusinasi faktual dan memperbaiki akurasi penalaran matematis/logika.")
        ]
    ]
    build_pdf(path, title, pages)

if __name__ == "__main__":
    generate_bahasa_inggris_sample()
    generate_informatika_sample()
    generate_kka_sample()
    print("Semua dokumen contoh kurikulum Kelas X berhasil dibuat!")
