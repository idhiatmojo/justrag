import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import EVALUATION_DIR

DEFAULT_GOLDEN_TESTSET: List[Dict[str, Any]] = [
    # --- BAHASA INGGRIS ---
    {
        "id": "bi_01",
        "subject": "bahasa_inggris",
        "topic": "Narrative Text Structure",
        "question": "What are the core generic structures of a Narrative Text, and what is the function of the complication?",
        "ground_truth_answer": "The core structures of a narrative text are Orientation, Complication, Resolution, and optional Re-orientation (Coda). The complication introduces problems or crises that the main characters must face.",
        "expected_keywords": ["orientation", "complication", "resolution", "characters", "problem"],
        "bloom_level": "C2"
    },
    {
        "id": "bi_02",
        "subject": "bahasa_inggris",
        "topic": "Grammar: Simple Past vs Present Perfect",
        "question": "Explain when to use Simple Past Tense versus Present Perfect Tense with concrete sentence examples for high school students.",
        "ground_truth_answer": "Simple Past is used for finished actions in a specific past time (e.g., 'I visited Bali last year'). Present Perfect is used for actions connected to the present without a specific past time or ongoing experience (e.g., 'I have lived here for two years').",
        "expected_keywords": ["simple past", "present perfect", "specific time", "experience", "have"],
        "bloom_level": "C4"
    },
    {
        "id": "bi_03",
        "subject": "bahasa_inggris",
        "topic": "Announcement Text Analysis",
        "question": "What essential information must be included in a school announcement text to ensure student comprehension?",
        "ground_truth_answer": "An effective school announcement must include the title/purpose, target audience, day/date/time, location/venue, contact person or organizer, and call to action.",
        "expected_keywords": ["announcement", "date", "time", "location", "contact person", "audience"],
        "bloom_level": "C3"
    },

    # --- INFORMATIKA ---
    {
        "id": "inf_01",
        "subject": "informatika",
        "topic": "4 Pilar Berpikir Komputasional",
        "question": "Sebutkan dan jelaskan empat fondasi utama dalam Berpikir Komputasional (Computational Thinking) Kelas X!",
        "ground_truth_answer": "Empat fondasi berpikir komputasional adalah: 1) Dekomposisi (memecah masalah kompleks menjadi bagian kecil), 2) Pengenalan Pola (mencari kesamaan karakteristik), 3) Abstraksi (fokus pada informasi penting dan mengabaikan detail tak relevan), dan 4) Algoritma (menyusun langkah-langkah terurut untuk menyelesaikan masalah).",
        "expected_keywords": ["dekomposisi", "pengenalan pola", "abstraksi", "algoritma"],
        "bloom_level": "C2"
    },
    {
        "id": "inf_02",
        "subject": "informatika",
        "topic": "Jaringan Komputer: Topologi Jaringan",
        "question": "Bandingkan kelebihan dan kelemahan topologi Star vs topologi Bus dalam jaringan komputer sekolah!",
        "ground_truth_answer": "Topologi Star menggunakan switch/hub pusat; kelebihannya jika satu kabel putus yang lain tetap berfungsi, kelemahannya biaya kabel lebih tinggi dan bergantung pada switch. Topologi Bus menggunakan kabel tunggal backbone; kelebihannya hemat kabel dan instalasi sederhana, tetapi jika backbone rusak seluruh jaringan mati dan deteksi error sulit.",
        "expected_keywords": ["topologi star", "topologi bus", "switch", "kabel backbone", "kelebihan", "kelemahan"],
        "bloom_level": "C4"
    },
    {
        "id": "inf_03",
        "subject": "informatika",
        "topic": "Analisis Algoritma Percabangan & Perulangan",
        "question": "Bagaimana cara kerja struktur kontrol perulangan 'for' dan 'while' serta kapan masing-masing sebaiknya digunakan?",
        "ground_truth_answer": "Perulangan 'for' (counted loop) digunakan ketika jumlah iterasi sudah diketahui secara pasti sebelumnya. Perulangan 'while' (uncounted/conditional loop) digunakan ketika pengulangan bergantung pada kondisi boolean tertentu yang bernilai benar, dan akan berhenti saat kondisi menjadi salah.",
        "expected_keywords": ["perulangan", "for", "while", "kondisi", "iterasi"],
        "bloom_level": "C3"
    },

    # --- KKA (Koding & Kecerdasan Artifisial) ---
    {
        "id": "kka_01",
        "subject": "kka",
        "topic": "Dasar Logika AI: Supervised vs Unsupervised Learning",
        "question": "Jelaskan perbedaan mendasar antara Supervised Learning dan Unsupervised Learning dalam Kecerdasan Artifisial disertai contoh kasus nyata!",
        "ground_truth_answer": "Supervised Learning melatih model menggunakan data berlabel (ada input dan target output yang benar), contohnya klasifikasi email spam atau deteksi penyakit. Unsupervised Learning melatih model pada data tanpa label untuk menemukan pola tersembunyi, contohnya segmentasi perilaku pelanggan (clustering).",
        "expected_keywords": ["supervised", "unsupervised", "data berlabel", "clustering", "klasifikasi"],
        "bloom_level": "C4"
    },
    {
        "id": "kka_02",
        "subject": "kka",
        "topic": "Etika & Bias Algoritma AI",
        "question": "Mengapa bias dalam dataset pelatihan dapat menyebabkan masalah etika pada sistem Kecerdasan Artifisial, dan bagaimana pencegahannya?",
        "ground_truth_answer": "Dataset yang bias mencerminkan diskriminasi historis manusia atau ketidakseimbangan representasi, sehingga keputusan AI menjadi tidak adil atau diskriminatif. Pencegahannya dilakukan dengan kurasi data representatif, audit keadilan algoritma (fairness metric), dan pengawasan manusia (human-in-the-loop).",
        "expected_keywords": ["bias", "etika", "dataset", "diskriminasi", "keadilan", "fairness"],
        "bloom_level": "C5"
    },
    {
        "id": "kka_03",
        "subject": "kka",
        "topic": "Prompt Engineering & Koding Python AI",
        "question": "Jelaskan konsep Few-Shot Prompting dan peran Chain-of-Thought (CoT) dalam memaksimalkan output model bahasa generatif!",
        "ground_truth_answer": "Few-Shot Prompting memberikan beberapa contoh pasangan input-output sebelum meminta jawaban model, membimbing format dan gaya respon. Chain-of-Thought (CoT) menginstruksikan model untuk menuliskan langkah penalaran secara bertahap sebelum kesimpulan akhir, meningkatkan akurasi penalaran kompleks.",
        "expected_keywords": ["few-shot", "chain-of-thought", "cot", "penalaran", "contoh"],
        "bloom_level": "C4"
    }
]

class EvaluationDatasetManager:
    """Manages storage, augmentation, and retrieval of Golden Evaluation Datasets."""
    
    def __init__(self, file_path: Optional[str | Path] = None):
        self.file_path = Path(file_path) if file_path else (EVALUATION_DIR / "golden_testset.json")
        self._ensure_dataset()

    def _ensure_dataset(self):
        if not self.file_path.exists():
            self.save_dataset(DEFAULT_GOLDEN_TESTSET)

    def load_dataset(self, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if subject and subject.strip():
            sub = subject.lower().strip()
            return [item for item in data if item.get("subject", "").lower() == sub]
        return data

    def save_dataset(self, testset: List[Dict[str, Any]]):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(testset, f, indent=2, ensure_ascii=False)

    def add_custom_test_case(self, item: Dict[str, Any]):
        current = self.load_dataset()
        current.append(item)
        self.save_dataset(current)
