import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage.vector_store import ChromaVectorStore
from src.storage.bm25_store import BM25SubjectStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import PedagogicalGenerator
from src.evaluation.evaluator import SystematicRAGEvaluator
from src.llm_factory import get_llm_client
from src.config import ALLOWED_SUBJECTS

def main():
    parser = argparse.ArgumentParser(description="Jalankan Metodologi Evaluasi Sistematis & Ablation Study RAG Mapel")
    parser.add_argument("--subject", type=str, default=None, help="Mapel khusus (bahasa_inggris, informatika, kka) atau None untuk semua")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "openai", "grok", "custom", "mock"], help="Penyedia LLM")
    parser.add_argument("--model", type=str, default=None, help="Nama model LLM (e.g., gemini-2.0-flash, gpt-4o-mini)")
    args = parser.parse_args()

    print("=" * 80)
    print("   METODOLOGI EVALUASI SISTEMATIS RAG GENERATIF MAPEL (S2 LEVEL)")
    print("   Jenjang: Kelas X (Fase E) | Mapel: Bahasa Inggris, Informatika, KKA")
    print(f"   LLM Provider: {args.provider.upper()} | Model: {args.model or 'Default'}")
    print("=" * 80)

    # 1. Initialize LLM Client
    llm_client = get_llm_client(provider=args.provider, model_name=args.model)

    # 2. Initialize Stores & Retriever
    vector_store = ChromaVectorStore()
    bm25_store = BM25SubjectStore()
    retriever = HybridRetriever(vector_store=vector_store, bm25_store=bm25_store, embedder=llm_client)
    generator = PedagogicalGenerator(llm_client=llm_client)

    # 3. Initialize Evaluator
    evaluator = SystematicRAGEvaluator(retriever=retriever, generator=generator)

    print(f"\n[INFO] Menjalankan Ablation Study pada Golden Dataset...")
    if args.subject:
        print(f"[INFO] Filter khusus Mapel: {args.subject}")
    else:
        print("[INFO] Evaluasi mencakup seluruh Mapel (Cross-Domain Testing).")

    try:
        summary_df = evaluator.run_ablation_benchmark(subject=args.subject)
        print("\n" + "=" * 80)
        print("HASIL KOMPARATIF ABLATION STUDY (KUANTITATIF):")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)
        print("\n[SUKSES] Hasil evaluasi telah diekspor ke: data/evaluation/ablation_benchmark_results.csv")
    except Exception as e:
        print(f"[ERROR] Evaluasi gagal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
