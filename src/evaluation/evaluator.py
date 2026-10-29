import time
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import LexicalSemanticReranker
from src.generation.generator import PedagogicalGenerator
from src.evaluation.synthetic_data import EvaluationDatasetManager
from src.evaluation.metrics import RAGEvaluationMetrics
from src.llm_factory import BaseLLMClient
from src.config import EVALUATION_DIR

class SystematicRAGEvaluator:
    """
    Executes an academic ablation study comparing 4 pipeline configurations:
    1. Baseline Naive RAG (Dense Only, No Mapel Filter)
    2. Naive Filtered (Dense Only, With Mapel Filter)
    3. Hybrid RAG (BM25 + Dense RRF, With Mapel Filter)
    4. Proposed Advanced Pipeline (Hybrid + Mapel Filter + Reranker + Pedagogical Grounding)
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: PedagogicalGenerator,
        dataset_manager: Optional[EvaluationDatasetManager] = None
    ):
        self.retriever = retriever
        self.generator = generator
        self.reranker = LexicalSemanticReranker()
        self.dataset_manager = dataset_manager or EvaluationDatasetManager()

    def run_single_experiment(
        self,
        test_item: Dict[str, Any],
        config_name: str,
        retrieval_mode: str,
        use_subject_filter: bool,
        use_reranker: bool
    ) -> Dict[str, Any]:
        query = test_item["question"]
        subject = test_item["subject"]
        exp_kw = test_item.get("expected_keywords", [])
        bloom = test_item.get("bloom_level", "C4")

        start_time = time.perf_counter()

        # Step 1: Retrieval
        candidate_k = 6 if use_reranker else 4
        retrieved = self.retriever.retrieve(
            query=query,
            subject=subject,
            mode=retrieval_mode,
            use_subject_filter=use_subject_filter,
            top_k=candidate_k
        )

        # Step 2: Reranking (if enabled)
        if use_reranker and retrieved:
            retrieved = self.reranker.rerank(query=query, candidates=retrieved, top_k=4)

        # Step 3: Generation
        gen_result = self.generator.generate(
            query=query,
            subject=subject,
            retrieved_chunks=retrieved,
            task_type="qa_chat",
            temperature=0.2
        )
        latency = time.perf_counter() - start_time
        answer = gen_result["response"]

        # Step 4: Metric Computations
        hit_rate = RAGEvaluationMetrics.hit_rate_at_k(retrieved, exp_kw, top_k=4)
        mrr = RAGEvaluationMetrics.mean_reciprocal_rank(retrieved, exp_kw)
        ctx_precision = RAGEvaluationMetrics.context_precision(retrieved, exp_kw)
        ctx_recall = RAGEvaluationMetrics.context_recall(retrieved, exp_kw)
        faithfulness = RAGEvaluationMetrics.faithfulness_score(answer, retrieved)
        ans_relevance = RAGEvaluationMetrics.answer_relevance(query, answer)
        bloom_score = RAGEvaluationMetrics.bloom_taxonomy_compliance(answer, bloom)

        return {
            "config": config_name,
            "test_id": test_item["id"],
            "subject": subject,
            "latency_sec": round(latency, 3),
            "hit_rate@4": hit_rate,
            "mrr": round(mrr, 4),
            "context_precision": round(ctx_precision, 4),
            "context_recall": round(ctx_recall, 4),
            "faithfulness": faithfulness,
            "answer_relevance": ans_relevance,
            "bloom_compliance": bloom_score
        }

    def run_ablation_benchmark(self, subject: Optional[str] = None) -> pd.DataFrame:
        """Runs the complete ablation matrix over the golden evaluation dataset."""
        testset = self.dataset_manager.load_dataset(subject=subject)
        if not testset:
            raise ValueError("Dataset evaluasi kosong. Tambahkan data uji terlebih dahulu.")

        configs = [
            {
                "name": "1. Baseline (Naive Dense, No Filter)",
                "mode": "dense_only",
                "filter": False,
                "reranker": False
            },
            {
                "name": "2. Naive + Mapel Filter",
                "mode": "dense_only",
                "filter": True,
                "reranker": False
            },
            {
                "name": "3. Hybrid (Dense+BM25) + Filter",
                "mode": "hybrid",
                "filter": True,
                "reranker": False
            },
            {
                "name": "4. Proposed (Hybrid+Filter+Reranker)",
                "mode": "hybrid",
                "filter": True,
                "reranker": True
            }
        ]

        raw_records = []
        for cfg in configs:
            for item in testset:
                record = self.run_single_experiment(
                    test_item=item,
                    config_name=cfg["name"],
                    retrieval_mode=cfg["mode"],
                    use_subject_filter=cfg["filter"],
                    use_reranker=cfg["reranker"]
                )
                raw_records.append(record)

        df_raw = pd.DataFrame(raw_records)
        
        # Summary Aggregated Table
        summary_df = df_raw.groupby("config").agg({
            "latency_sec": "mean",
            "hit_rate@4": "mean",
            "mrr": "mean",
            "context_precision": "mean",
            "context_recall": "mean",
            "faithfulness": "mean",
            "answer_relevance": "mean",
            "bloom_compliance": "mean"
        }).reset_index()

        summary_df = summary_df.round(4)
        
        # Save results to disk
        out_csv = EVALUATION_DIR / "ablation_benchmark_results.csv"
        summary_df.to_csv(out_csv, index=False)
        
        return summary_df
