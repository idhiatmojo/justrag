import re
from typing import List, Dict, Any

class RAGEvaluationMetrics:
    """
    Academic quantitative evaluation metric suite for RAG pipelines.
    Measures retrieval efficacy, generation groundedness, and pedagogical fidelity.
    """

    @staticmethod
    def _extract_tokens(text: str) -> set:
        clean = re.sub(r'[^a-zA-Z0-9_\-\s]', ' ', text.lower())
        stopwords = {
            "dan", "atau", "yang", "di", "ke", "dari", "ini", "itu", "untuk",
            "pada", "adalah", "the", "and", "or", "in", "on", "at", "to", "for",
            "with", "is", "are", "sebagai", "dengan", "dapat", "akan", "dalam"
        }
        return {w for w in clean.split() if len(w) > 2 and w not in stopwords}

    @classmethod
    def hit_rate_at_k(cls, retrieved_chunks: List[Dict[str, Any]], expected_keywords: List[str], top_k: int = 4) -> float:
        """
        Hit Rate@K: Returns 1.0 if any chunk in top_k contains a critical threshold of expected keywords.
        """
        if not retrieved_chunks or not expected_keywords:
            return 0.0

        exp_set = {k.lower().strip() for k in expected_keywords}
        candidates = retrieved_chunks[:top_k]

        for chunk in candidates:
            chunk_tokens = cls._extract_tokens(chunk.get("text", ""))
            matches = sum(1 for kw in exp_set if any(kw in t or t in kw for t in chunk_tokens))
            if matches / len(exp_set) >= 0.4:
                return 1.0
        return 0.0

    @classmethod
    def mean_reciprocal_rank(cls, retrieved_chunks: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
        """
        MRR: 1 / rank of the first relevant retrieved chunk.
        """
        if not retrieved_chunks or not expected_keywords:
            return 0.0

        exp_set = {k.lower().strip() for k in expected_keywords}

        for rank, chunk in enumerate(retrieved_chunks, 1):
            chunk_tokens = cls._extract_tokens(chunk.get("text", ""))
            matches = sum(1 for kw in exp_set if any(kw in t or t in kw for t in chunk_tokens))
            if matches / len(exp_set) >= 0.4:
                return 1.0 / rank
        return 0.0

    @classmethod
    def context_precision(cls, retrieved_chunks: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
        """
        Context Precision: Ratio of relevant chunks among all retrieved chunks.
        """
        if not retrieved_chunks or not expected_keywords:
            return 0.0

        exp_set = {k.lower().strip() for k in expected_keywords}
        relevant_count = 0

        for chunk in retrieved_chunks:
            chunk_tokens = cls._extract_tokens(chunk.get("text", ""))
            matches = sum(1 for kw in exp_set if any(kw in t or t in kw for t in chunk_tokens))
            if matches / len(exp_set) >= 0.3:
                relevant_count += 1

        return relevant_count / len(retrieved_chunks)

    @classmethod
    def context_recall(cls, retrieved_chunks: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
        """
        Context Recall: Proportion of expected ground-truth concepts captured across all retrieved chunks.
        """
        if not retrieved_chunks or not expected_keywords:
            return 0.0

        exp_set = {k.lower().strip() for k in expected_keywords}
        combined_tokens = set()
        for chunk in retrieved_chunks:
            combined_tokens.update(cls._extract_tokens(chunk.get("text", "")))

        found = sum(1 for kw in exp_set if any(kw in t or t in kw for t in combined_tokens))
        return found / len(exp_set)

    @classmethod
    def faithfulness_score(cls, generated_answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Faithfulness (Groundedness): Measures the ratio of factual statements/tokens
        in the generated answer that are supported by the retrieved context.
        """
        if not generated_answer or not retrieved_chunks:
            return 0.0

        # Extract tokens from entire retrieved context
        context_corpus = " ".join([c.get("text", "") for c in retrieved_chunks])
        context_tokens = cls._extract_tokens(context_corpus)

        answer_sentences = [s.strip() for s in generated_answer.split('.') if len(s.strip()) > 15]
        if not answer_sentences:
            return 1.0

        supported_sentences = 0
        for sent in answer_sentences:
            sent_tokens = cls._extract_tokens(sent)
            if not sent_tokens:
                continue
            supported_tokens = sum(1 for t in sent_tokens if t in context_tokens)
            overlap_ratio = supported_tokens / len(sent_tokens)
            # A sentence is considered grounded if at least 40% of its content words are in context
            if overlap_ratio >= 0.4:
                supported_sentences += 1

        return round(supported_sentences / len(answer_sentences), 4)

    @classmethod
    def answer_relevance(cls, question: str, generated_answer: str) -> float:
        """
        Answer Relevance: Lexical & semantic overlap between user query and generated response.
        """
        q_tokens = cls._extract_tokens(question)
        if not q_tokens or not generated_answer:
            return 0.0

        ans_tokens = cls._extract_tokens(generated_answer)
        overlap = len(q_tokens.intersection(ans_tokens))
        score = min(1.0, overlap / max(1, len(q_tokens)))
        return round(score, 4)

    @classmethod
    def bloom_taxonomy_compliance(cls, generated_answer: str, target_level: str) -> float:
        """
        Evaluates cognitive taxonomy alignment (C1-C6 indicators).
        """
        level = target_level.upper().strip()
        bloom_keywords = {
            "C1": ["mengingat", "sebutkan", "apa itu", "definisi", "list", "identify", "state"],
            "C2": ["menjelaskan", "memahami", "uraikan", "explain", "describe", "summarize", "fondasi"],
            "C3": ["menerapkan", "gunakan", "buatlah", "apply", "execute", "implement", "cara kerja"],
            "C4": ["menganalisis", "bandingkan", "perbedaan", "kelebihan", "kelemahan", "analyze", "compare", "contrast"],
            "C5": ["mengevaluasi", "menilai", "justifikasi", "kritisi", "evaluate", "assess", "judge", "etika", "bias"],
            "C6": ["merancang", "menciptakan", "kembangkan", "rancanglah", "create", "design", "synthesize"]
        }
        
        indicators = bloom_keywords.get(level, bloom_keywords["C4"])
        ans_lower = generated_answer.lower()
        matches = sum(1 for ind in indicators if ind in ans_lower)
        return min(1.0, 0.5 + (0.5 * (matches / max(1, len(indicators)))) if matches > 0 else 0.5)
