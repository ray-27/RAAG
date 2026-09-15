from collections import defaultdict

from config import (
    BATCH_SIZE,
    BM25_B,
    BM25_K1,
    CHUNK,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEVICE,
    HYBRID_ALPHA,
    MODEL_NAME,
)
from retrievers.base import Retriever
from retrievers.bm25 import BM25Retriever
from retrievers.cosine import CosineRetriever


def _minmax(ranked: list[tuple[str, float]]) -> dict[str, float]:
    if not ranked:
        return {}
    scores = [score for _, score in ranked]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return {doc_id: 1.0 for doc_id, _ in ranked}
    return {doc_id: (score - lo) / (hi - lo) for doc_id, score in ranked}


class HybridRetriever(Retriever):
    def __init__(
        self,
        alpha: float = HYBRID_ALPHA,
        model_name: str = MODEL_NAME,
        batch_size: int = BATCH_SIZE,
        k1: float = BM25_K1,
        b: float = BM25_B,
        device: str | None = DEVICE,
        chunk: bool = CHUNK,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.alpha = alpha
        self.bm25 = BM25Retriever(k1=k1, b=b)
        self.cosine = CosineRetriever(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            chunk=chunk,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        self.bm25.index(corpus)
        self.cosine.index(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return self.retrieve_many({"q": query}, top_k=top_k)["q"]

    def retrieve_many(
        self, queries: dict[str, str], top_k: int = 10
    ) -> dict[str, list[tuple[str, float]]]:
        pool = max(top_k * 4, top_k)
        bm25_hits = self.bm25.retrieve_many(queries, top_k=pool)
        cosine_hits = self.cosine.retrieve_many(queries, top_k=pool)
        results = {}
        for qid in queries:
            fused: dict[str, float] = defaultdict(float)
            for doc_id, score in _minmax(bm25_hits.get(qid, [])).items():
                fused[doc_id] += (1.0 - self.alpha) * score
            for doc_id, score in _minmax(cosine_hits.get(qid, [])).items():
                fused[doc_id] += self.alpha * score
            ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
            results[qid] = ranked[:top_k]
        return results
