import numpy as np


def recall_at_k(relevant: dict[str, int], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def precision_at_k(relevant: dict[str, int], retrieved: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def ndcg_at_k(relevant: dict[str, int], retrieved: list[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevant.get(doc_id, 0)
        if rel:
            dcg += rel / np.log2(i + 2)
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal) if rel)
    return dcg / idcg if idcg else 0.0


def mrr_at_k(relevant: dict[str, int], retrieved: list[str], k: int) -> float:
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def map_at_k(relevant: dict[str, int], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(relevant), k)
