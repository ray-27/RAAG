from collections import defaultdict

import numpy as np
from tqdm import tqdm

from config import K_VALUES
from metrics import map_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


def evaluate(retriever, queries: dict[str, str], qrels: dict[str, dict[str, int]], k_values=None):
    k_values = tuple(k_values or K_VALUES)
    max_k = max(k_values)
    ranked = retriever.retrieve_many(queries, top_k=max_k)

    buckets = defaultdict(list)
    for qid in tqdm(queries, desc="metrics", disable=len(queries) < 32):
        relevant = qrels.get(qid, {})
        retrieved = [doc_id for doc_id, _ in ranked.get(qid, [])]
        for k in k_values:
            buckets[f"recall@{k}"].append(recall_at_k(relevant, retrieved, k))
            buckets[f"precision@{k}"].append(precision_at_k(relevant, retrieved, k))
            buckets[f"ndcg@{k}"].append(ndcg_at_k(relevant, retrieved, k))
            buckets[f"mrr@{k}"].append(mrr_at_k(relevant, retrieved, k))
            buckets[f"map@{k}"].append(map_at_k(relevant, retrieved, k))

    return {name: float(np.mean(values)) if values else 0.0 for name, values in buckets.items()}


GREEN = "\033[32m"
RESET = "\033[0m"


def format_results(all_results: dict[str, dict[str, float]], k_values) -> str:
    names = list(all_results.keys())
    col_w = max(14, max(len(name) for name in names) + 2)
    metric_order = ["recall", "precision", "ndcg", "mrr", "map"]
    header = f"{'metric':<16}" + "".join(f"{name:>{col_w}}" for name in names)
    lines = [header, "-" * len(header)]
    for metric in metric_order:
        for k in k_values:
            key = f"{metric}@{k}"
            values = [all_results[name].get(key, 0.0) for name in names]
            best = max(values) if values else None
            row = f"{key:<16}"
            for value in values:
                cell = f"{value:>{col_w}.4f}"
                if best is not None and value == best:
                    cell = f"{GREEN}{cell}{RESET}"
                row += cell
            lines.append(row)
    return "\n".join(lines)
