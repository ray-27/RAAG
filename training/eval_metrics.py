from metrics import map_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k

METRIC_FNS = {
    "ndcg@10": (ndcg_at_k, 10),
    "mrr@10": (mrr_at_k, 10),
    "recall@10": (recall_at_k, 10),
    "recall@100": (recall_at_k, 100),
    "map@10": (map_at_k, 10),
    "precision@10": (precision_at_k, 10),
}
METRIC_NAMES = list(METRIC_FNS)
MAX_K = max(k for _, k in METRIC_FNS.values())


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{name}={metrics[name]:.4f}" for name in METRIC_NAMES if name in metrics)


def score_ranking(relevant: dict, retrieved: list) -> dict[str, float]:
    return {name: float(fn(relevant, retrieved, k)) for name, (fn, k) in METRIC_FNS.items()}
