import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import tqdm

import config
from chunking import chunk_document
from data_loader import BEIR_DATASETS, load_beir
from model_loader import resolve_device
from training.encoder import FrozenEncoder
from training.eval_metrics import MAX_K, METRIC_NAMES, format_metrics, score_ranking
from training.projection import ProjectionHead
from training.text import passage_text


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained projection head on DBpedia")
    parser.add_argument(
        "--model",
        required=True,
        help="path to projection_head.pt or a run directory containing it",
    )
    parser.add_argument("--title", default=None, help="graph title; default is the encoder model name")
    parser.add_argument("--dataset", default="dbpedia-entity", choices=list(BEIR_DATASETS))
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument(
        "--max-queries",
        type=int,
        default=config.TEST_MAX_QUERIES,
        help="queries to sample; omit for every query in the split",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=config.TEST_MAX_DOCS,
        help="corpus cap: all golds plus random distractors (default 100000)",
    )
    parser.add_argument("--full", action="store_true", help="use every query and the full 4.63M corpus")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--chunk",
        action=argparse.BooleanOptionalAction,
        default=config.CHUNK,
    )
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument(
        "--concat-title",
        action=argparse.BooleanOptionalAction,
        default=config.CONCAT_TITLE,
    )
    return parser.parse_args()


def resolve_model_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "projection_head.pt"
    if not path.exists():
        raise FileNotFoundError(f"projection head not found: {path}")
    return path


def load_checkpoint(path: Path, device: str):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    head = ProjectionHead(ckpt["in_dim"], ckpt["hidden_dim"], ckpt["out_dim"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head, ckpt


def model_title(ckpt: dict, override: str | None) -> str:
    if override:
        return override
    name = (ckpt.get("meta") or {}).get("model_name") or config.MODEL_NAME
    return Path(str(name)).name


def corpus_items(corpus: dict[str, dict[str, str]], chunk: bool, chunk_size: int, overlap: int, concat_title: bool):
    texts = []
    chunk_doc_idx = []
    doc_ids = list(corpus.keys())
    for i, doc_id in enumerate(doc_ids):
        doc = corpus[doc_id]
        if chunk:
            pieces = chunk_document(doc.get("title", ""), doc.get("text", ""), chunk_size, overlap)
        else:
            pieces = [passage_text(doc, concat_title)]
        for piece in pieces:
            texts.append(piece)
            chunk_doc_idx.append(i)
    return doc_ids, texts, chunk_doc_idx


@torch.no_grad()
def encode_projected(encoder: FrozenEncoder, head: ProjectionHead, texts: list[str], batch_size: int, device: str):
    head.eval()
    out = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode"):
        raw = encoder.encode(texts[start : start + batch_size], batch_size=batch_size, show_progress_bar=False)
        x = torch.tensor(np.asarray(raw, dtype=np.float32), device=device)
        out.append(head(x))
    return torch.cat(out, dim=0)


@torch.no_grad()
def retrieve_many(q_z: torch.Tensor, p_z: torch.Tensor, doc_ids: list[str], chunk_doc_idx: list[int], qids: list[str], top_k: int):
    idx = torch.tensor(chunk_doc_idx, device=p_z.device, dtype=torch.long)
    n_docs = len(doc_ids)
    k = min(top_k, n_docs)
    pooled = idx.numel() != n_docs or not torch.equal(idx, torch.arange(n_docs, device=p_z.device))
    results = {}
    for i, qid in enumerate(tqdm(qids, desc="rank", disable=len(qids) < 16)):
        chunk_scores = p_z @ q_z[i]
        if pooled:
            doc_scores = torch.full((n_docs,), -1e9, device=p_z.device, dtype=chunk_scores.dtype)
            doc_scores.scatter_reduce_(0, idx, chunk_scores, reduce="amax", include_self=True)
        else:
            doc_scores = chunk_scores
        vals, top = torch.topk(doc_scores, k)
        results[qid] = [(doc_ids[int(j)], float(v)) for v, j in zip(vals.tolist(), top.tolist())]
    return results


def evaluate_rankings(queries: dict[str, str], qrels: dict[str, dict[str, int]], ranked: dict):
    buckets = {name: [] for name in METRIC_NAMES}
    for qid in queries:
        relevant = qrels.get(qid, {})
        retrieved = [doc_id for doc_id, _ in ranked.get(qid, [])]
        row = score_ranking(relevant, retrieved)
        for name, value in row.items():
            buckets[name].append(value)
    return {name: float(np.mean(vals)) if vals else 0.0 for name, vals in buckets.items()}


def save_csv(path: Path, row: dict) -> None:
    fields = ["dataset", "split", "model", "device", *METRIC_NAMES]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def save_plot(path: Path, title: str, metrics: dict[str, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = METRIC_NAMES
    values = [metrics[name] for name in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, values, color="#3b82f6")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for label, value in zip(names, values):
        ax.text(label, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if device == "cuda":
        print(f"using cuda: {torch.cuda.get_device_name(0)}")
    else:
        print(f"using {device}")

    model_path = resolve_model_path(args.model)
    head, ckpt = load_checkpoint(model_path, device)
    title = model_title(ckpt, args.title)
    encoder_name = (ckpt.get("meta") or {}).get("model_name") or config.MODEL_NAME
    encoder = FrozenEncoder(encoder_name, device)

    spec = BEIR_DATASETS[args.dataset]
    data_dir = args.data_dir or spec["data_dir"]
    max_queries = None if args.full else args.max_queries
    max_docs = None if args.full else args.max_docs
    if args.full:
        print(f"loading {args.dataset} split={args.split} FULL corpus on {device}")
    else:
        q_desc = "all split queries" if not max_queries else f"{max_queries} queries"
        print(
            f"loading {args.dataset} split={args.split} sample: {q_desc}, "
            f"up to {max_docs} docs (golds + distractors) on {device}"
        )
    corpus, queries, qrels = load_beir(
        dataset=args.dataset,
        split=args.split,
        max_queries=max_queries,
        max_docs=max_docs,
        seed=args.seed,
        data_dir=data_dir,
    )
    print(f"corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")

    doc_ids, texts, chunk_doc_idx = corpus_items(
        corpus, args.chunk, args.chunk_size, args.chunk_overlap, args.concat_title
    )
    print(f"indexing {len(texts)} vectors for {len(doc_ids)} docs (chunk={args.chunk})")
    p_z = encode_projected(encoder, head, texts, args.batch_size, device)
    qids = list(queries.keys())
    q_z = encode_projected(encoder, head, [queries[qid] for qid in qids], args.batch_size, device)
    ranked = retrieve_many(q_z, p_z, doc_ids, chunk_doc_idx, qids, MAX_K)
    metrics = evaluate_rankings(queries, qrels, ranked)
    print("dbpedia test " + format_metrics(metrics))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else model_path.parent / f"dbpedia_test_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    plot_path = out_dir / "metrics.png"
    info = {
        "model": str(model_path),
        "title": title,
        "encoder": encoder_name,
        "dataset": args.dataset,
        "split": args.split,
        "device": device,
        "corpus": len(corpus),
        "queries": len(queries),
        "vectors": len(texts),
        "full": args.full,
        "max_queries": max_queries,
        "max_docs": max_docs,
        "chunk": args.chunk,
        "metrics": metrics,
    }
    save_csv(
        csv_path,
        {
            "dataset": args.dataset,
            "split": args.split,
            "model": title,
            "device": device,
            **{name: f"{metrics[name]:.6f}" for name in METRIC_NAMES},
        },
    )
    save_plot(plot_path, title, metrics)
    (out_dir / "results.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
