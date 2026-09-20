import argparse
import json
from collections import OrderedDict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import config
from training.encoder import FrozenEncoder
from training.text import passage_text


def parse_args():
    parser = argparse.ArgumentParser(description="Build frozen embedding cache from hard-negative JSONL")
    parser.add_argument("--input", default=str(config.HARD_NEG_JSONL))
    parser.add_argument("--cache-dir", default=str(config.CACHE_DIR))
    parser.add_argument("--val-ratio", type=float, default=config.VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=config.TEST_RATIO)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument(
        "--concat-title",
        action=argparse.BooleanOptionalAction,
        default=config.CONCAT_TITLE,
    )
    return parser.parse_args()


def load_rows(path: Path, max_rows: int | None):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def assign_splits(n: int, val_ratio: float, test_ratio: float, seed: int) -> list[str]:
    n_val = int(round(n * val_ratio)) if val_ratio > 0 else 0
    n_test = int(round(n * test_ratio)) if test_ratio > 0 else 0
    if n >= 3:
        if val_ratio > 0:
            n_val = max(1, n_val)
        if test_ratio > 0:
            n_test = max(1, n_test)
    if n_val + n_test >= n:
        n_test = min(n_test, max(0, n - 2))
        n_val = min(n_val, max(0, n - 1 - n_test))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    splits = ["train"] * n
    for i in order[:n_test]:
        splits[int(i)] = "test"
    for i in order[n_test : n_test + n_val]:
        splits[int(i)] = "val"
    return splits


def main():
    args = parse_args()
    src = Path(args.input)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(src, args.max_rows)
    splits = assign_splits(len(rows), args.val_ratio, args.test_ratio, args.seed)

    passages: OrderedDict[str, str] = OrderedDict()
    query_ids = []
    query_texts = []
    records = []
    same_row_bugs = 0

    for i, row in enumerate(rows):
        qid = str(row["query_id"])
        query_ids.append(qid)
        query_texts.append(row["query"])

        pos_ids = []
        pos_set = set()
        for item in row.get("positives") or []:
            pid = str(item["_id"])
            pos_set.add(pid)
            pos_ids.append(pid)
            passages.setdefault(pid, passage_text(item, args.concat_title))

        neg_ids = []
        neg_ranks = []
        seen_neg = set()
        for item in row.get("hard_negatives") or []:
            nid = str(item["_id"])
            if nid in pos_set:
                same_row_bugs += 1
                continue
            if nid in seen_neg:
                continue
            seen_neg.add(nid)
            passages.setdefault(nid, passage_text(item, args.concat_title))
            neg_ids.append(nid)
            neg_ranks.append(int(item.get("bm25_rank") or 0))

        records.append({
            "query_id": qid,
            "split": splits[i],
            "pos_ids": pos_ids,
            "neg_ids": neg_ids,
            "neg_ranks": neg_ranks,
        })

    n_train = sum(s == "train" for s in splits)
    n_val = sum(s == "val" for s in splits)
    n_test = sum(s == "test" for s in splits)
    print(
        f"rows={len(rows)} train={n_train} val={n_val} test={n_test} "
        f"unique_passages={len(passages)} dropped_same_row_pos_neg={same_row_bugs}"
    )

    encoder = FrozenEncoder(config.MODEL_NAME, args.device)
    dim = encoder.dim
    query_emb = np.asarray(encoder.encode(query_texts, batch_size=args.batch_size), dtype=np.float32)
    passage_ids = list(passages.keys())
    passage_emb = np.asarray(
        encoder.encode([passages[pid] for pid in passage_ids], batch_size=args.batch_size),
        dtype=np.float32,
    )

    pid_to_idx = {pid: i for i, pid in enumerate(passage_ids)}
    qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}
    index_records = []
    for rec in records:
        pos_indices = [pid_to_idx[pid] for pid in rec["pos_ids"] if pid in pid_to_idx]
        neg_indices = [pid_to_idx[nid] for nid in rec["neg_ids"] if nid in pid_to_idx]
        if not pos_indices:
            continue
        index_records.append({
            "query_index": qid_to_idx[rec["query_id"]],
            "query_id": rec["query_id"],
            "split": rec["split"],
            "pos_indices": pos_indices,
            "neg_indices": neg_indices,
            "neg_ranks": rec["neg_ranks"][: len(neg_indices)],
        })

    np.save(cache_dir / "query_emb.npy", query_emb)
    np.save(cache_dir / "passage_emb.npy", passage_emb)
    (cache_dir / "query_ids.txt").write_text("\n".join(query_ids) + "\n", encoding="utf-8")
    (cache_dir / "passage_ids.txt").write_text("\n".join(passage_ids) + "\n", encoding="utf-8")
    with (cache_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for rec in index_records:
            f.write(json.dumps(rec) + "\n")
    meta = {
        "source": str(src),
        "model_name": config.MODEL_NAME,
        "concat_title": args.concat_title,
        "dim": dim,
        "n_queries": len(query_ids),
        "n_passages": len(passage_ids),
        "n_train": sum(1 for r in index_records if r["split"] == "train"),
        "n_val": sum(1 for r in index_records if r["split"] == "val"),
        "n_test": sum(1 for r in index_records if r["split"] == "test"),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote cache to {cache_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
