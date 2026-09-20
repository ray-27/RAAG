import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedRetrievalDataset(Dataset):
    def __init__(self, records: list[dict], query_emb: np.ndarray, passage_emb: np.ndarray):
        self.records = records
        self.query_emb = query_emb
        self.passage_emb = passage_emb

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        return {
            "query_index": rec["query_index"],
            "pos_indices": rec["pos_indices"],
            "neg_indices": rec["neg_indices"],
        }


def load_cache(cache_dir: str | Path):
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    query_emb = np.load(cache_dir / "query_emb.npy", mmap_mode="r")
    passage_emb = np.load(cache_dir / "passage_emb.npy", mmap_mode="r")
    records = []
    with (cache_dir / "records.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    test = [r for r in records if r["split"] == "test"]
    if not test and len(val) > 1:
        rng = np.random.default_rng(int(meta.get("seed", 42)))
        n_test = max(1, len(val) // 2)
        chosen = set(int(i) for i in rng.choice(len(val), size=n_test, replace=False))
        new_val, test = [], []
        for i, rec in enumerate(val):
            rec = dict(rec)
            if i in chosen:
                rec["split"] = "test"
                test.append(rec)
            else:
                new_val.append(rec)
        val = new_val
    return meta, query_emb, passage_emb, train, val, test


def collate_indices(batch):
    return {
        "query_index": [item["query_index"] for item in batch],
        "pos_indices": [item["pos_indices"] for item in batch],
        "neg_indices": [item["neg_indices"] for item in batch],
    }


def to_torch(arr: np.ndarray, idx, device):
    return torch.tensor(np.array(arr[idx], copy=True), dtype=torch.float32, device=device)
