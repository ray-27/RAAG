import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import DATA_DIR, MAX_DOCS, MAX_QUERIES, SEED, SPLIT


def _field(row, *names):
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(names)


def load_dummy():
    corpus = {
        "d1": {
            "title": "Albert Einstein",
            "text": "Albert Einstein developed the theory of relativity and the mass-energy formula E = mc2.",
        },
        "d2": {
            "title": "Wheat beer",
            "text": "Wheat beer is brewed with a large proportion of wheat relative to malted barley.",
        },
        "d3": {
            "title": "Apollo 11",
            "text": "Apollo 11 landed the first two humans on the Moon in 1969, Neil Armstrong and Buzz Aldrin.",
        },
        "d4": {
            "title": "Python",
            "text": "Python is a high-level programming language used for web development, data science, and automation.",
        },
        "d5": {
            "title": "Photosynthesis",
            "text": "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
        },
    }
    queries = {
        "q1": "Who developed the mass-energy equivalence formula?",
        "q2": "Which beer is brewed with a large proportion of wheat?",
        "q3": "Who were the first humans to land on the Moon?",
    }
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d2": 1},
        "q3": {"d3": 1},
    }
    return corpus, queries, qrels


def _is_saved(path: Path) -> bool:
    return (path / "dataset_info.json").exists() or (path / "dataset_dict.json").exists()


def _save_local(ds, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    ds.save_to_disk(str(tmp))
    if path.exists():
        shutil.rmtree(path)
    shutil.move(str(tmp), str(path))


def prepare_hotpotqa(data_dir: str | Path = DATA_DIR):
    from datasets import load_dataset, load_from_disk

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "corpus": lambda: load_dataset("BeIR/hotpotqa", "corpus", split="corpus"),
        "queries": lambda: load_dataset("BeIR/hotpotqa", "queries", split="queries"),
        "qrels": lambda: load_dataset("BeIR/hotpotqa-qrels"),
    }

    local = {}
    for name, loader in loaders.items():
        path = data_dir / name
        if _is_saved(path):
            print(f"loading {name} from {path}")
            local[name] = load_from_disk(str(path))
            continue
        print(f"downloading {name} from Hugging Face, saving to {path}")
        ds = loader()
        _save_local(ds, path)
        local[name] = ds
    return local


def _load_qrels(ds, max_queries: int | None, seed: int):
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for row in ds:
        qid = str(_field(row, "query-id", "query_id", "_id"))
        doc_id = str(_field(row, "corpus-id", "corpus_id"))
        score = int(_field(row, "score"))
        if score > 0:
            grouped[qid][doc_id] = score

    query_ids = list(grouped.keys())
    if max_queries is not None and max_queries > 0 and max_queries < len(query_ids):
        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(len(query_ids), size=max_queries, replace=False)
        grouped = {query_ids[i]: grouped[query_ids[i]] for i in chosen_idx}
    return grouped


def _load_queries(ds, qrels: dict[str, dict[str, int]]):
    needed = set(qrels)
    queries = {}
    for row in ds:
        qid = str(row["_id"])
        if qid in needed:
            queries[qid] = row["text"]
            if len(queries) == len(needed):
                break
    missing = needed - set(queries)
    if missing:
        raise ValueError(f"Missing {len(missing)} queries in HotpotQA queries split")
    return queries


def _row_doc(row):
    return {
        "title": row.get("title") or "",
        "text": row.get("text") or "",
    }


def _load_corpus(ds, qrels: dict[str, dict[str, int]], max_docs: int | None, seed: int):
    n = len(ds)

    if max_docs is None or max_docs <= 0:
        corpus = {}
        for row in tqdm(ds, desc="load corpus"):
            corpus[str(row["_id"])] = _row_doc(row)
        return corpus

    gold_ids = {doc_id for rels in qrels.values() for doc_id in rels}
    ids = ds["_id"]
    gold_indices = {}
    remaining = set(gold_ids)
    for i, doc_id in enumerate(tqdm(ids, desc="scan corpus ids")):
        doc_id = str(doc_id)
        if doc_id in remaining:
            gold_indices[doc_id] = i
            remaining.remove(doc_id)
            if not remaining:
                break

    if remaining:
        print(f"warning: {len(remaining)} relevant docs not found in corpus")

    selected = set(gold_indices.values())
    extra_needed = max(0, max_docs - len(selected))
    if extra_needed:
        rng = np.random.default_rng(seed)
        candidate = rng.choice(n, size=min(n, extra_needed * 3), replace=False)
        for i in candidate:
            i = int(i)
            if i not in selected:
                selected.add(i)
            if len(selected) >= max_docs:
                break

    corpus = {}
    for i in tqdm(sorted(selected), desc="load corpus"):
        row = ds[int(i)]
        corpus[str(row["_id"])] = _row_doc(row)
    return corpus


def load_hotpotqa(
    split: str = SPLIT,
    max_queries: int | None = MAX_QUERIES,
    max_docs: int | None = MAX_DOCS,
    seed: int = SEED,
    data_dir: str | Path = DATA_DIR,
):
    local = prepare_hotpotqa(data_dir)
    qrels = _load_qrels(local["qrels"][split], max_queries, seed)
    queries = _load_queries(local["queries"], qrels)
    corpus = _load_corpus(local["corpus"], qrels, max_docs, seed)

    keep_qrels = {}
    keep_queries = {}
    for qid, rels in qrels.items():
        present = {doc_id: score for doc_id, score in rels.items() if doc_id in corpus}
        if present:
            keep_qrels[qid] = present
            keep_queries[qid] = queries[qid]

    return corpus, keep_queries, keep_qrels
