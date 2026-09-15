import re
from collections import defaultdict

import numpy as np

from config import BM25_B, BM25_K1
from retrievers.base import Retriever

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever(Retriever):
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: np.ndarray | None = None
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        self.doc_ids = list(corpus.keys())
        n_docs = len(self.doc_ids)
        lengths = np.zeros(n_docs, dtype=np.float32)
        df: dict[str, int] = defaultdict(int)
        raw_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for i, doc_id in enumerate(self.doc_ids):
            doc = corpus[doc_id]
            tokens = tokenize(f"{doc.get('title', '')} {doc.get('text', '')}")
            lengths[i] = len(tokens)
            tf: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            for token, freq in tf.items():
                df[token] += 1
                raw_postings[token].append((i, freq))

        self.doc_len = lengths
        self.avgdl = float(lengths.mean()) if n_docs else 0.0
        self.idf = {
            token: float(np.log(1.0 + (n_docs - count + 0.5) / (count + 0.5)))
            for token, count in df.items()
        }
        self.postings = {
            token: (
                np.array([item[0] for item in pairs], dtype=np.int32),
                np.array([item[1] for item in pairs], dtype=np.float32),
            )
            for token, pairs in raw_postings.items()
        }

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []

        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        k1, b, avgdl = self.k1, self.b, self.avgdl
        doc_len = self.doc_len

        for token in set(tokenize(query)):
            posting = self.postings.get(token)
            if posting is None:
                continue
            idxs, tfs = posting
            denom = tfs + k1 * (1.0 - b + b * doc_len[idxs] / avgdl)
            scores[idxs] += self.idf[token] * (tfs * (k1 + 1.0)) / denom

        k = min(top_k, len(scores))
        if k == 0:
            return []
        if k == len(scores):
            ranked = np.argsort(scores)[::-1]
        else:
            ranked = np.argpartition(scores, -k)[-k:]
            ranked = ranked[np.argsort(scores[ranked])[::-1]]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked if scores[i] > 0]
