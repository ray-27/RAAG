import numpy as np
from tqdm import tqdm

from chunking import iter_corpus_chunks
from config import BATCH_SIZE, CHUNK, CHUNK_OVERLAP, CHUNK_SIZE, DEVICE, MODEL_NAME
from model_loader import prepare_model
from retrievers.base import Retriever


class CosineRetriever(Retriever):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = BATCH_SIZE,
        device: str | None = DEVICE,
        chunk: bool = CHUNK,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.chunk = chunk
        self.chunk_size = chunk_size if chunk else None
        self.chunk_overlap = chunk_overlap if chunk else 0
        self.model = None
        self.doc_ids: list[str] = []
        self.chunk_doc_idx: np.ndarray | None = None
        self.embeddings: np.ndarray | None = None

    def _load_model(self):
        if self.model is None:
            self.model = prepare_model(self.model_name, self.device)
        return self.model

    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        model = self._load_model()
        self.doc_ids = list(corpus.keys())
        doc_index = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        ordered = {doc_id: corpus[doc_id] for doc_id in self.doc_ids}

        texts = []
        chunk_doc_idx = []
        for doc_id, chunk in iter_corpus_chunks(
            ordered, chunk_size=self.chunk_size, overlap=self.chunk_overlap
        ):
            texts.append(chunk)
            chunk_doc_idx.append(doc_index[doc_id])

        if self.chunk:
            print(
                f"chunking {len(self.doc_ids)} docs -> {len(texts)} chunks "
                f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
            )
        else:
            print(f"chunking off, 1 vector per doc ({len(texts)} docs)")

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.chunk_doc_idx = np.asarray(chunk_doc_idx, dtype=np.int32)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return self.retrieve_many({"q": query}, top_k=top_k)["q"]

    def retrieve_many(
        self, queries: dict[str, str], top_k: int = 10
    ) -> dict[str, list[tuple[str, float]]]:
        if self.embeddings is None or not self.doc_ids or self.chunk_doc_idx is None:
            return {qid: [] for qid in queries}

        model = self._load_model()
        qids = list(queries.keys())
        q_emb = model.encode(
            [queries[qid] for qid in qids],
            batch_size=self.batch_size,
            show_progress_bar=len(qids) > 8,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        q_emb = np.asarray(q_emb, dtype=np.float32)
        chunk_scores = q_emb @ self.embeddings.T
        n_docs = len(self.doc_ids)
        k = min(top_k, n_docs)
        results = {}
        for row, qid in enumerate(tqdm(qids, desc="ranking", disable=len(qids) < 16)):
            doc_scores = np.full(n_docs, -np.inf, dtype=np.float32)
            np.maximum.at(doc_scores, self.chunk_doc_idx, chunk_scores[row])
            if k == n_docs:
                ranked = np.argsort(doc_scores)[::-1]
            else:
                ranked = np.argpartition(doc_scores, -k)[-k:]
                ranked = ranked[np.argsort(doc_scores[ranked])[::-1]]
            results[qid] = [
                (self.doc_ids[i], float(doc_scores[i]))
                for i in ranked
                if np.isfinite(doc_scores[i])
            ]
        return results
