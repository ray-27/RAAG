from retrievers.base import Retriever


class CustomRetriever(Retriever):
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.corpus = {}
        self.doc_ids = []
        self.model = None

    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        if self.model_path:
            # self.model = joblib.load(self.model_path)
            raise NotImplementedError("Load your trained model in index()")

    def score(self, query: str, title: str, text: str) -> float:
        q_terms = set(query.lower().split())
        d_terms = set(f"{title} {text}".lower().split())
        if not q_terms:
            return 0.0
        return len(q_terms & d_terms) / len(q_terms)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scored = [
            (doc_id, float(self.score(query, doc.get("title", ""), doc.get("text", ""))))
            for doc_id, doc in self.corpus.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
