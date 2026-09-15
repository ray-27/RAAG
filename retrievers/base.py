class Retriever:
    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        raise NotImplementedError

    def retrieve_many(
        self, queries: dict[str, str], top_k: int = 10
    ) -> dict[str, list[tuple[str, float]]]:
        return {qid: self.retrieve(query, top_k=top_k) for qid, query in queries.items()}
