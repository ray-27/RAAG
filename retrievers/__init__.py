import inspect

from retrievers.bm25 import BM25Retriever
from retrievers.cosine import CosineRetriever
from retrievers.hybrid import HybridRetriever

REGISTRY = {
    "bm25": BM25Retriever,
    "cosine": CosineRetriever,
    "hybrid": HybridRetriever,
}


def _filter_kwargs(cls, kwargs):
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {k: v for k, v in kwargs.items() if v is not None}
    return {k: v for k, v in kwargs.items() if k in params and v is not None}


def get_retriever(spec: str, **kwargs):
    if spec in REGISTRY:
        cls = REGISTRY[spec]
        return cls(**_filter_kwargs(cls, kwargs))

    if ":" not in spec:
        raise ValueError(
            f"Unknown retriever '{spec}'. Built-in: {list(REGISTRY)}. "
            "Custom format: path.to.module:ClassName"
        )

    import importlib

    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**_filter_kwargs(cls, kwargs))
