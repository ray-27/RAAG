from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from model_loader import prepare_model


class FrozenEncoder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model = prepare_model(model_name, device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def dim(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64, show_progress_bar: bool | None = None):
        if show_progress_bar is None:
            show_progress_bar = len(texts) > 16
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
