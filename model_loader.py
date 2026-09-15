from pathlib import Path

import config


def local_model_path(model_name: str | None = None) -> Path:
    name = (model_name or config.MODEL_NAME).replace("/", "--")
    return Path(config.MODEL_DIR) / name


def _is_saved(path: Path) -> bool:
    return (path / "config.json").exists() or (path / "modules.json").exists()


def resolve_device(device: str | None = None) -> str:
    import torch

    device = device or config.DEVICE
    if device == "mps" and not (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ):
        print("MPS not available, using cpu")
        return "cpu"
    return device


def prepare_model(model_name: str | None = None, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    model_name = model_name or config.MODEL_NAME
    device = resolve_device(device)
    path = local_model_path(model_name)
    if _is_saved(path):
        print(f"loading model from {path} on {device}")
        return SentenceTransformer(str(path), device=device)

    print(f"downloading {model_name}, saving to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, device=device)
    model.save(str(path))
    return model
