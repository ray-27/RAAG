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
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if device == "mps" and mps is not None and mps.is_available() and mps.is_built():
        return "mps"
    if device == "cpu":
        return "cpu"
    picked = config.pick_device()
    if device != picked:
        print(f"{device} not available, using {picked}")
    return picked


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
