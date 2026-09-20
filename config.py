from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "hotpotqa"
DBPEDIA_DIR = ROOT / "dbpedia"


def pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return "mps"
    return "cpu"


DEVICE = pick_device() 
BATCH_SIZE = 64
SEED = 42
SPLIT = "test"
MAX_QUERIES = 100
MAX_DOCS = 10000
K_VALUES = [1, 5, 10, 20, 50, 100]
RETRIEVERS = ["bm25"]
BM25_K1 = 1.5
BM25_B = 0.75
HYBRID_ALPHA = 0.5
CHUNK = False
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

HARD_NEG_JSONL = DATA_DIR / "hard_negatives" / "train.jsonl"
CACHE_DIR = ROOT / "data" / "training_cache"
CONCAT_TITLE = True
VAL_RATIO = 0.05
TEST_RATIO = 0.05
RUNS_DIR = ROOT / "runs"
PROJ_HIDDEN = 256
PROJ_OUT = 128
TRAIN_TAU = 0.07
TRAIN_LR = 1e-3
TRAIN_BATCH = 64
TRAIN_EPOCHS = 5
