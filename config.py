from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "hotpotqa"
DEVICE = "mps"
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
CHUNK = True
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
