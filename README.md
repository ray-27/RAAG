# RAAG

Frozen MiniLM encoder + trainable projection head for retrieval. Device is CUDA, then MPS, then CPU.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch
```

For GPU, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org).

## Download

Datasets and the encoder are fetched once from Hugging Face and cached locally (`data/hotpotqa`, `dbpedia/`, `models/`).

```bash
python -c "from data_loader import prepare_beir; prepare_beir('hotpotqa'); prepare_beir('dbpedia-entity')"
python -c "from model_loader import prepare_model; prepare_model()"
```

Hard negatives for training (needs Arrow; skip if `data/hotpotqa/hard_negatives/train.jsonl` already exists):

```bash
make -C training
./training/mine_hard_negatives --split train
```

## Train

```bash
python training/prepare_cache.py
python training/train.py --run-name try1
```

Writes `runs/<timestamp>_try1/` (`projection_head.pt`, `metrics.csv`, `run.json`). Query split is 90% train / 5% val / 5% test on HotpotQA hard negatives.

## Test

HotpotQA cache test is part of `train.py`. DBpedia (after training):

```bash
python training/test_dbpedia.py --model runs/<run_id>
```

Defaults to CUDA. `--model` can be the run folder or `projection_head.pt`. Writes `metrics.csv`, `metrics.png`, `results.json` under the run directory.

```bash
python training/test_dbpedia.py --model runs/<run_id> --full --batch-size 512
```

BM25 / cosine baseline on a dataset:

```bash
python run.py --dataset hotpotqa --retrievers bm25
python run.py --dataset dbpedia-entity --retrievers bm25 cosine
```
