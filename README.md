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

Hard negatives for training (skip if `data/hotpotqa/hard_negatives/train.jsonl` already exists). Needs Apache Arrow.

Ubuntu:

```bash
sudo apt update
sudo apt install -y g++ pkg-config wget ca-certificates lsb-release
wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt install -y ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt update
sudo apt install -y libarrow-dev
```

macOS: `brew install apache-arrow`

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

Defaults to CUDA, every test query, and 100k docs (all golds plus random distractors). `--model` can be the run folder or `projection_head.pt`. Writes `metrics.csv`, `metrics.png`, `results.json` under the run directory.

```bash
python training/test_dbpedia.py --model runs/<run_id> --full --batch-size 512
```

`--full` encodes the entire 4.63M corpus and is much slower.

BM25 / cosine baseline on a dataset:

```bash
python run.py --dataset hotpotqa --retrievers bm25
python run.py --dataset dbpedia-entity --retrievers bm25 cosine
```
