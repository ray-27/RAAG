import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model_loader import resolve_device
from training.dataset import CachedRetrievalDataset, collate_indices, load_cache, to_torch
from training.eval_metrics import MAX_K, METRIC_FNS, METRIC_NAMES, format_metrics
from training.loss import in_pool_contrastive
from training.projection import ProjectionHead

CSV_FIELDS = ["epoch", "split", "loss", *METRIC_NAMES]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a projection head on cached frozen embeddings")
    parser.add_argument("--cache-dir", default=str(config.CACHE_DIR))
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH)
    parser.add_argument("--lr", type=float, default=config.TRAIN_LR)
    parser.add_argument("--tau", type=float, default=config.TRAIN_TAU)
    parser.add_argument("--hidden-dim", type=int, default=config.PROJ_HIDDEN)
    parser.add_argument("--out-dim", type=int, default=config.PROJ_OUT)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--runs-dir", default=str(config.RUNS_DIR))
    parser.add_argument("--run-name", default=None, help="optional suffix for the run folder")
    parser.add_argument("--output-dir", default=None, help="use this run folder instead of a new timestamped one")
    return parser.parse_args()


def make_run_dir(runs_dir: Path, run_name: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{run_name}" if run_name else stamp
    path = runs_dir / name
    suffix = 2
    while path.exists():
        path = runs_dir / f"{name}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def append_csv(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def batch_loss(head, query_emb, passage_emb, batch, tau, device):
    q_idx = batch["query_index"]
    pos_lists = batch["pos_indices"]
    hard_lists = batch["neg_indices"]
    pos_sets = [set(p) for p in pos_lists]
    q_z = head(to_torch(query_emb, q_idx, device))
    p_needed = sorted({pid for ids in pos_lists + hard_lists for pid in ids})
    p_z_all = head(to_torch(passage_emb, p_needed, device))
    p_map = {pid: i for i, pid in enumerate(p_needed)}

    losses = []
    for i in range(len(q_idx)):
        pos_z = p_z_all[[p_map[pid] for pid in pos_lists[i]]]
        neg_ids = []
        seen = set(pos_sets[i])
        for nid in hard_lists[i]:
            if nid not in seen:
                neg_ids.append(nid)
                seen.add(nid)
        for j, other_pos in enumerate(pos_lists):
            if j == i:
                continue
            for pid in other_pos:
                if pid not in seen:
                    neg_ids.append(pid)
                    seen.add(pid)
        if not neg_ids:
            continue
        neg_z = p_z_all[[p_map[nid] for nid in neg_ids]]
        losses.append(in_pool_contrastive(q_z[i], pos_z, neg_z, tau))
    if not losses:
        return None
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_split(head, query_emb, passage_emb, records, device):
    if not records:
        return {}
    head.eval()
    p_z = []
    bs = 2048
    for start in range(0, passage_emb.shape[0], bs):
        chunk = to_torch(passage_emb, slice(start, start + bs), device)
        p_z.append(head(chunk).cpu())
    p_z = torch.cat(p_z, dim=0).numpy()

    buckets = {name: [] for name in METRIC_NAMES}
    for rec in records:
        q = head(to_torch(query_emb, [rec["query_index"]], device)).cpu().numpy()[0]
        scores = p_z @ q
        ranked = np.argsort(-scores)[:MAX_K].tolist()
        relevant = {pid: 1 for pid in rec["pos_indices"]}
        for name, (fn, k) in METRIC_FNS.items():
            buckets[name].append(fn(relevant, ranked, k))
    return {name: float(np.mean(vals)) for name, vals in buckets.items()}


def save_checkpoint(path: Path, head, args, meta, extra: dict | None = None) -> None:
    payload = {
        "state_dict": head.state_dict(),
        "in_dim": meta["dim"],
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "tau": args.tau,
        "meta": meta,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    meta, query_emb, passage_emb, train_recs, val_recs, test_recs = load_cache(args.cache_dir)
    run_dir = Path(args.output_dir) if args.output_dir else make_run_dir(Path(args.runs_dir), args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"
    best_path = run_dir / "projection_head.pt"
    print(meta)
    print(f"run_dir={run_dir} train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")

    if not train_recs:
        raise ValueError("cache has no train split")

    ds = CachedRetrievalDataset(train_recs, query_emb, passage_emb)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_indices,
        drop_last=False,
    )
    head = ProjectionHead(meta["dim"], args.hidden_dim, args.out_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    best_score = -1.0
    best_epoch = 0
    best_val = {}

    run_info = {
        "run_dir": str(run_dir),
        "cache_dir": str(args.cache_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "tau": args.tau,
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "device": device,
        "seed": args.seed,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "metrics": METRIC_NAMES,
        "cache_meta": meta,
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        n_batches = 0
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            loss = batch_loss(head, query_emb, passage_emb, batch, args.tau, device)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_batches += 1
        avg_loss = total / max(n_batches, 1)
        append_csv(csv_path, {"epoch": epoch, "split": "train", "loss": f"{avg_loss:.6f}"})

        val_metrics = evaluate_split(head, query_emb, passage_emb, val_recs, device) if val_recs else {}
        if val_metrics:
            append_csv(csv_path, {"epoch": epoch, "split": "val", "loss": "", **{k: f"{v:.6f}" for k, v in val_metrics.items()}})
            print(f"epoch {epoch} loss={avg_loss:.4f} val " + format_metrics(val_metrics))
            score = val_metrics["mrr@10"]
        else:
            print(f"epoch {epoch} loss={avg_loss:.4f}")
            score = -avg_loss

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val = val_metrics
            save_checkpoint(
                best_path,
                head,
                args,
                meta,
                extra={"epoch": epoch, "split": "val", "metrics": val_metrics},
            )
            print(f"saved {best_path}")

    test_metrics = {}
    if test_recs:
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location=device)
        head.load_state_dict(ckpt["state_dict"])
        test_metrics = evaluate_split(head, query_emb, passage_emb, test_recs, device)
        append_csv(
            csv_path,
            {"epoch": best_epoch, "split": "test", "loss": "", **{k: f"{test_metrics[k]:.6f}" for k in METRIC_NAMES}},
        )
        print(f"test (best epoch {best_epoch}) " + format_metrics(test_metrics))
    else:
        print("no test split in cache; skip test")

    run_info.update({
        "best_epoch": best_epoch,
        "best_val_mrr@10": best_score if val_recs else None,
        "best_val": best_val,
        "test": test_metrics,
        "projection_head": str(best_path),
        "metrics_csv": str(csv_path),
    })
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
