import argparse

import config
from data_loader import BEIR_DATASETS, load_beir, load_dummy
from evaluate import evaluate, format_results
from retrievers import get_retriever


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrievers on a BEIR dataset")
    parser.add_argument(
        "--dataset",
        default="hotpotqa",
        choices=list(BEIR_DATASETS),
        help="BEIR dataset to load and evaluate",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=config.RETRIEVERS,
        help="bm25, cosine, hybrid, or module:ClassName for a custom retriever",
    )
    parser.add_argument("--split", default=config.SPLIT, choices=["train", "validation", "test"])
    parser.add_argument("--max-queries", type=int, default=config.MAX_QUERIES)
    parser.add_argument("--max-docs", type=int, default=config.MAX_DOCS)
    parser.add_argument("--full", action="store_true", help="use the full split and corpus")
    parser.add_argument("--dummy", action="store_true", help="run on a tiny in-memory sample")
    parser.add_argument("--k", nargs="+", type=int, default=config.K_VALUES)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--alpha", type=float, default=config.HYBRID_ALPHA, help="hybrid cosine weight")
    parser.add_argument("--model-path", default=None, help="optional path for a custom retriever")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="local folder for the dataset; downloads from HF only if missing",
    )
    parser.add_argument(
        "--chunk",
        action=argparse.BooleanOptionalAction,
        default=config.CHUNK,
        help="split documents into overlapping chunks before embedding",
    )
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    return parser.parse_args()


def main():
    args = parse_args()
    k_values = tuple(args.k)
    spec = BEIR_DATASETS[args.dataset]
    data_dir = args.data_dir or spec["data_dir"]

    if args.dummy:
        corpus, queries, qrels = load_dummy()
    else:
        max_queries = None if args.full else args.max_queries
        max_docs = None if args.full else args.max_docs
        corpus, queries, qrels = load_beir(
            dataset=args.dataset,
            split=args.split,
            max_queries=max_queries,
            max_docs=max_docs,
            seed=args.seed,
            data_dir=data_dir,
        )

    print(f"dataset={args.dataset} corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")

    kwargs = {
        "model_name": args.model,
        "batch_size": args.batch_size,
        "device": args.device,
        "alpha": args.alpha,
        "model_path": args.model_path,
        "chunk": args.chunk,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    all_results = {}
    for spec_name in args.retrievers:
        retriever = get_retriever(spec_name, **kwargs)
        print(f"\nindexing {spec_name} ...")
        retriever.index(corpus)
        print(f"evaluating {spec_name} ...")
        all_results[spec_name] = evaluate(retriever, queries, qrels, k_values=k_values)

    print("\n" + format_results(all_results, k_values))


if __name__ == "__main__":
    main()
