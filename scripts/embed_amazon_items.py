from __future__ import annotations

import argparse
from pathlib import Path

import torch

from regenrec.embeddings import (
    blair_text_embeddings,
    hashing_text_embeddings,
    load_item_texts,
    save_embedding_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed Amazon Reviews 2023 item metadata")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/amazon2023/processed/Musical_Instruments/last_out/items.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to a backend-specific file next to --input",
    )
    parser.add_argument(
        "--backend",
        choices=("blair", "sentence-transformers", "hashing"),
        default="blair",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--hash-dimension", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    item_ids, texts = load_item_texts(args.input)
    default_names = {
        "blair": "item_embeddings_blair.pt",
        "sentence-transformers": "item_embeddings_sentence_transformer.pt",
        "hashing": "item_embeddings_hashing.pt",
    }
    output_path = args.output or args.input.parent / default_names[args.backend]
    if args.backend == "hashing":
        embeddings = hashing_text_embeddings(texts, dimension=args.hash_dimension)
        metadata = {"backend": "hashing", "dimension": args.hash_dimension}
    elif args.backend == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for semantic item embeddings. "
                "Install it with: pip install -e '.[embeddings]'"
            ) from exc
        model_name = args.model or "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name, device=args.device)
        model.max_seq_length = min(args.max_length, model.max_seq_length)
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).to(torch.float32)
        metadata = {
            "backend": "sentence-transformers",
            "model": model_name,
            "normalized": True,
            "max_length": model.max_seq_length,
        }
    else:
        model_name = args.model or "hyp1231/blair-roberta-base"
        embeddings = blair_text_embeddings(
            texts,
            model_name=model_name,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        metadata = {
            "backend": "blair",
            "model": model_name,
            "pooling": "cls",
            "normalized": True,
            "max_length": args.max_length,
        }
    save_embedding_bundle(
        output_path,
        item_ids=item_ids,
        embeddings=embeddings,
        metadata=metadata,
    )
    print(
        f"saved {len(item_ids)} item embeddings with dimension {embeddings.size(1)} "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
