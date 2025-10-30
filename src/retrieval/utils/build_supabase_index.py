#!/usr/bin/env python3
import argparse
import gzip
from pathlib import Path

import faiss
import json
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("supabase_faiss")


def build_index(ndjson_gz_path: Path, out_dir: Path, use_quantized: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading embeddings from {ndjson_gz_path}...")
    embeddings = []
    metadata = []

    # Read gzip NDJSON
    with gzip.open(ndjson_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            emb = obj.get("embedding")
            if not emb:
                continue
            embeddings.append(emb)
            # Keep minimal metadata
            metadata.append(
                {
                    "id": obj.get("id"),
                    "text": obj.get("content", "")[:100],  # short preview
                }
            )

    embeddings = np.array(embeddings, dtype=np.float32)
    log.info(f"Loaded {len(embeddings)} embeddings with dim={embeddings.shape[1]}")

    # Normalize embeddings (cosine similarity)
    faiss.normalize_L2(embeddings)

    # Pick index type
    if use_quantized:
        log.info("Using scalar-quantized index (8-bit)")
        index = faiss.IndexScalarQuantizer(
            embeddings.shape[1], faiss.ScalarQuantizer.QT_8bit
        )

        index.train(embeddings.shape[0], embeddings)
    else:
        log.info("Using flat inner product index")
        index = faiss.IndexFlatIP(embeddings.shape[1])

    # Add to index
    index.add(embeddings.shape[0], embeddings)
    log.info(f"Index built with {index.ntotal} vectors")

    # Save outputs
    model_tag = ndjson_gz_path.stem
    out_index = out_dir / f"{model_tag}.index"
    out_meta = out_dir / f"{model_tag}_meta.jsonl"

    faiss.write_index(index, str(out_index))
    with open(out_meta, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    log.info(f"Saved index → {out_index}")
    log.info(f"Saved metadata → {out_meta}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build FAISS index from Supabase embeddings (NDJSON.gz)"
    )
    ap.add_argument("--input", required=True, help="Path to wiki_gte.ndjson.gz")
    ap.add_argument("--out-dir", default="data/faiss_indexes", help="Output directory")
    ap.add_argument(
        "--no-quant",
        action="store_true",
        help="Use uncompressed IndexFlatIP instead of quantized",
    )
    args = ap.parse_args()

    build_index(Path(args.input), Path(args.out_dir), use_quantized=not args.no_quant)
