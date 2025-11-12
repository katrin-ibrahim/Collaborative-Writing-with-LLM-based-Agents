#!/usr/bin/env python3
import argparse
import gc
import gzip
import hashlib
from pathlib import Path
from urllib.parse import quote

import faiss
import json
import logging
import numpy as np
import os
import psutil
import torch
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Tuple

from src.retrieval.utils.chunker import ContentChunker
from src.retrieval.utils.description_generator import DescriptionGenerator
from src.utils.data.models import ResearchChunk

os.environ["OMP_NUM_THREADS"] = "1"
faiss.omp_set_num_threads(1)

# ----------------------------------------------------------------------
# INDEX STRATEGY EXPLANATION (MANDATORY FOR MACOS/LIMITED RAM)
# ----------------------------------------------------------------------

# PROBLEM:
# On macOS (especially Apple Silicon/MPS) or systems with limited RAM,
# attempting to build large, uncompressed FAISS indexes (like IndexFlatIP)
# often results in a 'Segmentation Fault (SIGSEGV)'. This is caused by
# the underlying C++ libraries (FAISS/PyTorch) attempting to allocate
# more memory than is available in a single, contiguous block,
# or due to library conflicts (OpenMP/MPS) during large allocations.

# SOLUTION: IndexIVFFlat + ScalarQuantizer (IndexIVFSQ)
# To resolve the SIGSEGV and drastically reduce RAM usage, we use a
# composite index:
#
# 1. IndexFlatL2 (quantizer): The top-level index for clustering the centroids.
# 2. IndexScalarQuantizer (SQ): The actual vector storage layer. This layer
#    compresses the 384-dimensional embeddings from float32 (1.5KB/vector)
#    down to an 8-bit integer representation (approx. 384 bytes/vector)
#    with minimal loss of accuracy, making the index stable in memory.
#
# Training is required for SQ, which is why we sample the first batch
# for the K-means clustering step. The overall memory footprint is reduced
# by a factor of 4-5, making the build process robust.

# ----------------------------------------------------------------------


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("chunked_faiss")


def log_memory_usage(context: str = ""):
    """Log current memory usage."""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        log.info(f"Memory usage {context}: {mem_info.rss / 1024**3:.2f} GB")
    except Exception as e:
        log.warning(f"Could not get memory info: {e}")


def make_chunk_id(title: str, chunk_idx: int) -> str:
    """Generate consistent chunk_id"""
    raw = f"wikipedia|{title}|{chunk_idx}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def save_batch_embeddings(
    checkpoint_dir: Path, batch_num: int, embeddings: np.ndarray
) -> Path:
    """Save embeddings to disk."""
    emb_file = checkpoint_dir / f"batch_{batch_num}_embeddings.npy"
    np.save(emb_file, embeddings.astype(np.float32))
    log.info(f"Saved embeddings to {emb_file}")
    return emb_file


def create_research_chunks(
    batch_metadata: List[Tuple[str, int, str, int]],
    batch_chunks: List[str],
    batch_article_data: List[Dict[str, Any]],
    chunk_size: int,
    overlap: int,
    embedding_model: str,
) -> List[Dict[str, Any]]:
    """
    Create ResearchChunk objects from batch data.

    Args:
        batch_metadata: List of (title, chunk_idx, url, total_chunks)
        batch_chunks: List of chunk text content
        batch_article_data: List of full article data dicts
        chunk_size: Chunking size parameter
        overlap: Chunking overlap parameter
        embedding_model: Name of embedding model used

    Returns:
        List of ResearchChunk dicts
    """
    research_chunks = []

    for (title, j, url, total_chunks), chunk_text, article_data in zip(
        batch_metadata, batch_chunks, batch_article_data
    ):
        chunk_id = make_chunk_id(title, j)

        # Use the parametrized description generator
        description = DescriptionGenerator.create_description(
            content=chunk_text,
            title=title,
            chunk_idx=j,
            total_chunks=total_chunks,
            categories=article_data.get("categories", []),
            include_position=True,
            include_categories=True,
            max_preview_length=200,
        )

        research_chunk = ResearchChunk(
            chunk_id=chunk_id,
            description=description,
            content=chunk_text,
            source="wikipedia_simple_english",
            url=url,
            metadata={
                "article_title": title,
                "article_id": article_data.get("id"),
                "chunk_index": j,
                "total_chunks": total_chunks,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "embedding_model": embedding_model,
                "categories": article_data.get("categories", []),
                "language": "simple_english",
            },
        )
        research_chunks.append(research_chunk.model_dump())

    return research_chunks


def save_batch_chunks(
    checkpoint_dir: Path, batch_num: int, research_chunks: List[Dict[str, Any]]
) -> Path:
    """Save chunk metadata to disk."""
    meta_file = checkpoint_dir / f"batch_{batch_num}_chunks.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(research_chunks, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(research_chunks)} chunks to {meta_file}")
    return meta_file


def encode_and_save_batch(
    batch_num: int,
    batch_chunks: List[str],
    batch_metadata: List[Tuple[str, int, str, int]],
    batch_article_data: List[Dict[str, Any]],
    model: SentenceTransformer,
    checkpoint_dir: Path,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    device: str,
) -> int:
    """
    Encode a batch of chunks and save to disk.

    Returns:
        Number of chunks processed
    """
    log.info(f"Encoding batch {batch_num} of {len(batch_chunks)} chunks...")
    log_memory_usage(f"before encoding batch {batch_num}")

    # Encode current batch
    batch_embeddings = model.encode(
        batch_chunks,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device,
    )

    # Save embeddings
    save_batch_embeddings(checkpoint_dir, batch_num, batch_embeddings)

    # Create ResearchChunk objects
    research_chunks = create_research_chunks(
        batch_metadata,
        batch_chunks,
        batch_article_data,
        chunk_size,
        overlap,
        embedding_model,
    )

    # Save chunks
    save_batch_chunks(checkpoint_dir, batch_num, research_chunks)

    # Cleanup
    num_chunks = len(batch_chunks)
    del batch_embeddings, research_chunks
    gc.collect()

    if device == "mps":
        torch.mps.empty_cache()

    log_memory_usage(f"after encoding batch {batch_num}")

    return num_chunks


def process_article(
    article_data: Dict[str, Any],
    chunker: ContentChunker,
    batch_chunks: List[str],
    batch_metadata: List[Tuple[str, int, str, int]],
    batch_article_data: List[Dict[str, Any]],
) -> None:
    """
    Process a single article and add its chunks to the current batch.

    Args:
        article_data: Full article data dict
        chunker: ContentChunker instance
        batch_chunks: List to append chunk text to
        batch_metadata: List to append metadata tuples to
        batch_article_data: List to append article data to
    """
    title = article_data.get("title", "Unknown").strip()
    content = (
        article_data.get("content")
        or article_data.get("body")
        or article_data.get("text", "")
    )

    if not content:
        return

    url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    chunks = chunker.chunk_text(content)

    for j, chunk_text in enumerate(chunks):
        batch_chunks.append(chunk_text)
        batch_metadata.append((title, j, url, len(chunks)))
        batch_article_data.append(article_data)


def check_resume_state(checkpoint_dir: Path, resume: bool) -> int:
    """
    Check for existing checkpoints and determine starting batch.

    Args:
        checkpoint_dir: Directory containing checkpoints
        resume: Whether to resume from checkpoints

    Returns:
        Starting batch number (0 if starting fresh)
    """
    existing_batches = sorted(checkpoint_dir.glob("batch_*_embeddings.npy"))

    if not existing_batches:
        return 0

    if existing_batches and not resume:
        raise RuntimeError(
            f"Checkpoints found in {checkpoint_dir}, but resume=False. "
            f"Use --resume to continue or delete checkpoints to start fresh."
        )

    if resume:
        start_batch = len(existing_batches)
        log.info(
            f"Found {start_batch} existing checkpoint batches. Resuming from batch {start_batch}"
        )
        return start_batch

    return 0


def get_embedding_dim_from_file(npy_path: Path) -> int:
    """
    Read embedding dimension from .npy file without loading full array.

    Args:
        npy_path: Path to .npy file

    Returns:
        Embedding dimension (second element of shape)
    """
    with open(npy_path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported .npy version: {version}")
        return shape[1]


def build_faiss_index_from_checkpoints(
    checkpoint_dir: Path, num_batches: int, use_streaming: bool = True
) -> Tuple[faiss.Index, int]:
    """
    Build FAISS index from saved checkpoint batches.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        num_batches: Total number of batches to load
        use_streaming: If True, stream chunks to avoid loading all in memory

    Returns:
        Tuple of (FAISS index, total number of chunks)
    """
    log.info("Building FAISS index from checkpoints...")
    log_memory_usage("before building index")

    # Get dimensions without loading full array
    first_batch_path = checkpoint_dir / "batch_0_embeddings.npy"
    if not first_batch_path.exists():
        raise FileNotFoundError(f"First batch not found: {first_batch_path}")

    embedding_dim = get_embedding_dim_from_file(first_batch_path)
    log.info(f"Embedding dimension: {embedding_dim}")

    # Initialize FAISS index
    index = faiss.IndexScalarQuantizer(embedding_dim, faiss.ScalarQuantizer.QT_8bit)
    # Since the original IndexFlatIP was performing Inner Product (IP) search
    # (which is Cosine Similarity after normalization), explicitly
    # set the MetricType if using a different quantizer, but the
    # two-argument constructor defaults to METRIC_L2, which is correct for
    # normalized vectors (as IP and L2 are mathematically equivalent for normalized data).

    # Scalar quantization requires a training step before adding vectors
    log.info("Starting FAISS index training (required for compression)...")

    # 2. Get Training Data (Use the first batch)
    emb_file = checkpoint_dir / "batch_0_embeddings.npy"
    if not emb_file.exists():
        raise FileNotFoundError(f"Cannot find batch 0 for training: {emb_file}")

    # E_train prepared
    E_train = np.load(emb_file).astype(np.float32)
    faiss.normalize_L2(E_train)  # in-place

    # sanity checks
    assert E_train.ndim == 2, f"expected 2D array, got {E_train.ndim}D"
    assert (
        E_train.shape[1] == embedding_dim
    ), f"dim mismatch: {E_train.shape[1]} vs {embedding_dim}"
    E_train = np.ascontiguousarray(E_train, dtype=np.float32)

    # train (pick one style)
    # index.train(E_train)  # simple
    index.train(E_train.shape[0], E_train)  # pylance-happy

    assert index.is_trained, "index did not train"
    log.info(f"FAISS index trained with {len(E_train)} vectors.")
    del E_train
    gc.collect()

    temp_chunks_file = checkpoint_dir / "temp_all_chunks.jsonl"
    # Define the permanent file path relative to the final output directory (out_dir)
    final_chunks_jsonl = checkpoint_dir.parent / "chunks.jsonl"  # final location
    total_chunks = 0

    log.info("Streaming mode: writing chunks to temporary file...")

    with open(temp_chunks_file, "w", encoding="utf-8") as out_f:
        for i in range(num_batches):
            log.info(f"Loading batch {i}/{num_batches}...")

            try:
                # Load embeddings, normalize, and add to index (memory required for FAISS)
                # 1) Load embeddings
                emb_file = checkpoint_dir / f"batch_{i}_embeddings.npy"
                E_batch = np.load(emb_file).astype(np.float32)

                # 2) Normalize + ensure contiguity (matches cosine/L2 assumption)
                faiss.normalize_L2(E_batch)
                E_batch = np.ascontiguousarray(E_batch, dtype=np.float32)

                # 3) Add to FAISS (use positional args; no keyword 'x')
                #    Either of these is fine; the second keeps Pylance happiest:
                # index.add(E_batch)
                index.add(E_batch.shape[0], E_batch)

                # Stream chunks to file (memory-safe I/O)
                chunks_file = checkpoint_dir / f"batch_{i}_chunks.json"
                with open(chunks_file, encoding="utf-8") as in_f:
                    batch_chunks = json.load(in_f)
                    for chunk in batch_chunks:
                        out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        total_chunks += 1

                log.info(f"Added {len(E_batch)} vectors. Total: {index.ntotal}")
                del E_batch, batch_chunks
                gc.collect()

            except Exception as e:
                log.error(f"Error loading batch {i}: {e}")
                raise

    # ------------------------------------------------------------------

    log.info(f"All chunks streamed to {temp_chunks_file}")
    temp_chunks_file.rename(final_chunks_jsonl)
    log.info(f"Permanent chunk metadata saved to: {final_chunks_jsonl}")

    log_memory_usage("after building index")
    log.info(f"FAISS index built with {index.ntotal} vectors")
    return index, total_chunks


def save_final_outputs(
    out_dir: Path,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
    index: faiss.Index,
    num_chunks: int,
    embedding_dim: int,
) -> Tuple[Path, Path]:
    """
    Save final FAISS index and metadata.

    Returns:
        Tuple of (index_path, metadata_path)
    """
    model_tag = embedding_model.replace("/", "_").replace("-", "_")
    out_json = out_dir / f"wikipedia_{model_tag}_cs{chunk_size}_ov{overlap}.json"
    out_index = out_json.with_suffix(".index")

    log.info(f"Writing FAISS index to {out_index}...")
    log_memory_usage("before writing index")
    faiss.write_index(index, str(out_index))

    log.info(f"Writing chunks metadata to {out_json}...")
    CHUNKS_JSONL_FILENAME = "chunks.jsonl"

    meta = {
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "num_chunks": num_chunks,  # Use the count
        "source": "wikipedia_simple_english",
        # New key pointing to the separate, large metadata file
        "chunk_data_path": CHUNKS_JSONL_FILENAME,
        # "chunks": all_chunks,  <--- LINE REMOVED to save memory
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log_memory_usage("after writing outputs")

    return out_index, out_json


def build(
    input_jsonl_gz: Path,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
    out_dir: Path,
    batch_articles: int = 25000,
    resume: bool = False,
    use_streaming: bool = True,
):
    """
    Build FAISS index from Wikipedia embeddings with checkpoint support.

    Args:
        input_jsonl_gz: Path to input NDJSON.gz file
        embedding_model: Name of sentence transformer model
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
        out_dir: Output directory for index and metadata
        batch_articles: Number of articles per batch
        resume: Whether to resume from existing checkpoints
        use_streaming: Use streaming mode to reduce memory usage
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"Using device: {device}")
    log.info(f"Streaming mode: {use_streaming}")
    log_memory_usage("initial")

    model = SentenceTransformer(embedding_model, device=device)
    chunker = ContentChunker(chunk_size=chunk_size, overlap=overlap)

    # Check for existing checkpoints
    start_batch = check_resume_state(checkpoint_dir, resume)
    articles_to_skip = start_batch * batch_articles

    # Process in batches
    current_batch_chunks = []
    current_batch_metadata = []
    current_batch_article_data = []
    article_count = 0
    batch_num = start_batch

    log.info(f"Processing articles in batches of {batch_articles}...")
    if articles_to_skip > 0:
        log.info(f"Skipping first {articles_to_skip} articles (already processed)")

    with gzip.open(input_jsonl_gz, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # Skip already processed articles
            if i < articles_to_skip:
                continue

            article_data = json.loads(line)

            # Process article and add to current batch
            process_article(
                article_data,
                chunker,
                current_batch_chunks,
                current_batch_metadata,
                current_batch_article_data,
            )

            article_count += 1

            # Process batch when we hit the limit
            if article_count % batch_articles == 0:
                log.info(
                    f"Processed {article_count + articles_to_skip} articles → "
                    f"{len(current_batch_chunks)} chunks in current batch"
                )

                encode_and_save_batch(
                    batch_num,
                    current_batch_chunks,
                    current_batch_metadata,
                    current_batch_article_data,
                    model,
                    checkpoint_dir,
                    chunk_size,
                    overlap,
                    embedding_model,
                    device,
                )

                # Clear batch
                current_batch_chunks = []
                current_batch_metadata = []
                current_batch_article_data = []
                batch_num += 1

                # Periodic garbage collection
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()

    # Process remaining chunks
    if current_batch_chunks:
        log.info(
            f"Encoding final batch {batch_num} of {len(current_batch_chunks)} chunks..."
        )
        encode_and_save_batch(
            batch_num,
            current_batch_chunks,
            current_batch_metadata,
            current_batch_article_data,
            model,
            checkpoint_dir,
            chunk_size,
            overlap,
            embedding_model,
            device,
        )
        batch_num += 1

    log.info(f"All batches encoded and saved. Total batches: {batch_num}")

    log.info("Releasing model from memory...")
    del model, chunker
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    log_memory_usage("after releasing model")

    # Build FAISS index from checkpoints
    index, num_chunks = build_faiss_index_from_checkpoints(
        checkpoint_dir, batch_num, use_streaming=use_streaming
    )

    # Save final outputs
    embedding_dim = get_embedding_dim_from_file(
        checkpoint_dir / "batch_0_embeddings.npy"
    )
    out_index, out_json = save_final_outputs(
        out_dir,
        embedding_model,
        chunk_size,
        overlap,
        index,
        num_chunks,  # <-- CHANGED: Passing the count
        embedding_dim,
    )

    log.info(f"✓ Complete! {num_chunks} chunks indexed")
    log.info(f"  Index: {out_index}")
    log.info(f"  Metadata: {out_json}")
    log_memory_usage("final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build chunked FAISS index from Wikipedia embeddings"
    )
    ap.add_argument("--embedding-model", required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--overlap", type=int, required=True)
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-dir", default="data/faiss_indexes")
    ap.add_argument(
        "--batch-articles",
        type=int,
        default=25000,
        help="Number of articles per batch (reduce if running out of memory)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoints if available",
    )
    ap.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (loads all chunks in memory)",
    )
    args = ap.parse_args()

    build(
        Path(args.input_json),
        args.embedding_model,
        args.chunk_size,
        args.overlap,
        args.output_dir,
        args.batch_articles,
        args.resume,
        use_streaming=not args.no_streaming,
    )
