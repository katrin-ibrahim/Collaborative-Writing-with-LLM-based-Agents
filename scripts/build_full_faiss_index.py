#!/usr/bin/env python3
"""
Build a full-scale FAISS index from all Wikipedia articles.
This creates a production-ready index that can handle the full Wikipedia dump efficiently.
"""

import argparse
import sys
from pathlib import Path

import faiss
import logging
import os
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from retrieval.data_loader import WikipediaDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_full_faiss_index(
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 1000,
    max_articles: int = None,
    output_dir: str = ".",
):
    """
    Build a FAISS index from all available Wikipedia articles.

    Args:
        embedding_model: SentenceTransformer model name
        batch_size: Number of articles to process at once
        max_articles: Maximum articles to process (None = all)
        output_dir: Directory to save index files
    """
    logger.info(f"Building full FAISS index with model: {embedding_model}")

    # Load all available articles
    logger.info("Loading Wikipedia articles...")
    articles = WikipediaDataLoader.load_articles(num_articles=max_articles or 1000000)
    logger.info(f"Loaded {len(articles)} articles")

    # Load embedding model
    logger.info(f"Loading embedding model: {embedding_model}")
    encoder = SentenceTransformer(embedding_model)

    # Process articles in batches to avoid memory issues
    all_embeddings = []
    article_metadata = []

    for i in tqdm(range(0, len(articles), batch_size), desc="Creating embeddings"):
        batch = articles[i : i + batch_size]

        # Extract text (truncate for efficiency)
        texts = [
            article["text"][:512] for article in batch
        ]  # Use more text than before

        # Create embeddings
        embeddings = encoder.encode(texts, show_progress_bar=False)
        all_embeddings.append(embeddings)

        # Store metadata (title, url, etc.)
        for article in batch:
            article_metadata.append(
                {
                    "title": article["title"],
                    "url": article["url"],
                    "text_preview": article["text"][
                        :200
                    ],  # Store preview for quick access
                }
            )

    # Combine all embeddings
    logger.info("Combining embeddings...")
    import numpy as np

    all_embeddings = np.vstack(all_embeddings)

    # Build FAISS index
    logger.info("Building FAISS index...")
    dimension = all_embeddings.shape[1]

    # Use IndexFlatIP for exact search (can upgrade to IndexIVFFlat for larger datasets)
    if len(articles) > 100000:
        # Use approximate search for large datasets
        nlist = min(4096, len(articles) // 100)  # Number of clusters
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
        )

        # Train the index
        logger.info("Training FAISS index...")
        faiss.normalize_L2(all_embeddings)
        index.train(all_embeddings.astype("float32"))
        index.add(all_embeddings.astype("float32"))

        # Set search parameters
        index.nprobe = min(128, nlist // 4)  # Number of clusters to search
    else:
        # Use exact search for smaller datasets
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(all_embeddings)
        index.add(all_embeddings.astype("float32"))

    # Save everything
    num_articles = len(articles)
    embeddings_file = os.path.join(
        output_dir, f"faiss_embeddings_full_{num_articles}.pkl"
    )
    index_file = os.path.join(output_dir, f"faiss_index_full_{num_articles}.index")
    metadata_file = os.path.join(output_dir, f"faiss_metadata_full_{num_articles}.pkl")

    logger.info("Saving index files...")

    # Save embeddings
    with open(embeddings_file, "wb") as f:
        pickle.dump(all_embeddings, f)

    # Save FAISS index
    faiss.write_index(index, index_file)

    # Save metadata
    with open(metadata_file, "wb") as f:
        pickle.dump(article_metadata, f)

    logger.info(f"Full FAISS index built successfully!")
    logger.info(f"Articles indexed: {num_articles}")
    logger.info(f"Index dimension: {dimension}")
    logger.info(f"Files saved:")
    logger.info(f"  - Index: {index_file}")
    logger.info(f"  - Embeddings: {embeddings_file}")
    logger.info(f"  - Metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Build full-scale FAISS index")
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="Embedding model name"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for processing"
    )
    parser.add_argument(
        "--max-articles", type=int, default=None, help="Maximum articles to process"
    )
    parser.add_argument("--output-dir", default=".", help="Output directory")

    args = parser.parse_args()

    build_full_faiss_index(
        embedding_model=args.model,
        batch_size=args.batch_size,
        max_articles=args.max_articles,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
