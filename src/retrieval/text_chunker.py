import numpy as np
from typing import List, Tuple


class TextChunker:
    def __init__(self, passage_min_length: int, passage_max_length: int, embedder=None):
        self.min_len = passage_min_length
        self.max_len = passage_max_length
        self.embedder = (
            embedder  # plug in any embedding model (e.g., sentence-transformers)
        )

    def create_chunks(self, content: str) -> List[str]:
        """Split free text into chunks within [min_len, max_len]."""
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]

        if not paragraphs:
            sentences = [
                s.strip() + "." for s in content.split(".") if len(s.strip()) > 20
            ]
            paragraphs = sentences if sentences else [content[: self.max_len]]

        chunks, current = [], ""
        for para in paragraphs:
            if len(current + " " + para) > self.max_len and current:
                if len(current) >= self.min_len:
                    chunks.append(current.strip())
                current = para
            else:
                current += " " + para if current else para

        if current and len(current) >= self.min_len:
            chunks.append(current.strip())

        return chunks

    def rerank(
        self, query: str, chunks: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Rerank chunks by semantic similarity to query."""
        if not self.embedder:
            raise ValueError("No embedder provided for reranking")

        query_emb = self.embedder.embed([query])[0]
        chunk_embs = self.embedder.embed(chunks)

        scores = np.dot(chunk_embs, query_emb) / (
            np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
