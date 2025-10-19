"""
Retrieval module for the Collaborative Writing with LLM-based Agents project.
Contains multiple retrieval managers for HER testing and fair comparison.
"""

# Factory for creating retrieval managers
from .factory import create_retrieval_manager
from .rms.base_retriever import BaseRetriever
from .rms.wiki_rm import WikiRM

# Retrieval Managers (with graceful imports)
try:
    from .rms.faiss_rm import FaissRM
except ImportError as e:
    print(f"Warning: Could not import FaissRM: {e}")
    FaissRM = None


__all__ = [
    "BaseRetriever",
    "WikiRM",
    "FaissRM",
    "create_retrieval_manager",
]
