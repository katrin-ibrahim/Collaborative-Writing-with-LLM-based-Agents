"""
Retrieval module for the Collaborative Writing with LLM-based Agents project.
Contains multiple retrieval managers for HER testing and fair comparison.
"""

from .base_retriever import BaseRetriever

# HER Testing Components
from .data_loader import WikipediaDataLoader

# Factory for creating retrieval managers
from .factory import create_enhanced_retrieval_manager, create_retrieval_manager
from .wiki_rm import WikiRM
from .wikidata_enhancer import WikidataEnhancer

# Retrieval Managers (with graceful imports)
try:
    from .bm25_wiki_rm import BM25WikiRM
except ImportError:
    BM25WikiRM = None

try:
    from .faiss_wiki_rm import FAISSWikiRM
except ImportError:
    FAISSWikiRM = None

__all__ = [
    "BaseRetriever",
    "WikiRM",
    "WikipediaDataLoader",
    "WikidataEnhancer",
    "create_retrieval_manager",
    "create_enhanced_retrieval_manager",
    "BM25WikiRM",
    "FAISSWikiRM",
]
