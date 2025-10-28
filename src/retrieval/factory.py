"""
Retrieval Manager Factory
Creates appropriate retrieval manager instance based on configuration.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global cache for retrieval manager instances to prevent multiple model downloads
_retrieval_manager_cache = {}


def create_retrieval_manager(
    rm_type: Optional[str] = None,
    **kwargs,
):
    """
    Create a retrieval manager based on configuration.

    Args:
        config_name: Name of YAML config to load (e.g., 'txtai', 'bm25_wikidump')
        rm_type: Override retrieval manager type (optional)
        format_type: Output format type (optional)
        **kwargs: Additional arguments passed to the retrieval manager

    Returns:
        Configured retrieval manager instance
    """

    # Create cache key from rm_type and kwargs
    cache_key = (
        f"{rm_type}_{hash(frozenset(kwargs.items()) if kwargs else frozenset())}"
    )

    # Return cached instance if available
    if cache_key in _retrieval_manager_cache:
        logger.debug(f"Using cached retrieval manager: {rm_type}")
        return _retrieval_manager_cache[cache_key]

    logger.info(f"Creating new retrieval manager: {rm_type}")

    # Default arguments for all managers
    default_args = {
        "format_type": rm_type,  # Optional format type for output
    }
    default_args.update(kwargs)

    # Create base retrieval manager
    if rm_type == "wiki":
        from src.retrieval.rms.wiki_rm import WikiRM

        base_rm = WikiRM(**default_args)

    elif rm_type == "faiss":
        try:
            from src.retrieval.rms.faiss_rm import FaissRM

            base_rm = FaissRM(**default_args)
        except ImportError as e:
            logger.error(f"FAISS dependencies not available: {e}")
            logger.error("Install with: pip install faiss-cpu sentence-transformers")
            raise ImportError(f"FAISSWikiRM not available: {e}")
        except Exception as e:
            logger.error(f"FAISSWikiRM initialization failed: {e}")
            raise RuntimeError(f"FAISSWikiRM failed to initialize: {e}")

    elif rm_type == "hybrid":
        try:
            from src.retrieval.rms.hybrid_rm import HybridRM

            base_rm = HybridRM(**default_args)
        except Exception as e:
            logger.error(f"HybridRM initialization failed: {e}")
            raise RuntimeError(f"HybridRM failed to initialize: {e}")

    else:
        logger.error(f"Unknown retrieval manager type: {rm_type}")
        logger.error(f"Supported types: wiki, faiss, hybrid")
        raise ValueError(f"Unsupported retrieval manager type: {rm_type}")

    # Cache the instance to avoid repeated model downloads
    _retrieval_manager_cache[cache_key] = base_rm
    logger.debug(f"Cached retrieval manager: {rm_type}")

    return base_rm
