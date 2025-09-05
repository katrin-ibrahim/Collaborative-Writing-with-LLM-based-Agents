"""
Retrieval Manager Factory
Creates appropriate retrieval manager instance based on configuration.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_retrieval_manager(
    rm_type: Optional[str] = None,
    format_type: Optional[str] = None,
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

    # Default arguments for all managers
    default_args = {
        "format_type": rm_type,  # Optional format type for output
    }
    default_args.update(kwargs)

    # Create base retrieval manager
    if rm_type == "wiki":
        from src.retrieval.rms.wiki_rm import WikiRM

        base_rm = WikiRM(**default_args)

    elif rm_type == "supabase_faiss":
        try:
            from src.retrieval.rms.supabase_faiss_rm import FaissRM

            base_rm = FaissRM(**default_args)
        except ImportError as e:
            logger.error(f"FAISS dependencies not available: {e}")
            logger.error("Install with: pip install faiss-cpu sentence-transformers")
            raise ImportError(f"FAISSWikiRM not available: {e}")
        except Exception as e:
            logger.error(f"FAISSWikiRM initialization failed: {e}")
            raise RuntimeError(f"FAISSWikiRM failed to initialize: {e}")

    else:
        logger.error(f"Unknown retrieval manager type: {rm_type}")
        logger.error(f"Supported types: wiki, supabase_faiss")
        raise ValueError(f"Unsupported retrieval manager type: {rm_type}")

    return base_rm
