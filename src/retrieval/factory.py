"""
Retrieval Manager Factory
Creates appropriate retrieval manager instance based on configuration.
"""

import logging
from typing import Optional

from src.config.retrieval_config import RetrievalConfig

logger = logging.getLogger(__name__)


def create_retrieval_manager(
    retrieval_config: Optional[RetrievalConfig] = None,
    config_name: Optional[str] = None,
    rm_type: Optional[str] = None,
    format_type: Optional[str] = None,
    **kwargs,
):
    """
    Create a retrieval manager based on configuration.

    Args:
        retrieval_config: RetrievalConfig instance (optional)
        config_name: Name of YAML config to load (e.g., 'txtai', 'bm25_wikidump')
        rm_type: Override retrieval manager type (optional)
        format_type: Output format type (optional)
        **kwargs: Additional arguments passed to the retrieval manager

    Returns:
        Configured retrieval manager instance
    """
    # Load from YAML if config name provided
    if config_name and retrieval_config is None:
        retrieval_config = RetrievalConfig.from_yaml(config_name)
        logger.info(f"Loaded retrieval config from: {config_name}")

    # Use default config if none provided
    if retrieval_config is None:
        retrieval_config = RetrievalConfig()

    # Determine retrieval manager type
    manager_type = rm_type or retrieval_config.retrieval_manager_type

    logger.info(f"Creating retrieval manager of type: {manager_type}")

    # Default arguments for all managers
    default_args = {
        "config": retrieval_config,  # Pass the config to managers that support it
        "format_type": format_type,  # Optional format type for output
    }
    default_args.update(kwargs)

    # Create base retrieval manager
    if manager_type == "wiki":
        from src.retrieval.wiki_rm import WikiRM

        base_rm = WikiRM(**default_args)

    elif manager_type == "supabase_faiss":
        try:
            from src.retrieval.supabase_faiss_rm import FaissRM

            base_rm = FaissRM(**default_args)
        except ImportError as e:
            logger.error(f"FAISS dependencies not available: {e}")
            logger.error("Install with: pip install faiss-cpu sentence-transformers")
            raise ImportError(f"FAISSWikiRM not available: {e}")
        except Exception as e:
            logger.error(f"FAISSWikiRM initialization failed: {e}")
            raise RuntimeError(f"FAISSWikiRM failed to initialize: {e}")

    else:
        logger.error(f"Unknown retrieval manager type: {manager_type}")
        logger.error(f"Supported types: wiki, supabase_faiss")
        raise ValueError(f"Unsupported retrieval manager type: {manager_type}")

    return base_rm
