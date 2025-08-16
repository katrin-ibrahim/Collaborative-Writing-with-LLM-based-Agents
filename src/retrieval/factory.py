"""
Retrieval Manager Factory
Creates appropriate retrieval manager instances based on configuration.
"""

import logging
from typing import Optional

from src.config.retrieval_config import RetrievalConfig

logger = logging.getLogger(__name__)


def create_retrieval_manager(
    retrieval_config: Optional[RetrievalConfig] = None,
    config_name: Optional[str] = None,
    rm_type: Optional[str] = None,
    **kwargs,
):
    """
    Create a retrieval manager based on configuration.

    Args:
        retrieval_config: RetrievalConfig instance (optional)
        config_name: Name of YAML config to load (e.g., 'txtai', 'bm25_wikidump')
        rm_type: Override retrieval manager type (optional)
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
        "max_articles": retrieval_config.results_per_query,
        "max_sections": retrieval_config.max_content_pieces,
        "config": retrieval_config,  # Pass the config to managers that support it
    }
    default_args.update(kwargs)

    if manager_type == "wiki":
        from src.retrieval.wiki_rm import WikiRM

        return WikiRM(**default_args)

    elif manager_type == "bm25_wiki":
        try:
            from src.retrieval.bm25_wiki_rm import BM25WikiRM

            num_articles = kwargs.get("num_articles", 100000)
            # Don't pass default_args to avoid parameter conflicts
            return BM25WikiRM(num_articles=num_articles)
        except ImportError as e:
            logger.error(f"BM25 dependencies not available: {e}")
            logger.error("Install with: pip install rank-bm25")
            raise ImportError(f"BM25WikiRM not available: {e}")
        except Exception as e:
            logger.error(f"BM25WikiRM initialization failed: {e}")
            raise RuntimeError(f"BM25WikiRM failed to initialize: {e}")

    elif manager_type == "faiss_wiki":
        try:
            from src.retrieval.faiss_wiki_rm import FAISSWikiRM

            num_articles = kwargs.get("num_articles", 100000)
            embedding_model = kwargs.get("embedding_model", "all-MiniLM-L6-v2")
            # Don't pass default_args to avoid parameter conflicts
            return FAISSWikiRM(
                num_articles=num_articles, embedding_model=embedding_model
            )
        except ImportError as e:
            logger.error(f"FAISS dependencies not available: {e}")
            logger.error("Install with: pip install faiss-cpu sentence-transformers")
            raise ImportError(f"FAISSWikiRM not available: {e}")
        except Exception as e:
            logger.error(f"FAISSWikiRM initialization failed: {e}")
            raise RuntimeError(f"FAISSWikiRM failed to initialize: {e}")

    else:
        logger.error(f"Unknown retrieval manager type: {manager_type}")
        logger.error(f"Supported types: wiki, bm25_wiki, faiss_wiki")
        raise ValueError(f"Unsupported retrieval manager type: {manager_type}")


def create_enhanced_retrieval_manager(
    base_rm_type: str = "wiki",
    config_name: Optional[str] = None,
    use_wikidata_enhancement: bool = False,
    retrieval_config: Optional[RetrievalConfig] = None,
    **kwargs,
):
    """
    Create a retrieval manager with optional Wikidata enhancement.

    Args:
        base_rm_type: Base retrieval manager type
        config_name: Name of YAML config to load (e.g., 'txtai', 'bm25_wikidump')
        use_wikidata_enhancement: Whether to wrap with WikidataEnhancer
        retrieval_config: RetrievalConfig instance
        **kwargs: Additional arguments

    Returns:
        Retrieval manager (optionally enhanced)
    """
    # Load from YAML if config name provided
    if config_name:
        retrieval_config = RetrievalConfig.from_yaml(config_name)
        # Override enhancement setting from config if available
        if hasattr(retrieval_config, "use_wikidata_enhancement"):
            use_wikidata_enhancement = getattr(
                retrieval_config, "use_wikidata_enhancement", use_wikidata_enhancement
            )

    # Create base retrieval manager
    base_rm = create_retrieval_manager(
        retrieval_config=retrieval_config, rm_type=base_rm_type, **kwargs
    )

    # Optionally enhance with Wikidata
    if use_wikidata_enhancement:
        try:
            from src.retrieval.wikidata_enhancer import WikidataEnhancer

            logger.info("Enhancing retrieval manager with Wikidata entities")
            return WikidataEnhancer(base_rm)
        except ImportError as e:
            logger.warning(f"Wikidata enhancement not available: {e}")
            return base_rm

    return base_rm
