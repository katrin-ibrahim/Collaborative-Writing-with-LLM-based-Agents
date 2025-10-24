"""
STORM configuration using ConfigContext.
"""

import logging
from knowledge_storm import STORMWikiLMConfigs

from src.config.config_context import ConfigContext
from src.retrieval.factory import create_retrieval_manager

logger = logging.getLogger(__name__)


def setup_storm_config() -> STORMWikiLMConfigs:
    """
    Setup STORM LM configurations using ConfigContext.

    STORM uses different models for different tasks:
    - conv_simulator_lm: Fast model for conversation simulation
    - question_asker_lm: Fast model for question generation
    - outline_gen_lm: Writing model for outline generation
    - article_gen_lm: Writing model for article generation
    - article_polish_lm: Writing model for article polishing

    Returns:
        Configured STORMWikiLMConfigs
    """
    lm_config = STORMWikiLMConfigs()

    # Use fast model for conversation and question asking
    fast_engine = ConfigContext.get_client("fast")
    writing_engine = ConfigContext.get_client("writing")

    # Set up all STORM LM configurations
    lm_config.set_conv_simulator_lm(fast_engine)  # pyright: ignore[reportArgumentType]
    lm_config.set_question_asker_lm(fast_engine)  # pyright: ignore[reportArgumentType]
    lm_config.set_outline_gen_lm(writing_engine)  # pyright: ignore[reportArgumentType]
    lm_config.set_article_gen_lm(writing_engine)  # pyright: ignore[reportArgumentType]
    lm_config.set_article_polish_lm(
        writing_engine
    )  # pyright: ignore[reportArgumentType]

    logger.info("STORM LM configuration set up with ConfigContext engines")
    return lm_config


def setup_storm_retrieval():
    """
    Setup STORM retrieval manager using ConfigContext.

    Returns:
        Configured retrieval manager for STORM
    """
    retrieval_config = ConfigContext.get_retrieval_config()
    retrieval_manager = create_retrieval_manager(
        rm_type=retrieval_config.retrieval_manager, format_type="storm"
    )

    logger.info(f"STORM retrieval manager set up: {type(retrieval_manager).__name__}")
    return retrieval_manager


def get_storm_config_params() -> dict:
    """
    Get STORM configuration parameters from ConfigContext.

    Returns:
        Dictionary with STORM configuration parameters
    """
    storm_config = ConfigContext.get_storm_config()
    return vars(storm_config)
