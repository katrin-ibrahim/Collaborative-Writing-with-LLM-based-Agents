import sys
from pathlib import Path

import logging
from knowledge_storm import (
    STORMWikiLMConfigs,
    STORMWikiRunner,
    STORMWikiRunnerArguments,
)

from ...config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.config.baselines_model_config import ModelConfig
from src.retrieval.wiki_rm import WikiRM
from src.utils.ollama_client import OllamaClient

from .runner_utils import get_model_wrapper


def setup_storm_runner(
    client: OllamaClient = None,
    config: ModelConfig = None,
    storm_output_dir: str = None,
    storm_config: dict = None,
):
    """
    Setup STORM runner with optional configuration override.

    Args:
        client: Ollama client
        config: Model configuration
        storm_output_dir: Output directory for STORM
        storm_config: Optional Storm configuration parameters
    """
    lm_config = STORMWikiLMConfigs()

    lm_config.set_conv_simulator_lm(get_model_wrapper(client, config, "fast"))
    lm_config.set_question_asker_lm(get_model_wrapper(client, config, "fast"))
    lm_config.set_outline_gen_lm(get_model_wrapper(client, config, "outline"))
    lm_config.set_article_gen_lm(get_model_wrapper(client, config, "writing"))
    lm_config.set_article_polish_lm(get_model_wrapper(client, config, "polish"))

    # Default Storm configuration
    default_config = {
        "max_conv_turn": 4,
        "max_perspective": 4,
        "max_search_queries_per_turn": DEFAULT_RETRIEVAL_CONFIG.num_queries,
        "search_top_k": DEFAULT_RETRIEVAL_CONFIG.results_per_query,
        "max_thread_num": 4,
    }

    # Merge with provided config
    if storm_config:
        default_config.update(storm_config)

    # Setup search retrieval with configured parameters
    search_rm = WikiRM()
    logging.getLogger("baselines.wikipedia_search").setLevel(logging.DEBUG)

    engine_args = STORMWikiRunnerArguments(
        output_dir=storm_output_dir,
        max_conv_turn=default_config["max_conv_turn"],
        max_perspective=default_config["max_perspective"],
        search_top_k=default_config["search_top_k"],
        max_thread_num=default_config["max_thread_num"],
    )

    runner = STORMWikiRunner(engine_args, lm_config, search_rm)
    return runner, storm_output_dir
