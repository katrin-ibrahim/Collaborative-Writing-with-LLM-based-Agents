import sys
from pathlib import Path

import logging
from knowledge_storm import (
    STORMWikiLMConfigs,
    STORMWikiRunner,
    STORMWikiRunnerArguments,
)

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.utils.clients import OllamaClient

from src.config.baselines_model_config import ModelConfig
from src.config.retrieval_config import RetrievalConfig
from src.retrieval import create_retrieval_manager

from .runner_utils import get_model_wrapper


def setup_storm_runner(
    client: OllamaClient = None,
    config: ModelConfig = None,
    storm_output_dir: str = None,
    storm_config: dict = None,
    retrieval_config: RetrievalConfig = None,
):
    """
    Setup STORM runner with optional configuration override.

    Args:
        client: Ollama client
        config: Model configuration
        storm_output_dir: Output directory for STORM
        storm_config: Optional Storm configuration parameters
        retrieval_config: Optional retrieval configuration (from runner)
    """
    lm_config = STORMWikiLMConfigs()

    lm_config.set_conv_simulator_lm(get_model_wrapper(client, config, "fast"))
    lm_config.set_question_asker_lm(get_model_wrapper(client, config, "fast"))
    lm_config.set_outline_gen_lm(get_model_wrapper(client, config, "outline"))
    lm_config.set_article_gen_lm(get_model_wrapper(client, config, "writing"))
    lm_config.set_article_polish_lm(get_model_wrapper(client, config, "polish"))

    # Use provided retrieval config (with CLI args) or create default with wiki RM
    if retrieval_config:
        used_retrieval_config = retrieval_config
    else:
        # Fallback: create config for wiki RM that respects base configuration
        used_retrieval_config = RetrievalConfig.from_yaml_with_overrides(
            rm_type="supabase_faiss"
        )

    # Default Storm configuration using retrieval config values
    default_config = {
        "max_conv_turn": 3,
        "max_perspective": 2,
        "max_search_queries_per_turn": used_retrieval_config.queries_per_turn,
        "search_top_k": used_retrieval_config.results_per_query,
        "max_thread_num": 2,
    }

    # Merge with provided config
    if storm_config:
        default_config.update(storm_config)

    # Use the provided retrieval config for STORM search
    search_rm = create_retrieval_manager(
        retrieval_config=used_retrieval_config, format_type="storm"
    )
    logging.getLogger("baselines.wikipedia_search").setLevel(logging.DEBUG)

    engine_args = STORMWikiRunnerArguments(
        output_dir=storm_output_dir,
        max_conv_turn=default_config["max_conv_turn"],
        max_perspective=default_config["max_perspective"],
        search_top_k=default_config["search_top_k"],
        max_thread_num=default_config["max_thread_num"],
    )

    runner = STORMWikiRunner(engine_args, lm_config, search_rm)
    return runner, storm_output_dir, default_config
