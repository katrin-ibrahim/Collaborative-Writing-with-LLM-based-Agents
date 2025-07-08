import logging
from knowledge_storm import (
    STORMWikiLMConfigs,
    STORMWikiRunner,
    STORMWikiRunnerArguments,
)

from baselines.runner_utils import get_model_wrapper
from baselines.wikipedia_rm import WikipediaSearchRM
from config.model_config import ModelConfig
from utils.ollama_client import OllamaClient


def setup_storm_runner(
    client: OllamaClient, config: ModelConfig, storm_output_dir: str
):
    lm_config = STORMWikiLMConfigs()

    lm_config.set_conv_simulator_lm(get_model_wrapper(client, config, "fast"))
    lm_config.set_question_asker_lm(get_model_wrapper(client, config, "fast"))
    lm_config.set_outline_gen_lm(get_model_wrapper(client, config, "outline"))
    lm_config.set_article_gen_lm(get_model_wrapper(client, config, "writing"))
    lm_config.set_article_polish_lm(get_model_wrapper(client, config, "polish"))

    # search_rm = MockSearchRM(k=3)
    search_rm = WikipediaSearchRM(k=3)
    logging.getLogger("baselines.wikipedia_search").setLevel(logging.DEBUG)

    engine_args = STORMWikiRunnerArguments(
        output_dir=storm_output_dir,
        max_conv_turn=2,
        max_perspective=2,
        search_top_k=2,
        max_thread_num=4,
    )

    runner = STORMWikiRunner(engine_args, lm_config, search_rm)
    return runner, storm_output_dir
