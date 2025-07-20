import sys
from pathlib import Path
import tempfile
import os
import logging

def setup_cache_directories():
    import sys
    import builtins
    
    # Setup temp dirs first
    temp_dir = tempfile.gettempdir()
    pid = os.getpid()
    joblib_cache = os.path.join(temp_dir, f"joblib_cache_{pid}")
    os.makedirs(joblib_cache, exist_ok=True)
    
    original_import = builtins.__import__
    
    def patched_import(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        
        # Patch the main litellm module when it's imported
        if name == 'litellm':
            # Set cache to a no-op object that doesn't do anything
            class NoOpCache:
                def __init__(self, *args, **kwargs):
                    pass
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
                def __setattr__(self, name, value):
                    pass
            
            # Set cache to disabled state immediately
            module.cache = NoOpCache()
            
            # Also override the cache attribute to prevent future assignments
            class CacheProperty:
                def __set__(self, obj, value):
                    pass  # Ignore future cache assignments
                def __get__(self, obj, objtype=None):
                    return module.cache  # Return our no-op cache
            
            type(module).cache = CacheProperty()
            logging.info("Set litellm.cache to disabled no-op object")
        
        return module
    
    builtins.__import__ = patched_import

setup_cache_directories()
# Set up cache before any imports
from knowledge_storm import (
    STORMWikiLMConfigs,
    STORMWikiRunner,
    STORMWikiRunnerArguments,
)

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.config.baselines_model_config import ModelConfig
from src.utils.ollama_client import OllamaClient

from .runner_utils import get_model_wrapper
from .wikipedia_rm import WikipediaSearchRM


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
        "search_top_k": 5,
        "max_thread_num": 4,
    }

    # Merge with provided config
    if storm_config:
        default_config.update(storm_config)

    # Setup search retrieval with configured parameters
    search_rm = WikipediaSearchRM(k=default_config["search_top_k"])
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
