# FILE: runners/runner_utils.py
import sys
from pathlib import Path

import logging

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.config.baselines_model_config import ModelConfig
from src.utils.data_models import Article
from src.utils.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def get_model_wrapper(
    client: OllamaClient, config: ModelConfig, task: str
) -> OllamaClient:
    """Get configured OllamaClient for specific task."""
    model = config.get_model_for_task(task)
    temp = config.get_temperature_for_task(task)
    max_tokens = config.get_token_limit_for_task(task)

    # Create a new instance configured for this specific task
    return OllamaClient(
        host=client.host,
        model=model,
        temperature=temp,
        max_tokens=max_tokens,
    )


def log_result(method: str, article: Article):
    if "error" in article.metadata:
        logger.warning(f"✗ {method} failed")
    else:
        wc = article.metadata.get("word_count", 0)
        logger.info(f"✓ {method} completed ({wc} words)")
