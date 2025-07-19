# FILE: local_baselines/runner_utils.py
import sys
from pathlib import Path

import logging

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config.baselines_model_config import ModelConfig
from utils.baselines_utils import enhance_content_prompt, validate_article_quality

logger = logging.getLogger(__name__)


def get_local_model_engine(model_path: str, config: ModelConfig, task: str):
    """Get LocalModelEngine for specific task (equivalent to get_model_wrapper for Ollama)."""
    from .model_engine import LocalModelEngine

    model = config.get_model_for_task(task)
    temp = config.get_temperature_for_task(task)
    max_tokens = config.get_token_limit_for_task(task)

    return LocalModelEngine(
        model_path=model_path, model=model, temperature=temp, max_tokens=max_tokens
    )


def enhance_article_content(
    model_engine, model_config: ModelConfig, content: str, topic: str
) -> str:
    """Enhance article content if it's too short (local-specific implementation)."""
    validation = validate_article_quality(content, min_words=800)

    if not validation["valid"]:
        logger.info(f"Content needs enhancement: {validation['reason']}")

        enhancement_prompt = enhance_content_prompt(topic, content)

        try:
            enhanced = model_engine.generate(
                enhancement_prompt, max_length=1024, temperature=0.2
            )
            return enhanced if enhanced else content
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")
            return content

    return content
