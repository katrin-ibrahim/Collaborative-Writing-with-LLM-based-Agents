# FILE: runners/ollama_runner.py
import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.ollama_client import OllamaClient
from config.model_config import ModelConfig
from utils.data_models import Article
from baselines.mock_search import MockSearchRM
from baselines.configure_storm import setup_storm_runner
from baselines.runner_utils import (
    get_model_wrapper,
    build_direct_prompt,
    extract_storm_output,
    error_article,
    log_result
)

logger = logging.getLogger(__name__)


class BaselineRunner:
    def __init__(self, ollama_host: str = "http://10.167.31.201:11434/", 
                 model_config: Optional[ModelConfig] = None):
        self.ollama_host = ollama_host
        self.model_config = model_config or ModelConfig()
        self.client = OllamaClient(host=ollama_host)

        if not self.client.is_available():
            raise RuntimeError(f"Ollama server not available at {ollama_host}")

        available_models = self.client.list_models()
        logger.info(f"Connected to Ollama with {len(available_models)} models available")

        self.work_dir = os.getcwd()
        self.output_dir = os.path.join(self.work_dir, "ollama_output")
        os.makedirs(self.output_dir, exist_ok=True)

# --------------------- Direct Prompting Baseline ---------------------
    def run_direct_prompting(self, topic: str) -> Article:
        logger.info(f"Running Direct Prompting for: {topic}")
        prompt = build_direct_prompt(topic)

        try:
            start_time = time.time()
            wrapper = get_model_wrapper(self.client, self.model_config, "writing")

            content = wrapper.generate(prompt=prompt, max_tokens=2048)
            content_words = len(content.split()) if content else 0
            generation_time = time.time() - start_time

            if content and not content.startswith("#"):
                content = f"# {topic}\n\n{content}"

            return Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "direct",
                    "model": wrapper.model,
                    "word_count": content_words,
                    "generation_time": generation_time,
                    "temperature": wrapper.temperature
                }
            )
        except Exception as e:
            logger.error(f"Direct Prompting failed: {e}")
            return error_article(topic, e, "direct")

    def run_direct_batch(self, topics: List[str], max_workers: int = 3) -> List[Article]:
        logger.info(f"Running Direct Prompting batch for {len(topics)} topics")
        results = []

        def run_topic(topic):
            return self.run_direct_prompting(topic)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_topic, topic): topic for topic in topics}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    topic = futures[future]
                    logger.error(f"Direct batch failed for {topic}: {e}")
                    results.append(error_article(topic, e, "direct_batch"))

        return results

    # --------------------- STORM Baseline ---------------------
    def run_storm(self, topic: str) -> Article:
        logger.info(f"Running STORM for: {topic}")

        try:
            runner, storm_output_dir = setup_storm_runner(self.client, self.model_config, self.output_dir)

            start_time = time.time()
            runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True
            )
            generation_time = time.time() - start_time
            content = extract_storm_output(topic, storm_output_dir)

            return Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm",
                    "word_count": len(content.split()) if content else 0,
                    "generation_time": generation_time,
                    "model": self.model_config.get_model_for_task("writing"),
                }
            )

        except Exception as e:
            logger.error(f"STORM failed: {e}")
            return error_article(topic, e, "storm")

    def run_storm_batch(self, topics: List[str], max_workers: int = 3) -> List[Article]:
        logger.info(f"Running STORM batch for {len(topics)} topics")
        results = []

        def run_topic(topic):
            return self.run_storm(topic)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_topic, topic): topic for topic in topics}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    topic = futures[future]
                    logger.error(f"STORM batch failed for {topic}: {e}")
                    results.append(error_article(topic, e, "storm_batch"))

        return results
