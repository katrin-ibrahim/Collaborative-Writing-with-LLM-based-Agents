"""
Local STORM runner using local models (same as direct prompting).
Adapted from baselines STORM implementation to work with LocalModelWrapper.
"""

import sys
import time
from pathlib import Path

import logging
from typing import Optional

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from knowledge_storm import (
    STORMWikiLMConfigs,
    STORMWikiRunner,
    STORMWikiRunnerArguments,
)

from local.runner import LocalModelWrapper
from utils.data_models import Article
from utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


class LocalLiteLLMWrapper:
    """LiteLLM-compatible wrapper for local models to work with STORM."""

    def __init__(
        self,
        local_model_wrapper: LocalModelWrapper,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.local_wrapper = local_model_wrapper
        self.temperature = temperature
        self.max_tokens = max_tokens

        # LiteLLM compatibility attributes
        self.model_name = str(local_model_wrapper.model_path)
        self.kwargs = {
            "model": self.model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Cost tracking (always 0 for local models)
        self.completion_cost = 0.0
        self.prompt_cost = 0.0

        logger.info(f"LocalLiteLLMWrapper initialized with model: {self.model_name}")

    def __call__(self, messages=None, **kwargs):
        """Make wrapper callable for STORM compatibility."""
        if "max_tokens" not in kwargs and "max_output_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens
        if messages is not None:
            return self.complete(messages, **kwargs)

        # Handle string prompts
        if isinstance(kwargs.get("prompt"), str):
            return self._generate(kwargs["prompt"], **kwargs)

        # Default case
        return self.complete(str(kwargs), **kwargs)

    def complete(self, messages, **kwargs):
        """Complete a conversation with messages format."""
        try:
            # Convert messages to a single prompt
            if isinstance(messages, list):
                prompt = self._messages_to_prompt(messages)
            else:
                prompt = str(messages)

            # Extract parameters
            max_tokens = kwargs.get(
                "max_tokens", kwargs.get("max_output_tokens", self.max_tokens)
            )
            temperature = kwargs.get("temperature", self.temperature)

            logger.debug(
                f"Generating with local model: {len(prompt)} chars, max_tokens={max_tokens}"
            )

            # Generate using local model
            response = self.local_wrapper.generate(
                prompt=prompt, max_length=max_tokens, temperature=temperature
            )

            # Return in LiteLLM format
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Message", (), {"content": response})()},
                        )()
                    ],
                    "usage": type(
                        "Usage",
                        (),
                        {
                            "prompt_tokens": len(prompt.split()),
                            "completion_tokens": len(response.split()),
                            "total_tokens": len(prompt.split()) + len(response.split()),
                        },
                    )(),
                },
            )()

        except Exception as e:
            logger.error(f"Local model generation failed: {e}")
            raise

    def _generate(self, prompt: str, **kwargs):
        """Generate from a string prompt."""
        # Convert string prompt to messages format
        messages = [{"role": "user", "content": prompt}]
        return self.complete(messages, **kwargs)

    def _messages_to_prompt(self, messages: list) -> str:
        """Convert messages format to a single prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)

        return "\n\n".join(prompt_parts)


class LocalSTORMRunner:
    """Local STORM runner using local models."""

    def __init__(
        self,
        model_path: str = "models/",
        output_manager: Optional[OutputManager] = None,
        device: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.output_manager = output_manager
        self.device = device

        # Initialize local model wrapper (reuse from direct prompting)
        from local.runner import LocalBaselineRunner

        self.local_runner = LocalBaselineRunner(model_path, output_manager, device)

        logger.info(f"LocalSTORMRunner initialized with model: {self.model_path}")

    def setup_storm_runner(self, output_dir: str, storm_config: dict = None):
        """Setup STORM runner with local models."""
        lm_config = STORMWikiLMConfigs()

        # Create different model wrappers for different tasks
        # Use varying temperatures for different STORM components
        conv_wrapper = LocalLiteLLMWrapper(
            self.local_runner.model_wrapper,
            temperature=0.8,  # Higher temp for conversation simulation
            max_tokens=512,
        )
        question_wrapper = LocalLiteLLMWrapper(
            self.local_runner.model_wrapper,
            temperature=0.7,  # Medium temp for question asking
            max_tokens=256,
        )
        outline_wrapper = LocalLiteLLMWrapper(
            self.local_runner.model_wrapper,
            temperature=0.5,  # Lower temp for outline generation
            max_tokens=1024,
        )
        writing_wrapper = LocalLiteLLMWrapper(
            self.local_runner.model_wrapper,
            temperature=0.3,  # Low temp for article writing
            max_tokens=1024,
        )
        polish_wrapper = LocalLiteLLMWrapper(
            self.local_runner.model_wrapper,
            temperature=0.2,  # Very low temp for polishing
            max_tokens=1024,
        )

        # Configure STORM with local model wrappers
        lm_config.set_conv_simulator_lm(conv_wrapper)
        lm_config.set_question_asker_lm(question_wrapper)
        lm_config.set_outline_gen_lm(outline_wrapper)
        lm_config.set_article_gen_lm(writing_wrapper)
        lm_config.set_article_polish_lm(polish_wrapper)

        # Default STORM configuration (optimized for local testing)
        default_config = {
            "max_conv_turn": 2,  # Reduced for speed (was 4)
            "max_perspective": 2,  # Reduced for speed (was 4)
            "search_top_k": 3,  # Reduced for speed (was 5)
            "max_thread_num": 1,  # Single thread for stability
        }

        # Merge with provided config
        if storm_config:
            default_config.update(storm_config)

        # Setup simple search retrieval (using Wikipedia API)
        from baselines.wikipedia_rm import WikipediaSearchRM

        search_rm = WikipediaSearchRM(k=default_config["search_top_k"])

        engine_args = STORMWikiRunnerArguments(
            output_dir=output_dir,
            max_conv_turn=default_config["max_conv_turn"],
            max_perspective=default_config["max_perspective"],
            search_top_k=default_config["search_top_k"],
            max_thread_num=default_config["max_thread_num"],
        )

        runner = STORMWikiRunner(engine_args, lm_config, search_rm)
        return runner, default_config

    def run_storm(self, topic: str, storm_config: dict = None) -> Article:
        """Run STORM for a single topic."""
        logger.info(f"Running Local STORM for: {topic}")

        try:
            # Setup output directory
            if self.output_manager:
                storm_output_dir = self.output_manager.setup_storm_output_dir(topic)
            else:
                storm_output_dir = f"temp_storm_{topic.replace(' ', '_')}"
                Path(storm_output_dir).mkdir(parents=True, exist_ok=True)

            # Setup STORM runner
            runner, config = self.setup_storm_runner(storm_output_dir, storm_config)

            start_time = time.time()

            # Run STORM pipeline
            logger.info(f"Starting STORM pipeline for {topic}...")
            runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True,
            )

            generation_time = time.time() - start_time

            # Extract content from STORM output
            content = self._extract_storm_output(topic, storm_output_dir)

            # Create article with metadata
            metadata = {
                "method": "storm",
                "model": str(self.model_path),
                "word_count": len(content.split()) if content else 0,
                "generation_time": generation_time,
                "storm_config": config,
                "local_model": True,
            }

            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata=metadata,
            )

            # Save article
            if self.output_manager:
                self.output_manager.save_article(article, "storm")

            logger.info(
                f"Local STORM completed for {topic} ({metadata['word_count']} words)"
            )
            return article

        except Exception as e:
            logger.error(f"Local STORM failed for '{topic}': {e}")
            from shared.prompt_utils import error_article

            return error_article(topic, str(e), "storm")

    def _extract_storm_output(self, topic: str, output_dir: str) -> str:
        """Extract the final article from STORM output directory."""
        output_path = Path(output_dir)

        # Look for the polished article
        polished_file = output_path / "storm_gen_article_polished.txt"
        if polished_file.exists():
            with open(polished_file, "r", encoding="utf-8") as f:
                return f.read()

        # Fallback to unpolished article
        article_file = output_path / "storm_gen_article.txt"
        if article_file.exists():
            with open(article_file, "r", encoding="utf-8") as f:
                return f.read()

        # Fallback to outline if article generation failed
        outline_file = output_path / "storm_gen_outline.txt"
        if outline_file.exists():
            with open(outline_file, "r", encoding="utf-8") as f:
                outline = f.read()
                return f"# {topic}\n\nOutline:\n{outline}\n\n(Note: Full article generation failed)"

        return f"# {topic}\n\nError: No STORM output found"
