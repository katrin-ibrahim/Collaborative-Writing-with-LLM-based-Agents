# src/methods/base_method.py
"""
Abstract base class for all methods in the AI Writer Agent Framework.
"""

from abc import ABC, abstractmethod

import logging
from typing import Any, Dict

from src.config.config_context import ConfigContext
from src.utils.data import Article

logger = logging.getLogger(__name__)


class BaseMethod(ABC):
    """
    Abstract base class for all writing methods.

    Methods include: direct, rag, storm, writer, writer_reviewer, etc.
    Each method implements its own approach to generating articles.
    """

    def __init__(self):
        """Initialize base method with token tracking."""
        super().__init__()

    @abstractmethod
    def run(self, topic: str) -> Article:
        """
        Run the method on a single topic.

        Args:
            topic: The topic to generate an article about

        Returns:
            Generated article
        """

    def get_method_name(self) -> str:
        """Get the method name for logging and metadata."""
        return self.__class__.__name__.replace("Method", "").lower()

    def _collect_token_usage(self, task_models: Dict[str, str]) -> Dict[str, Any]:
        """
        Collect token usage from all clients used by this method.

        Args:
            task_models: Dictionary mapping task names to model names used

        Returns:
            Dictionary containing token usage statistics
        """
        token_usage = {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_calls": 0,
            "by_task": {},
            "by_model": {},
        }

        # Collect usage from all tasks/models used
        for task, model in task_models.items():
            try:
                client = ConfigContext.get_client(task)
                if hasattr(client, "get_usage_stats"):
                    usage_stats = client.get_usage_stats()
                    total_usage = usage_stats.get("total_usage", {})

                    if total_usage.get("calls", 0) > 0:
                        # Add to totals
                        token_usage["total_tokens"] += total_usage.get(
                            "total_tokens", 0
                        )
                        token_usage["total_prompt_tokens"] += total_usage.get(
                            "prompt_tokens", 0
                        )
                        token_usage["total_completion_tokens"] += total_usage.get(
                            "completion_tokens", 0
                        )
                        token_usage["total_calls"] += total_usage.get("calls", 0)

                        # Track by task
                        token_usage["by_task"][task] = {
                            "model": model,
                            "tokens": total_usage.get("total_tokens", 0),
                            "prompt_tokens": total_usage.get("prompt_tokens", 0),
                            "completion_tokens": total_usage.get(
                                "completion_tokens", 0
                            ),
                            "calls": total_usage.get("calls", 0),
                        }

                        # Track by model
                        if model not in token_usage["by_model"]:
                            token_usage["by_model"][model] = {
                                "tokens": 0,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "calls": 0,
                                "tasks": [],
                            }

                        model_stats = token_usage["by_model"][model]
                        model_stats["tokens"] += total_usage.get("total_tokens", 0)
                        model_stats["prompt_tokens"] += total_usage.get(
                            "prompt_tokens", 0
                        )
                        model_stats["completion_tokens"] += total_usage.get(
                            "completion_tokens", 0
                        )
                        model_stats["calls"] += total_usage.get("calls", 0)
                        model_stats["tasks"].append(task)

            except Exception as e:
                logger.warning(f"Could not collect token usage for task {task}: {e}")

        return token_usage

    def _reset_all_client_usage(self, task_models: Dict[str, str]):
        """Reset token usage counters for all clients used by this method."""
        for task in task_models.keys():
            try:
                client = ConfigContext.get_client(task)
                if hasattr(client, "reset_usage_stats"):
                    client.reset_usage_stats()
            except Exception as e:
                logger.warning(f"Could not reset token usage for task {task}: {e}")

    def _get_task_models_for_method(self) -> Dict[str, str]:
        """
        Get mapping of tasks to models for this specific method.
        Override in subclasses to specify which tasks are used.
        """
        method_name = self.get_method_name()

        # Default mappings based on method type
        if method_name in ["direct", "rag"]:
            return {
                "writing": ConfigContext.get_model_config().get_model_for_task(
                    "writing"
                )
            }
        elif method_name == "storm":
            # Storm uses multiple models
            return {
                "conv_simulator": ConfigContext.get_model_config().get_model_for_task(
                    "conv_simulator"
                ),
                "outline": ConfigContext.get_model_config().get_model_for_task(
                    "outline"
                ),
                "writing": ConfigContext.get_model_config().get_model_for_task(
                    "writing"
                ),
                "polish": ConfigContext.get_model_config().get_model_for_task("polish"),
            }
        elif "writer" in method_name:
            # Writer-Reviewer methods
            return {
                "research": ConfigContext.get_model_config().get_model_for_task(
                    "research"
                ),
                "create_outline": ConfigContext.get_model_config().get_model_for_task(
                    "create_outline"
                ),
                "writer": ConfigContext.get_model_config().get_model_for_task("writer"),
                "revision": ConfigContext.get_model_config().get_model_for_task(
                    "revision"
                ),
                "self_refine": ConfigContext.get_model_config().get_model_for_task(
                    "self_refine"
                ),
                "reviewer": ConfigContext.get_model_config().get_model_for_task(
                    "reviewer"
                ),
            }
        else:
            # Fallback to writing model
            return {
                "writing": ConfigContext.get_model_config().get_model_for_task(
                    "writing"
                )
            }
