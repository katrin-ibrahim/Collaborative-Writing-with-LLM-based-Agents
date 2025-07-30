import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.baselines.ollama_baselines.llm_wrapper import OllamaLiteLLMWrapper
from src.config.baselines_model_config import ModelConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return {
        "models": {
            "conversation": "llama3.1:8b",
            "outline": "llama3.1:8b",
            "article": "llama3.1:8b",
            "polish": "llama3.1:8b",
        },
        "storm": {
            "max_conv_turn": 3,
            "max_perspective": 4,
            "search_top_k": 3,
            "max_thread_num": 1,
        },
        "rag": {"search_top_k": 5, "max_context_length": 4000, "temperature": 0.7},
    }


@pytest.fixture
def model_config_instance(sample_model_config):
    """ModelConfig instance for testing."""
    return ModelConfig.from_dict(sample_model_config)


@pytest.fixture
def mock_ollama_response():
    """Mock successful Ollama API response."""
    return {
        "choices": [
            {"message": {"content": "This is a test response from the mocked LLM."}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


@pytest.fixture
def mock_ollama_error_response():
    """Mock error response from Ollama API."""
    return {"error": {"message": "Model not found", "type": "model_error", "code": 404}}


@pytest.fixture
def sample_article_result():
    """Sample article result object for testing."""

    class MockArticleResult:
        def __init__(self):
            self.title = "Test Topic"
            self.content = "This is test article content."
            self.metadata = {
                "word_count": 50,
                "generation_time": 2.5,
                "method": "storm",
            }
            self.sections = [
                {"heading": "Introduction", "content": "Intro content"},
                {"heading": "Main Content", "content": "Main content"},
            ]

    return MockArticleResult()


@pytest.fixture
def sample_evaluation_metrics():
    """Sample evaluation metrics for testing."""
    return {
        "rouge_1": 0.75,
        "rouge_2": 0.65,
        "rouge_l": 0.70,
        "heading_soft_recall": 0.80,
        "heading_entity_recall": 0.85,
        "article_entity_recall": 0.78,
    }


@pytest.fixture
def sample_freshwiki_topics():
    """Sample FreshWiki topics for testing."""
    return [
        "Artificial Intelligence in Healthcare",
        "Climate Change Mitigation Strategies",
        "Quantum Computing Applications",
        "Sustainable Energy Technologies",
        "Machine Learning Ethics",
    ]


@pytest.fixture
def mock_experiment_results():
    """Mock experiment results structure."""
    return {
        "metadata": {
            "timestamp": "2025-01-15T10:30:00",
            "ollama_host": "http://localhost:11434/",
            "methods": ["direct", "storm", "rag"],
            "num_topics": 3,
            "models": {"conversation": "llama3.1:8b", "outline": "llama3.1:8b"},
            "total_time": 450.5,
            "topics_processed": 3,
        },
        "results": {
            "Test Topic 1": {
                "direct": {
                    "success": True,
                    "word_count": 250,
                    "evaluation": {
                        "rouge_1": 0.65,
                        "rouge_l": 0.60,
                        "heading_soft_recall": 0.70,
                        "heading_entity_recall": 0.75,
                        "article_entity_recall": 0.68,
                    },
                },
                "storm": {
                    "success": True,
                    "word_count": 450,
                    "evaluation": {
                        "rouge_1": 0.78,
                        "rouge_l": 0.73,
                        "heading_soft_recall": 0.85,
                        "heading_entity_recall": 0.88,
                        "article_entity_recall": 0.81,
                    },
                },
            }
        },
    }


@pytest.fixture
def mock_llm_wrapper():
    """Mock LLM wrapper for testing."""
    wrapper = Mock(spec=OllamaLiteLLMWrapper)
    wrapper.generate.return_value = "Mocked LLM response"
    wrapper.model = "llama3.1:8b"
    wrapper.host = "http://localhost:11434/"
    return wrapper


@pytest.fixture
def sample_storm_config():
    """Sample STORM configuration for testing."""
    return {
        "max_conv_turn": 3,
        "max_perspective": 4,
        "search_top_k": 3,
        "max_thread_num": 1,
        "temperature": 0.7,
    }


@pytest.fixture
def sample_rag_config():
    """Sample RAG configuration for testing."""
    return {
        "search_top_k": 5,
        "max_context_length": 4000,
        "temperature": 0.7,
        "retrieval_method": "wikipedia",
    }


@pytest.fixture
def mock_wikipedia_search_results():
    """Mock Wikipedia search results."""
    return [
        {
            "title": "Artificial Intelligence",
            "snippet": "AI is the simulation of human intelligence...",
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        },
        {
            "title": "Machine Learning",
            "snippet": "ML is a subset of artificial intelligence...",
            "url": "https://en.wikipedia.org/wiki/Machine_learning",
        },
        {
            "title": "Deep Learning",
            "snippet": "Deep learning is part of machine learning...",
            "url": "https://en.wikipedia.org/wiki/Deep_learning",
        },
    ]


@pytest.fixture
def invalid_json_content():
    """Invalid JSON content for error testing."""
    return '{"invalid": json, "missing": quotes}'


@pytest.fixture
def nested_object_for_serialization():
    """Complex nested object for serialization testing."""

    class TestObject:
        def __init__(self):
            self.name = "test"
            self.value = 42
            self.nested = {"list": [1, 2, 3], "dict": {"key": "value"}}

    return {
        "simple": "string",
        "number": 123,
        "boolean": True,
        "list": [1, 2, TestObject()],
        "object": TestObject(),
        "none_value": None,
    }


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging

    logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def mock_optimization_state():
    """Mock optimization state for testing."""
    from src.baselines.ollama_baselines.optimization.adaptive_optimizer import (
        OptimizationConfig,
        OptimizationState,
    )

    state = OptimizationState()
    state.current_best_storm = OptimizationConfig(
        method="storm",
        parameters={"max_conv_turn": 3, "max_perspective": 4},
        composite_score=0.82,
        individual_scores={"rouge_1": 0.78, "heading_soft_recall": 0.85},
    )
    return state
