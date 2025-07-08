import os
import yaml
from dataclasses import dataclass


@dataclass
class DefaultConfig:
    # Model settings
    model_type: str = "local"  # "local" or "api"
    local_model_path: str = (
        "/storage/ukp/shared/shared_model_weights/Qwen2.5-VL-7B-Instruct"
    )

    # LLM settings (for API fallback)
    provider: str = "groq"
    model: str = "llama-3.1-8b-instant"

    # STORM settings
    max_conv_turn: int = 2
    max_perspective: int = 2
    search_top_k: int = 3
    max_thread_num: int = 1
    enable_polish: bool = False
    max_retries: int = 2

    # Local model settings
    max_new_tokens: int = 512
    temperature: float = 0.7


def load_config(config_file: str = "config.yaml") -> DefaultConfig:
    """Load config from YAML file, use defaults if file doesn't exist"""

    config = DefaultConfig()

    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                data = yaml.safe_load(f) or {}

            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        except Exception as e:
            print(f"Warning: Could not load {config_file}: {e}")
            print("Using default configuration")

    return config


def get_api_key(provider: str) -> str:
    """Get API key for provider from environment (fallback only)"""
    key_map = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "together_ai": "TOGETHER_API_KEY",
        "huggingface": "HF_TOKEN",
    }

    env_key = key_map.get(provider)
    if not env_key:
        raise ValueError(f"Unknown provider: {provider}")

    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError(f"Missing {env_key} environment variable")

    return api_key
