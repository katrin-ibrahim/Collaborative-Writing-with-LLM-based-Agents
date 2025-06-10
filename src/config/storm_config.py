import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider"""
    name: str
    model: str
    api_key_env: str
    api_base: Optional[str] = None
    timeout: int = 30
    max_tokens: int = 50
    temperature: float = 0.7
    priority: int = 1

@dataclass
class STORMConfig:
    """Configuration for STORM pipeline"""
    max_conv_turn: int = 2
    max_perspective: int = 2
    search_top_k: int = 3
    max_thread_num: int = 1
    enable_polish: bool = False
    max_retries: int = 2

class ConfigManager:
    """Manages configuration for STORM baselines"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/storm_config.yaml"
        self.providers = self._load_default_providers()
        self.storm_config = STORMConfig()
        
        # Load from file if it exists
        if Path(self.config_file).exists():
            self._load_from_file()
    
    def _load_default_providers(self) -> Dict[str, ProviderConfig]:
        """Load default provider configurations"""
        return {
            "together": ProviderConfig(
                name="together",
                model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                api_key_env="TOGETHER_API_KEY",
                api_base="https://api.together.xyz/v1",
                timeout=60,
                priority=1
            ),
            "groq": ProviderConfig(
                name="groq", 
                model="groq/llama-3.1-8b-instant",
                api_key_env="GROQ_API_KEY",
                api_base="https://api.groq.com/openai/v1",
                timeout=30,
                priority=2
            ),
            "huggingface": ProviderConfig(
                name="huggingface",
                model="microsoft/DialoGPT-medium", 
                api_key_env="HF_TOKEN",
                api_base="https://api-inference.huggingface.co/v1",
                timeout=90,
                priority=3
            ),
            "openai": ProviderConfig(
                name="openai",
                model="gpt-3.5-turbo",
                api_key_env="OPENAI_API_KEY", 
                api_base="https://api.openai.com/v1",
                timeout=30,
                priority=4
            )
        }
    
    def _load_from_file(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update provider configs
            if "providers" in config_data:
                for provider_name, provider_data in config_data["providers"].items():
                    if provider_name in self.providers:
                        # Update existing provider
                        for key, value in provider_data.items():
                            if hasattr(self.providers[provider_name], key):
                                setattr(self.providers[provider_name], key, value)
            
            # Update STORM config
            if "storm" in config_data:
                storm_data = config_data["storm"]
                for key, value in storm_data.items():
                    if hasattr(self.storm_config, key):
                        setattr(self.storm_config, key, value)
                        
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        # Ensure directory exists
        Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "providers": {
                name: {
                    "model": provider.model,
                    "timeout": provider.timeout,
                    "max_tokens": provider.max_tokens,
                    "temperature": provider.temperature,
                    "priority": provider.priority
                }
                for name, provider in self.providers.items()
            },
            "storm": {
                "max_conv_turn": self.storm_config.max_conv_turn,
                "max_perspective": self.storm_config.max_perspective,
                "search_top_k": self.storm_config.search_top_k,
                "max_thread_num": self.storm_config.max_thread_num,
                "enable_polish": self.storm_config.enable_polish,
                "max_retries": self.storm_config.max_retries
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def get_available_providers(self) -> Dict[str, ProviderConfig]:
        """Get providers that have API keys available"""
        available = {}
        for name, provider in self.providers.items():
            if os.getenv(provider.api_key_env):
                available[name] = provider
        return available
    
    def get_best_provider(self, preference: str = "auto") -> Optional[ProviderConfig]:
        """Get the best available provider based on preference"""
        available = self.get_available_providers()
        
        if not available:
            return None
        
        # If specific preference and available, use it
        if preference != "auto" and preference in available:
            return available[preference]
        
        # Otherwise, use highest priority available provider
        sorted_providers = sorted(available.values(), key=lambda p: p.priority)
        return sorted_providers[0] if sorted_providers else None
    
    def create_sample_config(self):
        """Create a sample configuration file"""
        sample_config = """# STORM Baseline Configuration
# Configure LLM providers and STORM pipeline settings

providers:
  together:
    model: "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    timeout: 60
    max_tokens: 50
    temperature: 0.7
    priority: 1
  
  groq:
    model: "groq/llama-3.1-8b-instant" 
    timeout: 30
    max_tokens: 50
    temperature: 0.7
    priority: 2
  
  huggingface:
    model: "microsoft/DialoGPT-medium"
    timeout: 90
    max_tokens: 50
    temperature: 0.7
    priority: 3

storm:
  max_conv_turn: 2        # Conversation turns (lower = faster)
  max_perspective: 2      # Perspectives to consider
  search_top_k: 3         # Search results per query
  max_thread_num: 1       # Threads (keep at 1 for stability)
  enable_polish: false    # Article polishing (slower)
  max_retries: 2          # Retry attempts per operation

# API keys are loaded from .env file
"""
        
        config_path = Path(self.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not config_path.exists():
            with open(config_path, 'w') as f:
                f.write(sample_config)
            print(f"Created sample configuration: {config_path}")
        else:
            print(f"Configuration file already exists: {config_path}")
    
    def check_environment_setup(self) -> Dict[str, str]:
        """Check which API keys are available"""
        available_keys = {}
        missing_keys = {}
        
        for name, provider in self.providers.items():
            key_value = os.getenv(provider.api_key_env)
            if key_value and key_value.strip():
                available_keys[name] = provider.api_key_env
            else:
                missing_keys[name] = provider.api_key_env
        
        return {
            "available": available_keys,
            "missing": missing_keys,
            "status": "ready" if available_keys else "needs_setup"
        }
        
        if not config_path.exists():
            with open(config_path, 'w') as f:
                f.write(sample_config)
            print(f"Created sample configuration: {config_path}")
        else:
            print(f"Configuration file already exists: {config_path}")

if __name__ == "__main__":
    # Demo the configuration system
    config_manager = ConfigManager()
    
    print("Creating sample configuration...")
    config_manager.create_sample_config()
    
    print("\nAvailable providers:")
    available = config_manager.get_available_providers()
    for name, provider in available.items():
        print(f"  {name}: {provider.model} (priority: {provider.priority})")
    
    if not available:
        print("  No providers available - check your API keys!")
        print("\nSet environment variables like:")
        print("export TOGETHER_API_KEY='your_key_here'")
        print("export GROQ_API_KEY='your_key_here'")
        print("export HF_TOKEN='your_key_here'")
    else:
        best = config_manager.get_best_provider()
        print(f"\nBest provider: {best.name} ({best.model})")