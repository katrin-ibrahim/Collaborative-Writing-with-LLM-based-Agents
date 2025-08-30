from abc import ABC, abstractmethod

import os
import yaml
from typing import Any, Dict, Optional, TypeVar

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(ABC):
    """Abstract base class for all configuration classes."""

    @abstractmethod
    def get_file_pattern(self) -> str:
        """Return YAML file pattern (e.g., 'collaboration_{}.yaml')."""

    @classmethod
    def from_dict(cls: type[T], config_dict: Dict[str, Any]) -> T:
        """Create instance from dictionary with field filtering."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    @classmethod
    def get_default(cls: type[T]) -> T:
        """Get default configuration - works for all dataclasses."""
        return cls()

    @classmethod
    def from_yaml_with_overrides(
        cls: type[T], config_name: Optional[str] = None, **overrides
    ) -> T:
        """Load from YAML with overrides using the abstract file pattern."""
        # Filter out None overrides
        actual_overrides = {k: v for k, v in overrides.items() if v is not None}

        if config_name:
            # Use the abstract method to get file pattern
            temp_instance = cls()  # Need instance to call get_file_pattern
            file_pattern = temp_instance.get_file_pattern()

            config_dir = os.path.dirname(__file__)
            config_path = os.path.join(config_dir, file_pattern.format(config_name))

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, "r") as f:
                yaml_data = yaml.safe_load(f)

            # Merge with overrides
            yaml_data.update(actual_overrides)
            return cls.from_dict(yaml_data)
        else:
            # No YAML, just create with overrides
            return cls(**actual_overrides)
