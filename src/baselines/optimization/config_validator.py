"""
Configuration Validator and Applier
Ensures optimized configurations are valid and provides easy application
"""

from pathlib import Path

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    warnings: list
    errors: list
    recommended_fixes: dict


class ConfigurationValidator:
    """Validates and applies optimized configurations."""

    def __init__(self):
        # Define valid parameter ranges
        self.storm_param_ranges = {
            "max_conv_turn": (1, 10),
            "max_perspective": (1, 20),
            "search_top_k": (1, 20),
            "max_thread_num": (1, 8),
        }

        self.rag_param_ranges = {
            "retrieval_k": (1, 20),
            "num_queries": (1, 20),
            "max_passages": (1, 30),
        }

        # Define performance recommendations
        self.storm_recommendations = {
            "max_conv_turn": {
                "min_recommended": 3,
                "optimal_range": (4, 6),
                "warning_threshold": 8,
            },
            "max_perspective": {
                "min_recommended": 3,
                "optimal_range": (4, 8),
                "warning_threshold": 12,
            },
            "search_top_k": {
                "min_recommended": 3,
                "optimal_range": (5, 10),
                "warning_threshold": 15,
            },
            "max_thread_num": {"optimal_range": (1, 2), "warning_threshold": 4},
        }

        self.rag_recommendations = {
            "retrieval_k": {
                "min_recommended": 3,
                "optimal_range": (5, 10),
                "warning_threshold": 15,
            },
            "num_queries": {
                "min_recommended": 3,
                "optimal_range": (5, 10),
                "warning_threshold": 15,
            },
            "max_passages": {
                "min_recommended": 5,
                "optimal_range": (8, 15),
                "warning_threshold": 25,
            },
        }

    def validate_storm_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate STORM configuration parameters."""
        warnings = []
        errors = []
        recommended_fixes = {}

        for param, value in config.items():
            if param not in self.storm_param_ranges:
                warnings.append(f"Unknown STORM parameter: {param}")
                continue

            # Check valid range
            min_val, max_val = self.storm_param_ranges[param]
            if not (min_val <= value <= max_val):
                errors.append(
                    f"{param}={value} outside valid range [{min_val}, {max_val}]"
                )
                recommended_fixes[param] = max(min_val, min(value, max_val))
                continue

            # Check recommendations
            if param in self.storm_recommendations:
                rec = self.storm_recommendations[param]

                if "min_recommended" in rec and value < rec["min_recommended"]:
                    warnings.append(
                        f"{param}={value} below recommended minimum {rec['min_recommended']}"
                    )

                if "optimal_range" in rec:
                    opt_min, opt_max = rec["optimal_range"]
                    if not (opt_min <= value <= opt_max):
                        warnings.append(
                            f"{param}={value} outside optimal range [{opt_min}, {opt_max}]"
                        )

                if "warning_threshold" in rec and value > rec["warning_threshold"]:
                    warnings.append(
                        f"{param}={value} above warning threshold {rec['warning_threshold']} (may impact performance)"
                    )

        # Check parameter interactions
        conv_turn = config.get("max_conv_turn", 2)
        perspective = config.get("max_perspective", 2)

        if conv_turn * perspective > 30:
            warnings.append(
                f"High complexity: {conv_turn} turns × {perspective} perspectives = {conv_turn * perspective} interactions (may be slow)"
            )

        # Threading warnings
        thread_num = config.get("max_thread_num", 1)
        if thread_num > 2 and perspective > 4:
            warnings.append(
                f"High threading ({thread_num}) + many perspectives ({perspective}) may cause resource contention"
            )

        is_valid = len(errors) == 0

        return ValidationResult(is_valid, warnings, errors, recommended_fixes)

    def validate_rag_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate RAG configuration parameters."""
        warnings = []
        errors = []
        recommended_fixes = {}

        for param, value in config.items():
            if param not in self.rag_param_ranges:
                warnings.append(f"Unknown RAG parameter: {param}")
                continue

            # Check valid range
            min_val, max_val = self.rag_param_ranges[param]
            if not (min_val <= value <= max_val):
                errors.append(
                    f"{param}={value} outside valid range [{min_val}, {max_val}]"
                )
                recommended_fixes[param] = max(min_val, min(value, max_val))
                continue

            # Check recommendations
            if param in self.rag_recommendations:
                rec = self.rag_recommendations[param]

                if "min_recommended" in rec and value < rec["min_recommended"]:
                    warnings.append(
                        f"{param}={value} below recommended minimum {rec['min_recommended']}"
                    )

                if "optimal_range" in rec:
                    opt_min, opt_max = rec["optimal_range"]
                    if not (opt_min <= value <= opt_max):
                        warnings.append(
                            f"{param}={value} outside optimal range [{opt_min}, {opt_max}]"
                        )

                if "warning_threshold" in rec and value > rec["warning_threshold"]:
                    warnings.append(
                        f"{param}={value} above warning threshold {rec['warning_threshold']} (may impact performance)"
                    )

        # Check parameter interactions
        retrieval_k = config.get("retrieval_k", 5)
        num_queries = config.get("num_queries", 5)
        max_passages = config.get("max_passages", 8)

        total_retrievals = retrieval_k * num_queries
        if total_retrievals > 100:
            warnings.append(
                f"High retrieval load: {retrieval_k} × {num_queries} = {total_retrievals} retrievals (may be slow)"
            )

        if max_passages > total_retrievals:
            warnings.append(
                f"max_passages ({max_passages}) > total retrievals ({total_retrievals}) - will be limited"
            )
            recommended_fixes["max_passages"] = total_retrievals

        is_valid = len(errors) == 0

        return ValidationResult(is_valid, warnings, errors, recommended_fixes)

    def apply_storm_config_to_file(
        self,
        config: Dict[str, Any],
        file_path: str = "src/baselines/configure_storm.py",
    ):
        """Apply STORM configuration to the configuration file."""
        config_path = Path(file_path)

        if not config_path.exists():
            logger.error(f"STORM config file not found: {config_path}")
            return False

        try:
            # Read current file
            with open(config_path, "r") as f:
                content = f.read()

            # Create backup
            backup_path = config_path.with_suffix(".py.backup")
            with open(backup_path, "w") as f:
                f.write(content)

            logger.info(f"Created backup: {backup_path}")

            # Replace STORMWikiRunnerArguments section
            new_args = f"""    engine_args = STORMWikiRunnerArguments(
        output_dir=storm_output_dir,
        max_conv_turn={config.get('max_conv_turn', 2)},
        max_perspective={config.get('max_perspective', 2)},
        search_top_k={config.get('search_top_k', 2)},
        max_thread_num={config.get('max_thread_num', 4)},
    )"""

            # Find and replace the engine_args section
            import re

            pattern = r"engine_args = STORMWikiRunnerArguments\([^)]+\)"

            if re.search(pattern, content, re.DOTALL):
                new_content = re.sub(pattern, new_args, content, flags=re.DOTALL)

                # Write updated content
                with open(config_path, "w") as f:
                    f.write(new_content)

                logger.info(f"✅ Applied STORM configuration to {config_path}")
                logger.info(f"📝 Configuration: {config}")
                return True
            else:
                logger.error(
                    "Could not find STORMWikiRunnerArguments section to replace"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to apply STORM configuration: {e}")
            return False

    def generate_rag_config_snippet(self, config: Dict[str, Any]) -> str:
        """Generate code snippet for RAG configuration."""
        return f"""
# Optimized RAG Configuration
# Add these parameters to your RAG implementation:

retrieval_system = WikipediaSearchRM(k={config.get('retrieval_k', 5)})

queries = generate_search_queries(
    client, model_config, topic,
    num_queries={config.get('num_queries', 5)}
)

context = create_context_from_passages(
    passages, max_passages={config.get('max_passages', 8)}
)
"""

    def validate_and_report(
        self, storm_config: Optional[Dict] = None, rag_config: Optional[Dict] = None
    ) -> bool:
        """Validate configurations and print detailed report."""

        all_valid = True

        if storm_config:
            logger.info("🌪️ Validating STORM Configuration...")
            storm_result = self.validate_storm_config(storm_config)

            if storm_result.is_valid:
                logger.info("✅ STORM configuration is valid")
            else:
                logger.error("❌ STORM configuration has errors")
                all_valid = False

            if storm_result.warnings:
                logger.warning("⚠️ STORM warnings:")
                for warning in storm_result.warnings:
                    logger.warning(f"  • {warning}")

            if storm_result.errors:
                logger.error("🚫 STORM errors:")
                for error in storm_result.errors:
                    logger.error(f"  • {error}")

                if storm_result.recommended_fixes:
                    logger.info("🔧 Recommended fixes:")
                    for param, fix in storm_result.recommended_fixes.items():
                        logger.info(f"  • {param}: {fix}")

        if rag_config:
            logger.info("\n🔍 Validating RAG Configuration...")
            rag_result = self.validate_rag_config(rag_config)

            if rag_result.is_valid:
                logger.info("✅ RAG configuration is valid")
            else:
                logger.error("❌ RAG configuration has errors")
                all_valid = False

            if rag_result.warnings:
                logger.warning("⚠️ RAG warnings:")
                for warning in rag_result.warnings:
                    logger.warning(f"  • {warning}")

            if rag_result.errors:
                logger.error("🚫 RAG errors:")
                for error in rag_result.errors:
                    logger.error(f"  • {error}")

                if rag_result.recommended_fixes:
                    logger.info("🔧 Recommended fixes:")
                    for param, fix in rag_result.recommended_fixes.items():
                        logger.info(f"  • {param}: {fix}")

        return all_valid


def load_optimization_results(results_dir: str) -> Optional[Dict]:
    """Load optimization results from saved state."""
    results_path = Path(results_dir) / "optimization_state.json"

    if not results_path.exists():
        logger.error(f"No optimization results found at {results_path}")
        return None

    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load optimization results: {e}")
        return None


def apply_optimization_results(results_dir: str):
    """Apply the best configurations from optimization results."""

    results = load_optimization_results(results_dir)
    if not results:
        return

    validator = ConfigurationValidator()

    # Extract best configurations
    best_storm_data = results.get("current_best_storm")
    best_rag_data = results.get("current_best_rag")

    if best_storm_data:
        storm_config = best_storm_data["parameters"]
        logger.info(f"🌪️ Best STORM config: {storm_config}")

        # Validate
        if validator.validate_storm_config(storm_config).is_valid:
            # Apply to file
            if validator.apply_storm_config_to_file(storm_config):
                logger.info("✅ STORM configuration applied successfully")
        else:
            logger.error("❌ STORM configuration validation failed")

    if best_rag_data:
        rag_config = best_rag_data["parameters"]
        logger.info(f"🔍 Best RAG config: {rag_config}")

        # Generate code snippet
        snippet = validator.generate_rag_config_snippet(rag_config)

        snippet_path = Path(results_dir) / "optimized_rag_config.py"
        with open(snippet_path, "w") as f:
            f.write(snippet)

        logger.info(f"📝 RAG configuration snippet saved to: {snippet_path}")


if __name__ == "__main__":
    # Example usage
    validator = ConfigurationValidator()

    # Test configurations
    storm_config = {
        "max_conv_turn": 5,
        "max_perspective": 6,
        "search_top_k": 7,
        "max_thread_num": 2,
    }

    rag_config = {"retrieval_k": 8, "num_queries": 7, "max_passages": 12}

    # Validate and report
    validator.validate_and_report(storm_config, rag_config)
