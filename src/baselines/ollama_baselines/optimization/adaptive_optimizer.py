import time
from pathlib import Path

import json
import logging
import numpy as np
from dataclasses import asdict, dataclass
from src.ollama_baselines.runner import BaselineRunner
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.evaluator import ArticleEvaluator
from src.utils.freshwiki_loader import FreshWikiLoader
from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for a single optimization run."""

    method: str
    parameters: Dict[str, Any]
    composite_score: float = 0.0
    individual_scores: Dict[str, float] = None
    test_results: List[Dict] = None
    generation_time: float = 0.0

    def __post_init__(self):
        if self.individual_scores is None:
            self.individual_scores = {}
        if self.test_results is None:
            self.test_results = []


@dataclass
class OptimizationState:
    """Tracks the current state of optimization."""

    current_best_storm: Optional[OptimizationConfig] = None
    current_best_rag: Optional[OptimizationConfig] = None
    baseline_storm: Optional[OptimizationConfig] = None
    baseline_rag: Optional[OptimizationConfig] = None
    tested_configurations: List[OptimizationConfig] = None
    total_tests_run: int = 0
    optimization_start_time: float = 0.0

    def __post_init__(self):
        if self.tested_configurations is None:
            self.tested_configurations = []


class PerformanceTracker:
    """Tracks performance and handles rollback logic."""

    def __init__(self, output_dir: Path, resume_state_path: Optional[Path] = None):
        self.output_dir = Path(output_dir)

        # If a specific resume state path is provided, use it to load but save to the output_dir
        if resume_state_path:
            self.resume_state_path = resume_state_path
            self.state_file = self.output_dir / "optimization_state.json"
            self.state = self.load_state(self.resume_state_path)
            logger.info(f"Loaded state from: {resume_state_path}")
            # Save a copy of the loaded state to the new output directory
            if self.resume_state_path != self.state_file and self.state:
                self.save_state()
                logger.info(f"Copied state to new location: {self.state_file}")
        else:
            # Normal operation - load and save from the same directory
            self.state_file = self.output_dir / "optimization_state.json"
            self.state = (
                self.load_state() if self.state_file.exists() else OptimizationState()
            )

    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted composite score from STORM metrics."""
        rouge_1 = metrics.get("rouge_1", 0.0)
        heading_soft_recall = metrics.get("heading_soft_recall", 0.0)
        article_entity_recall = metrics.get("article_entity_recall", 0.0)

        # Weighted composite: emphasize content quality and topic coverage
        composite = (
            0.4 * rouge_1 + 0.3 * heading_soft_recall + 0.3 * article_entity_recall
        )

        logger.debug(
            f"Composite calculation: "
            f"ROUGE-1({rouge_1:.2f}%) * 0.4 + "
            f"HSR({heading_soft_recall:.2f}%) * 0.3 + "
            f"AER({article_entity_recall:.2f}%) * 0.3 = {composite:.2f}"
        )

        return composite

    def is_improvement(
        self, new_config: OptimizationConfig, current_best: Optional[OptimizationConfig]
    ) -> bool:
        """Check if new configuration is an improvement."""
        if current_best is None:
            return True

        improvement = new_config.composite_score > current_best.composite_score
        logger.info(
            f"Performance comparison: "
            f"New({new_config.composite_score:.3f}) vs "
            f"Current({current_best.composite_score:.3f}) - "
            f"{'IMPROVEMENT' if improvement else 'NO IMPROVEMENT'}"
        )

        return improvement

    def update_best_configuration(self, config: OptimizationConfig):
        """Update best configuration if this is an improvement."""
        if config.method == "storm":
            if self.is_improvement(config, self.state.current_best_storm):
                logger.info(f"🚀 NEW BEST STORM CONFIG: {config.composite_score:.3f}")
                self.state.current_best_storm = config
                return True
        elif config.method == "rag":
            if self.is_improvement(config, self.state.current_best_rag):
                logger.info(f"🚀 NEW BEST RAG CONFIG: {config.composite_score:.3f}")
                self.state.current_best_rag = config
                return True

        return False

    def save_state(self):
        """Save optimization state to disk."""
        try:
            state_data = {
                "current_best_storm": (
                    asdict(self.state.current_best_storm)
                    if self.state.current_best_storm
                    else None
                ),
                "current_best_rag": (
                    asdict(self.state.current_best_rag)
                    if self.state.current_best_rag
                    else None
                ),
                "baseline_storm": (
                    asdict(self.state.baseline_storm)
                    if self.state.baseline_storm
                    else None
                ),
                "baseline_rag": (
                    asdict(self.state.baseline_rag) if self.state.baseline_rag else None
                ),
                "tested_configurations": [
                    asdict(config) for config in self.state.tested_configurations
                ],
                "total_tests_run": self.state.total_tests_run,
                "optimization_start_time": self.state.optimization_start_time,
            }

            with open(self.state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            logger.info(f"Optimization state saved to {self.state_file}")

        except Exception as e:
            logger.warning(f"Failed to save optimization state: {e}")

    def load_state(self, state_path: Optional[Path] = None) -> OptimizationState:
        """Load optimization state from disk if it exists."""
        try:
            # Use provided path or default to self.state_file
            file_path = state_path if state_path else self.state_file

            if not file_path.exists():
                logger.info(
                    f"No previous optimization state found at {file_path}. Starting fresh."
                )
                return OptimizationState()

            logger.info(f"Loading optimization state from: {file_path}")
            with open(file_path, "r") as f:
                state_data = json.load(f)

            state = OptimizationState(
                total_tests_run=state_data.get("total_tests_run", 0),
                optimization_start_time=state_data.get(
                    "optimization_start_time", time.time()
                ),
            )

            # Restore best configurations if available
            if state_data.get("current_best_storm"):
                storm_data = state_data["current_best_storm"]
                state.current_best_storm = OptimizationConfig(
                    method="storm",
                    parameters=storm_data["parameters"],
                    composite_score=storm_data["composite_score"],
                    individual_scores=storm_data["individual_scores"],
                    test_results=storm_data["test_results"],
                    generation_time=storm_data["generation_time"],
                )

            if state_data.get("current_best_rag"):
                rag_data = state_data["current_best_rag"]
                state.current_best_rag = OptimizationConfig(
                    method="rag",
                    parameters=rag_data["parameters"],
                    composite_score=rag_data["composite_score"],
                    individual_scores=rag_data["individual_scores"],
                    test_results=rag_data["test_results"],
                    generation_time=rag_data["generation_time"],
                )

            if state_data.get("baseline_storm"):
                storm_data = state_data["baseline_storm"]
                state.baseline_storm = OptimizationConfig(
                    method="storm",
                    parameters=storm_data["parameters"],
                    composite_score=storm_data["composite_score"],
                    individual_scores=storm_data["individual_scores"],
                    test_results=storm_data["test_results"],
                    generation_time=storm_data["generation_time"],
                )

            if state_data.get("baseline_rag"):
                rag_data = state_data["baseline_rag"]
                state.baseline_rag = OptimizationConfig(
                    method="rag",
                    parameters=rag_data["parameters"],
                    composite_score=rag_data["composite_score"],
                    individual_scores=rag_data["individual_scores"],
                    test_results=rag_data["test_results"],
                    generation_time=rag_data["generation_time"],
                )

            # Restore tested configurations
            if state_data.get("tested_configurations"):
                for config_data in state_data["tested_configurations"]:
                    config = OptimizationConfig(
                        method=config_data["method"],
                        parameters=config_data["parameters"],
                        composite_score=config_data["composite_score"],
                        individual_scores=config_data["individual_scores"],
                        test_results=config_data["test_results"],
                        generation_time=config_data["generation_time"],
                    )
                    state.tested_configurations.append(config)

            logger.info(
                f"Resuming from previous state: {len(state.tested_configurations)} configurations tested so far"
            )
            return state

        except Exception as e:
            logger.warning(f"Failed to load previous optimization state: {e}")
            logger.warning("Starting with a fresh optimization state")
            return OptimizationState()


class ConfigurationManager:
    """Manages parameter spaces and generates configurations to test."""

    def __init__(self):
        # STORM parameter space - expanded for better quality optimization
        self.storm_param_space = {
            "max_conv_turn": [4, 5, 6, 7, 8],
            "max_perspective": [4, 6, 8, 10, 12],
            "search_top_k": [5, 7, 9, 11, 13],
            "max_thread_num": [2, 4, 6],
        }

        # RAG parameter space - expanded for better performance
        self.rag_param_space = {
            "retrieval_k": [5, 7, 9, 11],
            "num_queries": [5, 7, 9, 11],
            "max_passages": [10, 15, 20, 25],
        }

        # Current baseline configurations - using original hardcoded values as baseline
        self.storm_baseline = {
            "max_conv_turn": 4,
            "max_perspective": 4,
            "search_top_k": 5,
            "max_thread_num": 4,
        }

        self.rag_baseline = {"retrieval_k": 7, "num_queries": 7, "max_passages": 15}

    def get_baseline_config(self, method: str) -> Dict[str, Any]:
        """Get baseline configuration for method."""
        if method == "storm":
            return self.storm_baseline.copy()
        elif method == "rag":
            return self.rag_baseline.copy()
        else:
            raise ValueError(f"Unknown method: {method}")

    def generate_storm_configurations(
        self, max_configs: int = 75
    ) -> List[Dict[str, Any]]:
        """Generate STORM configurations to test."""
        # Start with baseline
        configs = [self.storm_baseline.copy()]

        # Generate incremental improvements (one parameter at a time)
        for param, values in self.storm_param_space.items():
            for value in values:
                if value != self.storm_baseline[param]:
                    config = self.storm_baseline.copy()
                    config[param] = value
                    configs.append(config)

        # Generate promising combinations
        promising_combinations = []

        # High conversation turns + high perspectives (STORM's strength)
        for conv_turn in [6, 7, 8]:
            for perspective in [8, 10, 12]:
                for search_k in [9, 11]:
                    config = self.storm_baseline.copy()
                    config.update(
                        {
                            "max_conv_turn": conv_turn,
                            "max_perspective": perspective,
                            "search_top_k": search_k,
                            "max_thread_num": 4,  # Use more threads for higher quality
                        }
                    )
                    promising_combinations.append(config)

        # Also test some extreme high-quality configurations
        extreme_configs = [
            {
                "max_conv_turn": 8,
                "max_perspective": 12,
                "search_top_k": 13,
                "max_thread_num": 6,
            },
            {
                "max_conv_turn": 7,
                "max_perspective": 10,
                "search_top_k": 11,
                "max_thread_num": 4,
            },
        ]

        for config in extreme_configs:
            full_config = self.storm_baseline.copy()
            full_config.update(config)
            promising_combinations.append(full_config)

        configs.extend(promising_combinations)

        # Remove duplicates while preserving order
        seen = set()
        unique_configs = []
        for config in configs:
            config_tuple = tuple(sorted(config.items()))
            if config_tuple not in seen:
                seen.add(config_tuple)
                unique_configs.append(config)

        # Limit to max_configs, prioritizing early configs (incremental + promising)
        return unique_configs[:max_configs]

    def generate_rag_configurations(
        self, max_configs: int = 15
    ) -> List[Dict[str, Any]]:
        """Generate RAG configurations to test."""
        # Start with baseline
        configs = [self.rag_baseline.copy()]

        # Generate incremental improvements
        for param, values in self.rag_param_space.items():
            for value in values:
                if value != self.rag_baseline[param]:
                    config = self.rag_baseline.copy()
                    config[param] = value
                    configs.append(config)

        # Generate a few promising combinations
        promising = [
            {"retrieval_k": 7, "num_queries": 7, "max_passages": 12},
            {"retrieval_k": 10, "num_queries": 10, "max_passages": 15},
            {"retrieval_k": 7, "num_queries": 10, "max_passages": 12},
        ]

        configs.extend(promising)

        # Remove duplicates
        seen = set()
        unique_configs = []
        for config in configs:
            config_tuple = tuple(sorted(config.items()))
            if config_tuple not in seen:
                seen.add(config_tuple)
                unique_configs.append(config)

        return unique_configs[:max_configs]


class AdaptiveOptimizer:
    """Main optimization orchestrator with STORM priority."""

    def __init__(
        self,
        ollama_host: str,
        model_config,
        output_dir: str,
        topics_per_test: int = 5,
        resume: bool = False,
        resume_state_path: Optional[Path] = None,
    ):
        self.ollama_host = ollama_host
        self.model_config = model_config
        self.output_dir = Path(output_dir) / "optimization"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create configs directory for individual configuration test results
        (self.output_dir / "configs").mkdir(parents=True, exist_ok=True)

        self.resume = resume
        self.resume_state_path = resume_state_path

        self.topics_per_test = topics_per_test

        # Initialize components
        self.config_manager = ConfigurationManager()

        # Initialize performance tracker with resume capabilities
        if resume and resume_state_path and resume_state_path.exists():
            # Use the specified state file path directly
            self.performance_tracker = PerformanceTracker(
                self.output_dir, resume_state_path
            )
            logger.info(f"Resuming from specific state file: {resume_state_path}")
        else:
            self.performance_tracker = PerformanceTracker(self.output_dir)
            if resume:
                logger.info(f"Will attempt to resume from state in: {self.output_dir}")

        self.evaluator = ArticleEvaluator()

        # Load test topics
        freshwiki = FreshWikiLoader()
        all_entries = freshwiki.get_evaluation_sample(100)  # Get larger pool
        self.test_topics = [entry.topic for entry in all_entries[:topics_per_test]]

        logger.info(f"🎯 Optimizer initialized:")
        logger.info(f"   📍 Output: {self.output_dir}")
        logger.info(f"   📝 Topics per test: {topics_per_test}")
        logger.info(f"   🧪 Test topics: {self.test_topics}")

    def run_configuration_test(
        self, method: str, config: Dict[str, Any]
    ) -> OptimizationConfig:
        """Run a single configuration test."""
        logger.info(f"🧪 Testing {method.upper()} config: {config}")

        start_time = time.time()

        # Create a descriptive folder name for this configuration test
        config_desc = []
        for k, v in config.items():
            short_key = k.replace("max_", "").replace("_", "")[
                :5
            ]  # Shorten keys like max_conv_turn to "conv"
            config_desc.append(f"{short_key}{v}")
        config_folder_name = f"{method}_{'-'.join(config_desc)}_{int(time.time())}"

        # Create runner for this configuration with descriptive output folder
        config_output_dir = self.output_dir / "configs" / config_folder_name
        config_output_dir.mkdir(parents=True, exist_ok=True)

        output_manager = OutputManager(str(config_output_dir), debug_mode=False)
        runner = BaselineRunner(self.ollama_host, self.model_config, output_manager)

        # Run tests on all topics
        results = []
        total_metrics = {
            metric: []
            for metric in [
                "rouge_1",
                "rouge_2",
                "rouge_l",
                "heading_soft_recall",
                "heading_entity_recall",
                "article_entity_recall",
            ]
        }

        # Save configuration summary at the top level
        config_summary_file = config_output_dir / "config_summary.json"
        with open(config_summary_file, "w") as f:
            summary = {
                "method": method,
                "config": config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_topics": self.test_topics,
            }
            json.dump(summary, f, indent=2)

        logger.info(f"Config details saved to: {config_summary_file}")

        for topic in self.test_topics:
            try:
                # Generate article with configuration
                if method == "storm":
                    article = runner.run_storm_with_config(topic, storm_config=config)
                elif method == "rag":
                    article = runner.run_rag_with_config(topic, rag_config=config)

                # Evaluate article
                freshwiki = FreshWikiLoader()
                # Fix: Use get_entry_by_topic instead of get_entry_by_title
                reference = freshwiki.get_entry_by_topic(topic)

                if reference:
                    metrics = self.evaluator.evaluate_article(article, reference)

                    # Accumulate metrics
                    for metric, value in metrics.items():
                        if metric in total_metrics:
                            total_metrics[metric].append(value)

                    result_data = {
                        "topic": topic,
                        "config_used": config,
                        "metrics": metrics,
                        "word_count": (
                            len(article.content.split()) if article.content else 0
                        ),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    results.append(result_data)

                    # Save detailed result to topic folder
                    topic_dir = output_manager.get_topic_directory(topic)

                    # Save configuration and results to separate files for easy reference
                    config_file = Path(topic_dir) / "configuration.json"
                    results_file = Path(topic_dir) / "metrics.json"
                    try:
                        with open(config_file, "w") as f:
                            json.dump(config, f, indent=2)

                        with open(results_file, "w") as f:
                            json.dump(result_data, f, indent=2)

                        logger.debug(f"Configuration and metrics saved to {topic_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to save files to {topic_dir}: {e}")

                    composite = self.performance_tracker.calculate_composite_score(
                        metrics
                    )
                    logger.debug(f"   ✅ {topic}: composite = {composite:.3f}")
                else:
                    logger.error(
                        f"   ❌ {topic}: Reference article not found in FreshWiki data"
                    )
                    results.append(
                        {"topic": topic, "error": "Reference article not found"}
                    )

            except Exception as e:
                logger.error(f"   ❌ {topic}: {e}")
                results.append({"topic": topic, "error": str(e)})

        # Calculate average metrics
        avg_metrics = {}
        for metric, values in total_metrics.items():
            if values:
                avg_metrics[metric] = np.mean(values)
            else:
                avg_metrics[metric] = 0.0

        composite_score = self.performance_tracker.calculate_composite_score(
            avg_metrics
        )
        generation_time = time.time() - start_time

        # Create optimization config
        opt_config = OptimizationConfig(
            method=method,
            parameters=config.copy(),
            composite_score=composite_score,
            individual_scores=avg_metrics.copy(),
            test_results=results,
            generation_time=generation_time,
        )

        # Save summary metrics to the config folder
        metrics_summary_file = config_output_dir / "metrics_summary.json"
        try:
            with open(metrics_summary_file, "w") as f:
                summary = {
                    "method": method,
                    "config": config,
                    "composite_score": composite_score,
                    "metrics": avg_metrics,
                    "generation_time": generation_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "test_topics": self.test_topics,
                }
                json.dump(summary, f, indent=2)
            logger.info(f"Metrics summary saved to: {metrics_summary_file}")
        except Exception as e:
            logger.warning(f"Failed to save metrics summary: {e}")

        logger.info(f"📊 {method.upper()} Test Complete:")
        logger.info(f"   🎯 Composite Score: {composite_score:.2f}")
        logger.info(f"   📈 ROUGE-1: {avg_metrics.get('rouge_1', 0):.2f}%")
        logger.info(
            f"   📋 Heading Soft Recall: {avg_metrics.get('heading_soft_recall', 0):.2f}%"
        )
        logger.info(
            f"   🏷️  Article Entity Recall: {avg_metrics.get('article_entity_recall', 0):.2f}%"
        )
        logger.info(f"   ⏱️  Time: {generation_time:.1f}s")
        logger.info(f"   📁 Results saved to: {config_output_dir}")

        return opt_config

    def optimize_storm(self) -> OptimizationConfig:
        """Optimize STORM configuration with comprehensive search."""
        logger.info("🌪️  STARTING STORM OPTIMIZATION")
        logger.info("=" * 60)

        # Check if we're resuming from previous optimization
        resume_optimization = False
        if (
            self.performance_tracker.state.baseline_storm
            and self.performance_tracker.state.current_best_storm
        ):
            resume_optimization = True
            logger.info(f"🔄 Resuming STORM optimization from previous state")
            logger.info(
                f"📊 Current best STORM score: {self.performance_tracker.state.current_best_storm.composite_score:.3f}"
            )
        else:
            # Test baseline first
            baseline_config = self.config_manager.get_baseline_config("storm")
            baseline_result = self.run_configuration_test("storm", baseline_config)

            self.performance_tracker.state.baseline_storm = baseline_result
            self.performance_tracker.state.current_best_storm = baseline_result
            self.performance_tracker.state.tested_configurations.append(baseline_result)

            # Save initial state
            self.performance_tracker.save_state()

            logger.info(f"📊 STORM Baseline: {baseline_result.composite_score:.3f}")

        # Generate configurations to test (expanded for better optimization)
        configs_to_test = self.config_manager.generate_storm_configurations(
            max_configs=25
        )
        logger.info(f"🎯 Generated {len(configs_to_test)} STORM configurations to test")

        # Filter out already tested configurations if resuming
        if resume_optimization:
            tested_config_params = [
                config.parameters
                for config in self.performance_tracker.state.tested_configurations
                if config.method == "storm"
            ]

            configs_to_test = [
                config
                for config in configs_to_test
                if config not in tested_config_params
            ]
            logger.info(
                f"🔄 Resuming with {len(configs_to_test)} untested configurations"
            )

            if not configs_to_test:
                logger.info(
                    "✅ All configurations already tested. Returning best result."
                )
                return self.performance_tracker.state.current_best_storm

        # Test each configuration
        improvements_found = 0
        for i, config in enumerate(configs_to_test, 1):
            if not resume_optimization and i == 1:
                # Skip baseline if not resuming (already tested)
                continue

            logger.info(f"\n🧪 STORM Test {i}/{len(configs_to_test)}")
            logger.info("-" * 40)

            try:
                result = self.run_configuration_test("storm", config)
                self.performance_tracker.state.tested_configurations.append(result)
                self.performance_tracker.state.total_tests_run += 1

                # Check for improvement
                if self.performance_tracker.update_best_configuration(result):
                    improvements_found += 1
                    logger.info(f"🎉 IMPROVEMENT #{improvements_found} FOUND!")

                    # Save progress
                    self.performance_tracker.save_state()

                else:
                    logger.info("📉 No improvement - continuing search...")
                    # Still save state to record this test
                    self.performance_tracker.save_state()

            except Exception as e:
                logger.error(f"❌ STORM test failed: {e}")
                continue

        best_storm = self.performance_tracker.state.current_best_storm
        improvement = best_storm.composite_score - baseline_result.composite_score

        logger.info("\n" + "=" * 60)
        logger.info("🌪️  STORM OPTIMIZATION COMPLETE")
        logger.info(f"📊 Best Score: {best_storm.composite_score:.3f}")
        logger.info(f"📈 Improvement: +{improvement:.3f}")
        logger.info(f"🔧 Best Config: {best_storm.parameters}")
        logger.info(f"🧪 Tests Run: {len(configs_to_test)}")
        logger.info(f"🎯 Improvements Found: {improvements_found}")

        return best_storm

    def optimize_rag(self) -> OptimizationConfig:
        """Optimize RAG configuration with focused search."""
        logger.info("\n🔍 STARTING RAG OPTIMIZATION")
        logger.info("=" * 60)

        # Check if we're resuming from previous optimization
        resume_optimization = False
        if (
            self.performance_tracker.state.baseline_rag
            and self.performance_tracker.state.current_best_rag
        ):
            resume_optimization = True
            logger.info(f"🔄 Resuming RAG optimization from previous state")
            logger.info(
                f"📊 Current best RAG score: {self.performance_tracker.state.current_best_rag.composite_score:.3f}"
            )
        else:
            # Test baseline first
            baseline_config = self.config_manager.get_baseline_config("rag")
            baseline_result = self.run_configuration_test("rag", baseline_config)

            self.performance_tracker.state.baseline_rag = baseline_result
            self.performance_tracker.state.current_best_rag = baseline_result
            self.performance_tracker.state.tested_configurations.append(baseline_result)

            # Save initial state
            self.performance_tracker.save_state()

            logger.info(f"📊 RAG Baseline: {baseline_result.composite_score:.3f}")

        # Generate configurations to test (reduced for faster optimization)
        configs_to_test = self.config_manager.generate_rag_configurations(max_configs=5)
        logger.info(f"🎯 Generated {len(configs_to_test)} RAG configurations to test")

        # Filter out already tested configurations if resuming
        if resume_optimization:
            tested_config_params = [
                config.parameters
                for config in self.performance_tracker.state.tested_configurations
                if config.method == "rag"
            ]

            configs_to_test = [
                config
                for config in configs_to_test
                if config not in tested_config_params
            ]
            logger.info(
                f"🔄 Resuming with {len(configs_to_test)} untested configurations"
            )

            if not configs_to_test:
                logger.info(
                    "✅ All configurations already tested. Returning best result."
                )
                return self.performance_tracker.state.current_best_rag

        # Test each configuration
        improvements_found = 0
        for i, config in enumerate(configs_to_test, 1):
            if not resume_optimization and i == 1:
                # Skip baseline if not resuming (already tested)
                continue

            logger.info(f"\n🧪 RAG Test {i}/{len(configs_to_test)}")
            logger.info("-" * 40)

            try:
                result = self.run_configuration_test("rag", config)
                self.performance_tracker.state.tested_configurations.append(result)
                self.performance_tracker.state.total_tests_run += 1

                # Check for improvement
                if self.performance_tracker.update_best_configuration(result):
                    improvements_found += 1
                    logger.info(f"🎉 IMPROVEMENT #{improvements_found} FOUND!")

                    # Save progress
                    self.performance_tracker.save_state()

                else:
                    logger.info("📉 No improvement - continuing search...")
                    # Still save state to record this test
                    self.performance_tracker.save_state()

            except Exception as e:
                logger.error(f"❌ RAG test failed: {e}")
                continue

        best_rag = self.performance_tracker.state.current_best_rag
        improvement = best_rag.composite_score - baseline_result.composite_score

        logger.info("\n" + "=" * 60)
        logger.info("🔍 RAG OPTIMIZATION COMPLETE")
        logger.info(f"📊 Best Score: {best_rag.composite_score:.3f}")
        logger.info(f"📈 Improvement: +{improvement:.3f}")
        logger.info(f"🔧 Best Config: {best_rag.parameters}")
        logger.info(f"🧪 Tests Run: {len(configs_to_test)}")
        logger.info(f"🎯 Improvements Found: {improvements_found}")

        return best_rag

    def run_full_optimization(self) -> Tuple[OptimizationConfig, OptimizationConfig]:
        """Run complete dual optimization with STORM priority."""
        logger.info("🚀 STARTING DUAL OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(
            f"🎯 Strategy: STORM Priority (80% budget) + RAG Focus (20% budget)"
        )
        logger.info(f"📝 Topics per test: {self.topics_per_test}")
        logger.info(f"📊 Composite scoring: 0.4×ROUGE-1 + 0.3×HSR + 0.3×AER")

        self.performance_tracker.state.optimization_start_time = time.time()

        # Phase 1: STORM Optimization (Priority)
        best_storm = self.optimize_storm()

        # Phase 2: RAG Optimization (Focused)
        best_rag = self.optimize_rag()

        # Final comparison and reporting
        total_time = (
            time.time() - self.performance_tracker.state.optimization_start_time
        )

        logger.info("\n" + "=" * 80)
        logger.info("🏁 DUAL OPTIMIZATION COMPLETE")
        logger.info("=" * 80)

        logger.info("\n📊 FINAL RESULTS:")
        logger.info(
            f"🌪️  STORM - Best: {best_storm.composite_score:.3f} | Config: {best_storm.parameters}"
        )
        logger.info(
            f"🔍 RAG   - Best: {best_rag.composite_score:.3f} | Config: {best_rag.parameters}"
        )

        # Determine winner
        if best_storm.composite_score > best_rag.composite_score:
            winner = "STORM"
            advantage = best_storm.composite_score - best_rag.composite_score
        else:
            winner = "RAG"
            advantage = best_rag.composite_score - best_storm.composite_score

        logger.info(f"\n🏆 WINNER: {winner} (+{advantage:.3f} advantage)")

        logger.info(f"\n📈 OPTIMIZATION SUMMARY:")
        logger.info(f"   ⏱️  Total Time: {total_time/60:.1f} minutes")
        logger.info(
            f"   🧪 Total Tests: {self.performance_tracker.state.total_tests_run}"
        )
        logger.info(f"   📁 Results saved to: {self.output_dir}")

        # Save final state
        self.performance_tracker.save_state()

        # Generate configuration index for easy comparison
        self.generate_config_index()

        logger.info(
            f"📁 Configuration details and summaries available at: {self.output_dir}"
        )
        logger.info(
            f"📊 View config_summary.md for a ranked list of all configurations"
        )

        return best_storm, best_rag

    def generate_config_index(self):
        """Generate an index file with all configurations and their results for easy comparison."""
        configs_dir = self.output_dir / "configs"
        if not configs_dir.exists():
            logger.info("No configuration folders to index")
            return

        index_data = []

        # Collect data from all configuration folders
        for config_dir in sorted(
            configs_dir.glob("*"), key=lambda x: x.stat().st_mtime
        ):
            metrics_file = config_dir / "metrics_summary.json"

            if not metrics_file.exists():
                continue

            try:
                with open(metrics_file, "r") as f:
                    metrics_data = json.load(f)

                index_entry = {
                    "folder": config_dir.name,
                    "method": metrics_data.get("method", "unknown"),
                    "config": metrics_data.get("config", {}),
                    "composite_score": metrics_data.get("composite_score", 0),
                    "rouge_1": metrics_data.get("metrics", {}).get("rouge_1", 0),
                    "heading_soft_recall": metrics_data.get("metrics", {}).get(
                        "heading_soft_recall", 0
                    ),
                    "article_entity_recall": metrics_data.get("metrics", {}).get(
                        "article_entity_recall", 0
                    ),
                    "generation_time": metrics_data.get("generation_time", 0),
                    "timestamp": metrics_data.get("timestamp", ""),
                }
                index_data.append(index_entry)

            except Exception as e:
                logger.warning(f"Error processing {metrics_file}: {e}")

        # Sort by composite score (highest first)
        index_data.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        # Generate the index file
        index_file = self.output_dir / "config_index.json"
        try:
            with open(index_file, "w") as f:
                json.dump(
                    {
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_configs": len(index_data),
                        "configs": index_data,
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Generated configuration index with {len(index_data)} entries at {index_file}"
            )

            # Also create a readable markdown summary
            md_file = self.output_dir / "config_summary.md"
            with open(md_file, "w") as f:
                f.write("# Optimization Configuration Results\n\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Total configurations tested: {len(index_data)}\n\n")

                f.write("## Top 5 Configurations\n\n")
                f.write(
                    "| Rank | Method | Composite | ROUGE-1 | HSR | AER | Configuration | Folder |\n"
                )
                f.write(
                    "|------|--------|-----------|---------|-----|-----|--------------|--------|\n"
                )

                for i, entry in enumerate(index_data[:5], 1):
                    config_str = ", ".join(
                        [f"{k}={v}" for k, v in entry.get("config", {}).items()]
                    )
                    f.write(
                        f"| {i} | {entry.get('method', 'unknown')} | "
                        f"{entry.get('composite_score', 0):.2f} | "
                        f"{entry.get('rouge_1', 0):.2f} | "
                        f"{entry.get('heading_soft_recall', 0):.2f} | "
                        f"{entry.get('article_entity_recall', 0):.2f} | "
                        f"{config_str} | {entry.get('folder', '')} |\n"
                    )

                f.write("\n## All Configurations\n\n")
                f.write(
                    "| Rank | Method | Composite | ROUGE-1 | HSR | AER | Configuration | Folder |\n"
                )
                f.write(
                    "|------|--------|-----------|---------|-----|-----|--------------|--------|\n"
                )

                for i, entry in enumerate(index_data, 1):
                    config_str = ", ".join(
                        [f"{k}={v}" for k, v in entry.get("config", {}).items()]
                    )
                    f.write(
                        f"| {i} | {entry.get('method', 'unknown')} | "
                        f"{entry.get('composite_score', 0):.2f} | "
                        f"{entry.get('rouge_1', 0):.2f} | "
                        f"{entry.get('heading_soft_recall', 0):.2f} | "
                        f"{entry.get('article_entity_recall', 0):.2f} | "
                        f"{config_str} | {entry.get('folder', '')} |\n"
                    )

            logger.info(f"Generated readable configuration summary at {md_file}")

        except Exception as e:
            logger.error(f"Failed to generate configuration index: {e}")
