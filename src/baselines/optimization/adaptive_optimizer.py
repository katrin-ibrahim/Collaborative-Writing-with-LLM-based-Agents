import time
from pathlib import Path

import json
import logging
import numpy as np
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.baselines.runner import BaselineRunner
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

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / "optimization_state.json"
        self.state = OptimizationState()

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
            f"ROUGE-1({rouge_1:.3f}) * 0.4 + "
            f"HSR({heading_soft_recall:.3f}) * 0.3 + "
            f"AER({article_entity_recall:.3f}) * 0.3 = {composite:.3f}"
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

        except Exception as e:
            logger.warning(f"Failed to save optimization state: {e}")


class ConfigurationManager:
    """Manages parameter spaces and generates configurations to test."""

    def __init__(self):
        # STORM parameter space - reduced for faster optimization
        self.storm_param_space = {
            "max_conv_turn": [3, 4, 5],
            "max_perspective": [3, 5, 7],
            "search_top_k": [3, 5, 7],
            "max_thread_num": [1, 2],
        }

        # RAG parameter space - minimal for faster optimization
        self.rag_param_space = {
            "retrieval_k": [3, 5, 7],
            "num_queries": [3, 5, 7],
            "max_passages": [5, 10, 15],
        }

        # Current baseline configurations
        self.storm_baseline = {
            "max_conv_turn": 2,
            "max_perspective": 2,
            "search_top_k": 2,
            "max_thread_num": 4,
        }

        self.rag_baseline = {"retrieval_k": 5, "num_queries": 5, "max_passages": 8}

    def get_baseline_config(self, method: str) -> Dict[str, Any]:
        """Get baseline configuration for method."""
        if method == "storm":
            return self.storm_baseline.copy()
        elif method == "rag":
            return self.rag_baseline.copy()
        else:
            raise ValueError(f"Unknown method: {method}")

    def generate_storm_configurations(
        self, max_configs: int = 50
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
        for conv_turn in [4, 5, 6]:
            for perspective in [4, 6, 8]:
                for search_k in [5, 7]:
                    config = self.storm_baseline.copy()
                    config.update(
                        {
                            "max_conv_turn": conv_turn,
                            "max_perspective": perspective,
                            "search_top_k": search_k,
                            "max_thread_num": 1,  # Conservative threading
                        }
                    )
                    promising_combinations.append(config)

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
        self, ollama_host: str, model_config, output_dir: str, topics_per_test: int = 5
    ):
        self.ollama_host = ollama_host
        self.model_config = model_config
        self.output_dir = Path(output_dir) / "optimization"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.topics_per_test = topics_per_test

        # Initialize components
        self.config_manager = ConfigurationManager()
        self.performance_tracker = PerformanceTracker(self.output_dir)
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

        # Create modified runner for this configuration
        output_manager = OutputManager(
            str(self.output_dir / f"test_{method}"), debug_mode=True
        )
        runner = BaselineRunner(self.ollama_host, self.model_config, output_manager)

        # Apply configuration modifications
        if method == "storm":
            self._apply_storm_config(runner, config)
        elif method == "rag":
            self._apply_rag_config(runner, config)

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

        for topic in self.test_topics:
            try:
                # Generate article
                if method == "storm":
                    article = runner.run_storm(topic)
                elif method == "rag":
                    article = runner.run_rag(topic)

                # Evaluate article
                freshwiki = FreshWikiLoader()
                reference = freshwiki.get_entry_by_title(topic)

                if reference:
                    metrics = self.evaluator.evaluate_article(article, reference)

                    # Accumulate metrics
                    for metric, value in metrics.items():
                        if metric in total_metrics:
                            total_metrics[metric].append(value)

                    results.append(
                        {
                            "topic": topic,
                            "metrics": metrics,
                            "word_count": (
                                len(article.content.split()) if article.content else 0
                            ),
                        }
                    )

                    logger.debug(
                        f"   ✅ {topic}: composite = {self.performance_tracker.calculate_composite_score(metrics):.3f}"
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

        logger.info(f"📊 {method.upper()} Test Complete:")
        logger.info(f"   🎯 Composite Score: {composite_score:.3f}")
        logger.info(f"   📈 ROUGE-1: {avg_metrics.get('rouge_1', 0):.3f}")
        logger.info(
            f"   📋 Heading Soft Recall: {avg_metrics.get('heading_soft_recall', 0):.3f}"
        )
        logger.info(
            f"   🏷️  Article Entity Recall: {avg_metrics.get('article_entity_recall', 0):.3f}"
        )
        logger.info(f"   ⏱️  Time: {generation_time:.1f}s")

        return opt_config

    def _apply_storm_config(self, runner: BaselineRunner, config: Dict[str, Any]):
        """Apply STORM configuration to runner."""
        # Monkey patch the setup_storm_runner function to use our config
        runner.__class__.__module__

        def custom_setup_storm_runner(client, model_config, storm_output_dir):
            from knowledge_storm import (
                STORMWikiLMConfigs,
                STORMWikiRunner,
                STORMWikiRunnerArguments,
            )

            from baselines.runner_utils import get_model_wrapper
            from baselines.wikipedia_rm import WikipediaSearchRM

            lm_config = STORMWikiLMConfigs()
            lm_config.set_conv_simulator_lm(
                get_model_wrapper(client, model_config, "fast")
            )
            lm_config.set_question_asker_lm(
                get_model_wrapper(client, model_config, "fast")
            )
            lm_config.set_outline_gen_lm(
                get_model_wrapper(client, model_config, "outline")
            )
            lm_config.set_article_gen_lm(
                get_model_wrapper(client, model_config, "writing")
            )
            lm_config.set_article_polish_lm(
                get_model_wrapper(client, model_config, "polish")
            )

            search_rm = WikipediaSearchRM(k=config.get("search_top_k", 3))

            # Use our optimized configuration
            engine_args = STORMWikiRunnerArguments(
                output_dir=storm_output_dir,
                max_conv_turn=config.get("max_conv_turn", 2),
                max_perspective=config.get("max_perspective", 2),
                search_top_k=config.get("search_top_k", 2),
                max_thread_num=config.get("max_thread_num", 4),
            )

            return STORMWikiRunner(engine_args, lm_config, search_rm), storm_output_dir

        # Replace the setup function temporarily
        import baselines.configure_storm

        baselines.configure_storm.setup_storm_runner = custom_setup_storm_runner

    def _apply_rag_config(self, runner: BaselineRunner, config: Dict[str, Any]):
        """Apply RAG configuration to runner."""
        # Monkey patch the RAG method to use our config
        runner.run_rag

        def custom_run_rag(topic):
            logger.info(f"Running Enhanced RAG with config {config} for: {topic}")

            try:
                from baselines.runner_utils import (
                    create_context_from_passages,
                    generate_article_with_context,
                    generate_search_queries,
                    retrieve_and_format_passages,
                )
                from baselines.wikipedia_rm import WikipediaSearchRM
                from utils.data_models import Article

                start_time = time.time()

                # Use configured parameters
                retrieval_system = WikipediaSearchRM(k=config.get("retrieval_k", 5))

                queries = generate_search_queries(
                    runner.client,
                    runner.model_config,
                    topic,
                    num_queries=config.get("num_queries", 5),
                )

                passages = retrieve_and_format_passages(retrieval_system, queries)

                context = create_context_from_passages(
                    passages, max_passages=config.get("max_passages", 8)
                )

                content = generate_article_with_context(
                    runner.client, runner.model_config, topic, context
                )

                generation_time = time.time() - start_time

                article = Article(
                    title=topic,
                    content=content,
                    sections={},
                    metadata={
                        "method": "rag",
                        "word_count": len(content.split()) if content else 0,
                        "generation_time": generation_time,
                        "model": runner.model_config.get_model_for_task("writing"),
                        "config": config.copy(),
                    },
                )

                return article

            except Exception as e:
                logger.error(f"Enhanced RAG failed for {topic}: {e}")
                from baselines.runner_utils import error_article

                return error_article(topic, e, "rag")

        runner.run_rag = custom_run_rag

    def optimize_storm(self) -> OptimizationConfig:
        """Optimize STORM configuration with comprehensive search."""
        logger.info("🌪️  STARTING STORM OPTIMIZATION")
        logger.info("=" * 60)

        # Test baseline first
        baseline_config = self.config_manager.get_baseline_config("storm")
        baseline_result = self.run_configuration_test("storm", baseline_config)

        self.performance_tracker.state.baseline_storm = baseline_result
        self.performance_tracker.state.current_best_storm = baseline_result
        self.performance_tracker.state.tested_configurations.append(baseline_result)

        logger.info(f"📊 STORM Baseline: {baseline_result.composite_score:.3f}")

        # Generate configurations to test (reduced for faster optimization)
        configs_to_test = self.config_manager.generate_storm_configurations(
            max_configs=10
        )
        logger.info(f"🎯 Generated {len(configs_to_test)} STORM configurations to test")

        # Test each configuration
        improvements_found = 0
        for i, config in enumerate(
            configs_to_test[1:], 1
        ):  # Skip baseline (already tested)
            logger.info(f"\n🧪 STORM Test {i}/{len(configs_to_test)-1}")
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

        # Test baseline first
        baseline_config = self.config_manager.get_baseline_config("rag")
        baseline_result = self.run_configuration_test("rag", baseline_config)

        self.performance_tracker.state.baseline_rag = baseline_result
        self.performance_tracker.state.current_best_rag = baseline_result
        self.performance_tracker.state.tested_configurations.append(baseline_result)

        logger.info(f"📊 RAG Baseline: {baseline_result.composite_score:.3f}")

        # Generate configurations to test (reduced for faster optimization)
        configs_to_test = self.config_manager.generate_rag_configurations(max_configs=5)
        logger.info(f"🎯 Generated {len(configs_to_test)} RAG configurations to test")

        # Test each configuration
        improvements_found = 0
        for i, config in enumerate(configs_to_test[1:], 1):  # Skip baseline
            logger.info(f"\n🧪 RAG Test {i}/{len(configs_to_test)-1}")
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

        return best_storm, best_rag
