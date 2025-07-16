"""
CLI Integration for Adaptive Optimizer
Add this to your existing main.py or create as optimize.py
"""

import argparse
import sys
from pathlib import Path

import logging

from src.baselines.optimization.adaptive_optimizer import AdaptiveOptimizer
from src.config.baselines_model_config import ModelConfig
from src.utils.logging_setup import setup_logging


def add_optimization_args(parser: argparse.ArgumentParser):
    """Add optimization arguments to existing CLI parser."""

    # Optimization is the default behavior, no need for a separate flag

    # Optimization-specific parameters
    optimization_group = parser.add_argument_group("optimization options")

    optimization_group.add_argument(
        "--opt-topics",
        type=int,
        default=3,
        help="Number of topics to test per configuration (default: 3)",
    )

    optimization_group.add_argument(
        "--opt-methods",
        nargs="+",
        choices=["storm", "rag", "both"],
        default=["both"],
        help="Methods to optimize (default: both)",
    )

    optimization_group.add_argument(
        "--opt-metric",
        choices=["composite", "rouge_1", "heading_soft_recall"],
        default="composite",
        help="Primary optimization metric (default: composite)",
    )

    optimization_group.add_argument(
        "--opt-max-configs",
        type=int,
        default=10,
        help="Maximum configurations to test per method (default: 10 for STORM, 5 for RAG)",
    )


def load_model_config(config_file: str) -> ModelConfig:
    """Load model configuration from file."""
    if Path(config_file).exists():
        try:
            import yaml

            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
            return ModelConfig.from_dict(config_dict)
        except Exception as e:
            logging.warning(f"Failed to load model config: {e}")

    logging.info("Using default model configuration")
    return ModelConfig()


def run_optimization(args):
    """Run the optimization process."""

    # Setup logging with detailed output for optimization
    setup_logging("DEBUG" if args.debug else "INFO")
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting Adaptive Configuration Optimization")
    logger.info("=" * 80)

    # Load configuration
    try:
        model_config = load_model_config(args.model_config)
        logger.info(f"🤖 Model config loaded from: {args.model_config}")

    except Exception as e:
        logger.error(f"Failed to load model configuration: {e}")
        sys.exit(1)

    # Initialize optimizer
    try:
        optimizer = AdaptiveOptimizer(
            ollama_host=args.ollama_host,
            model_config=model_config,
            output_dir=args.output_dir,
            topics_per_test=args.opt_topics,
        )

        logger.info(f"🎯 Optimization Configuration:")
        logger.info(f"   📡 Ollama Host: {args.ollama_host}")
        logger.info(f"   📁 Output Dir: {args.output_dir}")
        logger.info(f"   📝 Topics per test: {args.opt_topics}")
        logger.info(f"   🎯 Methods: {args.opt_methods}")
        logger.info(f"   📊 Metric: {args.opt_metric}")

    except Exception as e:
        logger.error(f"Failed to initialize optimizer: {e}")
        sys.exit(1)

    # Run optimization based on methods specified
    try:
        if "both" in args.opt_methods:
            # Run full dual optimization
            best_storm, best_rag = optimizer.run_full_optimization()

            # Print final recommendations
            print_optimization_results(best_storm, best_rag, optimizer)

        elif "storm" in args.opt_methods:
            # STORM only optimization
            logger.info("🌪️ Running STORM-only optimization")
            best_storm = optimizer.optimize_storm()
            print_storm_results(best_storm, optimizer)

        elif "rag" in args.opt_methods:
            # RAG only optimization
            logger.info("🔍 Running RAG-only optimization")
            best_rag = optimizer.optimize_rag()
            print_rag_results(best_rag, optimizer)

    except KeyboardInterrupt:
        logger.info("\n⚠️ Optimization interrupted by user")
        logger.info("💾 Partial results saved to optimization state file")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


def print_optimization_results(best_storm, best_rag, optimizer):
    """Print comprehensive optimization results."""

    print("\n" + "=" * 80)
    print("🏁 OPTIMIZATION COMPLETE - CONFIGURATION RECOMMENDATIONS")
    print("=" * 80)

    # Performance comparison
    baseline_storm = optimizer.performance_tracker.state.baseline_storm
    baseline_rag = optimizer.performance_tracker.state.baseline_rag

    storm_improvement = best_storm.composite_score - baseline_storm.composite_score
    rag_improvement = best_rag.composite_score - baseline_rag.composite_score

    print(f"\n📊 PERFORMANCE IMPROVEMENTS:")
    print(
        f"🌪️  STORM: {baseline_storm.composite_score:.3f} → {best_storm.composite_score:.3f} (+{storm_improvement:.3f})"
    )
    print(
        f"🔍 RAG:   {baseline_rag.composite_score:.3f} → {best_rag.composite_score:.3f} (+{rag_improvement:.3f})"
    )

    # Determine winner
    if best_storm.composite_score > best_rag.composite_score:
        winner = "STORM"
        advantage = best_storm.composite_score - best_rag.composite_score
        print(f"\n🏆 RECOMMENDATION: Use optimized STORM (+{advantage:.3f} over RAG)")
    else:
        winner = "RAG"
        advantage = best_rag.composite_score - best_storm.composite_score
        print(f"\n🏆 RECOMMENDATION: Use optimized RAG (+{advantage:.3f} over STORM)")

    # Configuration details
    print(f"\n🔧 OPTIMIZED CONFIGURATIONS:")
    print(f"\n🌪️  STORM Configuration:")
    for param, value in best_storm.parameters.items():
        baseline_val = baseline_storm.parameters.get(param, "N/A")
        change_indicator = "📈" if value != baseline_val else "➡️"
        print(f"   {change_indicator} {param}: {baseline_val} → {value}")

    print(f"\n🔍 RAG Configuration:")
    for param, value in best_rag.parameters.items():
        baseline_val = baseline_rag.parameters.get(param, "N/A")
        change_indicator = "📈" if value != baseline_val else "➡️"
        print(f"   {change_indicator} {param}: {baseline_val} → {value}")

    # Detailed metrics breakdown
    print(f"\n📈 DETAILED METRICS BREAKDOWN:")
    print(f"\n🌪️  STORM (Best Score: {best_storm.composite_score:.3f}):")
    for metric, score in best_storm.individual_scores.items():
        print(f"   📊 {metric}: {score:.3f}")

    print(f"\n🔍 RAG (Best Score: {best_rag.composite_score:.3f}):")
    for metric, score in best_rag.individual_scores.items():
        print(f"   📊 {metric}: {score:.3f}")

    # Implementation instructions
    print(f"\n🛠️  IMPLEMENTATION INSTRUCTIONS:")
    print(f"\nTo use the optimized {winner} configuration:")

    if winner == "STORM":
        print(f"Update your src/baselines/configure_storm.py:")
        print(f"```python")
        print(f"engine_args = STORMWikiRunnerArguments(")
        print(f"    output_dir=storm_output_dir,")
        for param, value in best_storm.parameters.items():
            print(f"    {param}={value},")
        print(f")")
        print(f"```")
    else:
        print(f"Update your RAG parameters in src/baselines/runner.py:")
        print(f"```python")
        for param, value in best_rag.parameters.items():
            print(f"{param} = {value}")
        print(f"```")

    print(f"\n💾 Full results saved to: {optimizer.output_dir}")


def print_storm_results(best_storm, optimizer):
    """Print STORM-only optimization results."""
    baseline = optimizer.performance_tracker.state.baseline_storm
    improvement = best_storm.composite_score - baseline.composite_score

    print("\n" + "=" * 60)
    print("🌪️ STORM OPTIMIZATION RESULTS")
    print("=" * 60)
    print(
        f"📊 Performance: {baseline.composite_score:.3f} → {best_storm.composite_score:.3f} (+{improvement:.3f})"
    )
    print(f"🔧 Best Configuration: {best_storm.parameters}")


def print_rag_results(best_rag, optimizer):
    """Print RAG-only optimization results."""
    baseline = optimizer.performance_tracker.state.baseline_rag
    improvement = best_rag.composite_score - baseline.composite_score

    print("\n" + "=" * 60)
    print("🔍 RAG OPTIMIZATION RESULTS")
    print("=" * 60)
    print(
        f"📊 Performance: {baseline.composite_score:.3f} → {best_rag.composite_score:.3f} (+{improvement:.3f})"
    )
    print(f"🔧 Best Configuration: {best_rag.parameters}")


def main():
    """Main entry point for optimization CLI."""
    parser = argparse.ArgumentParser(
        description="Adaptive Configuration Optimizer for AI Writer Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize both STORM and RAG with 3 topics per test
  python optimize.py --opt-topics 3

  # Optimize only STORM with more thorough testing
  python optimize.py --opt-methods storm --opt-topics 5

  # Quick RAG optimization
  python optimize.py --opt-methods rag --opt-topics 3

  # Custom configuration
  python optimize.py --opt-topics 2 --opt-max-configs 5 --debug
        """,
    )

    # Basic arguments (similar to your existing CLI)
    parser.add_argument(
        "--ollama_host", default="http://10.167.31.201:11434/", help="Ollama server URL"
    )
    parser.add_argument(
        "--model_config", default="config/models.yaml", help="Model configuration file"
    )
    parser.add_argument(
        "--output_dir",
        default="results/optimization",
        help="Output directory for results",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with detailed logging"
    )

    # Add optimization arguments
    add_optimization_args(parser)

    args = parser.parse_args()

    # Run optimization directly (it's the module's primary function)
    run_optimization(args)


if __name__ == "__main__":
    main()
