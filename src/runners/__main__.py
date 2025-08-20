#!/usr/bin/env python3
"""
Main entry point for collaborative writing experiments.
Supports both Ollama and SLURM backends with factory pattern.
"""


import logging
import os
from dataclasses import replace

from src.config.baselines_model_config import ModelConfig
from src.config.collaboration_config import CollaborationConfig
from src.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from src.runners.cli_args import (
    parse_arguments,
    print_configuration,
    validate_arguments,
)
from src.runners.factory import create_runner
from src.utils.data import FreshWikiLoader
from src.utils.experiment import ExperimentStateManager, setup_output_directory
from src.utils.io import OutputManager, setup_logging

logger = logging.getLogger(__name__)


def setup_hf_cache_for_backend(backend: str):
    """Set HuggingFace cache directory based on backend mode."""
    if backend == "slurm":
        cache_dir = "/tmp/hf_cache_slurm"
    else:
        cache_dir = os.path.expanduser("~/.cache/huggingface")

    # Test write permissions and fallback if needed
    try:
        os.makedirs(cache_dir, exist_ok=True)
        test_file = os.path.join(cache_dir, "test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except (OSError, PermissionError):
        cache_dir = f"/tmp/hf_cache_{os.getenv('USER', 'unknown')}"
        os.makedirs(cache_dir, exist_ok=True)

    # Set environment variables that all HF libraries will use
    os.environ["HF_HOME"] = cache_dir
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir


def load_configurations(args):
    """Load all configuration objects from arguments."""

    # Load collaboration configuration
    try:
        collaboration_config = CollaborationConfig.from_yaml(args.collaboration_config)

        # Apply CLI overrides
        if args.max_iterations:
            collaboration_config.max_iterations = args.max_iterations
        if args.convergence_threshold:
            collaboration_config.convergence_threshold = args.convergence_threshold

        logger.info(f"📋 Loaded collaboration config: {args.collaboration_config}")
        logger.info(f"🤝 Max iterations: {collaboration_config.max_iterations}")
        logger.info(
            f"🎯 Convergence threshold: {collaboration_config.convergence_threshold}"
        )

    except Exception as e:
        logger.error(
            f"Failed to load collaboration config '{args.collaboration_config}': {e}"
        )
        logger.info("Falling back to default collaboration configuration")
        collaboration_config = CollaborationConfig()

    # Load retrieval configuration
    retrieval_config = None
    if args.retrieval_manager:
        try:
            retrieval_config = RetrievalConfig.from_base_config_with_overrides(
                rm_type=args.retrieval_manager,
                semantic_filtering_enabled=args.semantic_filtering or None,
            )
            logger.info(f"📋 Loaded retrieval config: {args.retrieval_manager}")
        except Exception as e:
            logger.error(f"Failed to create retrieval config: {e}")
            logger.info("Falling back to default retrieval configuration")
            retrieval_config = None
    else:
        # Use default config with CLI overrides
        try:
            retrieval_config = DEFAULT_RETRIEVAL_CONFIG
            if args.semantic_filtering:
                retrieval_config = replace(
                    retrieval_config, semantic_filtering_enabled=True
                )
                logger.info("✅ Enabled semantic filtering from CLI")
        except Exception as e:
            logger.error(f"Failed to apply CLI overrides to default config: {e}")
            retrieval_config = None

    # Load model configuration
    if args.model_config:
        try:
            model_config = ModelConfig.from_yaml(args.model_config)
            if args.override_model:
                model_config.override_model = args.override_model
            logger.info(f"📋 Loaded model config: {args.model_config}")
            logger.info(f"🎯 Model mode: {model_config.mode}")
        except Exception as e:
            logger.error(f"Failed to load model config '{args.model_config}': {e}")
            logger.info("Falling back to default model configuration")
            model_config = ModelConfig(
                mode=args.backend, override_model=args.override_model
            )
    else:
        # Default model config
        model_config = ModelConfig(
            mode=args.backend, override_model=args.override_model
        )

    return model_config, retrieval_config, collaboration_config


def run_collaborative_experiment(args):
    """Run the collaborative writing experiment."""

    setup_logging(args.log_level)

    try:
        # Load all configurations
        model_config, retrieval_config, collaboration_config = load_configurations(args)

        # Setup output directory
        output_dir = setup_output_directory(args)
        output_manager = OutputManager(str(output_dir), debug_mode=args.debug)

        # Initialize state manager
        state_manager = ExperimentStateManager(output_dir, args.methods)
        is_resume = (
            state_manager.load_checkpoint()
            if hasattr(state_manager, "load_checkpoint") and args.resume
            else False
        )

        if is_resume:
            logger.info("🔄 Resuming experiment from checkpoint")
        else:
            logger.info("🆕 Starting new collaborative writing experiment")

        # Create runner instance
        runner_class, runner_name = create_runner(args.backend)

        runner_kwargs = {}
        if args.backend == "slurm":
            # Add SLURM-specific parameters if needed
            pass

        runner = runner_class(
            model_config=model_config,
            output_manager=output_manager,
            retrieval_config=retrieval_config,
            collaboration_config=collaboration_config,
            **runner_kwargs,
        )

        runner.set_state_manager(state_manager)

        # Load topics
        freshwiki = FreshWikiLoader()
        entries = freshwiki.load_topics(args.num_topics)
        if not entries:
            logger.error("No FreshWiki entries found!")
            return 1

        topics = [entry.topic for entry in entries]
        logger.info(f"📚 Loaded {len(topics)} topics")

        # Print final configuration
        print_configuration(args)

        if args.dry_run:
            logger.info("🏃 Dry run mode - configuration validated, exiting")
            return 0

        # Run the experiment
        logger.info(f"🚀 Starting {runner_name} collaborative experiment")
        logger.info(f"📝 Methods: {', '.join(args.methods)}")
        logger.info(f"📊 Topics: {len(topics)}")

        results = runner.run_batch(topics, args.methods)

        # Log summary
        successful_results = [r for r in results if "error" not in r.metadata]
        failed_results = [r for r in results if "error" in r.metadata]

        logger.info(
            f"✅ Experiment completed: {len(successful_results)} successful, {len(failed_results)} failed"
        )

        if failed_results:
            logger.warning("Failed articles:")
            for article in failed_results:
                logger.warning(
                    f"  - {article.title}: {article.metadata.get('error', 'Unknown error')}"
                )

        # Save final state
        if state_manager and hasattr(state_manager, "save_checkpoint"):
            state_manager.save_checkpoint()

        return 0 if not failed_results else 1

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


def main():
    """Main entry point for collaborative writing experiments."""

    args = parse_arguments()

    try:
        validate_arguments(args)
    except ValueError as e:
        print(f"❌ Argument validation failed: {e}")
        return 1

    # Setup HuggingFace cache
    setup_hf_cache_for_backend(args.backend)

    # Run the experiment
    return run_collaborative_experiment(args)


if __name__ == "__main__":
    exit(main())
