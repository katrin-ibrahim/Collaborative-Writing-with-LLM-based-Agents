#!/usr/bin/env python3
"""
Main entry point for collaborative writing experiments.
Supports both Ollama and SLURM backends with factory pattern.
"""


import time

import logging
import os

from src.config import (
    CollaborationConfig,
    ModelConfig,
    RetrievalConfig,
)
from src.config.config_context import ConfigContext
from src.main.cli_args import parse_arguments
from src.main.runner import Runner
from src.utils.data import FreshWikiLoader
from src.utils.experiment import save_final_results, setup_output_directory
from src.utils.io import OutputManager

# Logging will be configured after parsing args to respect --debug flag
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
    """Load all configuration objects from arguments using consistent abstract base."""

    # Collaboration configuration - always has default
    try:
        collaboration_config = CollaborationConfig.from_yaml_with_overrides(
            args.collaboration_config,
            max_iterations=args.max_iterations,
            min_iterations=getattr(args, "min_iterations", None),
            convergence_threshold=args.convergence_threshold,
            resolution_rate_threshold=getattr(args, "resolution_rate_threshold", None),
            stall_tolerance=getattr(args, "stall_tolerance", None),
            min_improvement=getattr(args, "min_improvement", None),
            small_tail_max=getattr(args, "small_tail_max", None),
            writing_mode=args.writing_mode,
            revise_mode=args.revise_mode,
            should_self_refine=not args.no_self_refine,
            adaptive_iterations=getattr(args, "adaptive_iterations", None),
            adaptive_extension_max=getattr(args, "adaptive_extension_max", None),
            adaptive_improvement_threshold=getattr(
                args, "adaptive_improvement_threshold", None
            ),
            adaptive_check_iteration=getattr(args, "adaptive_check_iteration", None),
        )
        logger.info(f"Loaded collaboration config: {args.collaboration_config}")

    except Exception as e:
        logger.error(
            f"Failed to load collaboration config '{args.collaboration_config}': {e}, Falling back to default configuration"
        )
        collaboration_config = CollaborationConfig.get_default()

    # Model configuration - may be None
    try:
        model_config = ModelConfig.from_yaml_with_overrides(
            args.model_config,  # Could be None
            override_model=args.override_model,
            mode=args.backend,  # Pass backend as override
            outline_model=args.outline_model,
            research_model=args.research_model,
            writer_model=args.writer_model,
            revision_model=args.revision_model,
            self_refine_model=args.self_refine_model,
            reviewer_model=args.reviewer_model,
        )
        if args.model_config:
            logger.info(f"Loaded model config: {args.model_config}")
        else:
            logger.info(f"Using default model config with backend: {args.backend}")

    except Exception as e:
        logger.error(f"Failed to load model config '{args.model_config}': {e}")
        logger.info("Falling back to default model configuration")
        model_config = ModelConfig.get_default()
        model_config.mode = args.backend
        if args.override_model:
            model_config.override_model = args.override_model

    # Retrieval configuration - may be None
    try:

        retrieval_config = RetrievalConfig.from_yaml_with_overrides(
            retrieval_manager=args.retrieval_manager,  # Could be None
        )
        if args.retrieval_manager:
            logger.info(f"Loaded retrieval config: {args.retrieval_manager}")
        else:
            logger.info("Using default retrieval configuration")

    except Exception as e:
        logger.error(f"Failed to load retrieval config '{args.retrieval_manager}': {e}")
        logger.info("Falling back to default retrieval configuration")
        retrieval_config = RetrievalConfig.get_default()

    return model_config, retrieval_config, collaboration_config


def main():
    """Main entry point for collaborative writing experiments."""

    args = parse_arguments()

    # Configure logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M",
        force=True,  # Override any existing configuration
    )

    # Setup HuggingFace cache
    setup_hf_cache_for_backend(args.backend)

    # Create runner instance
    model_config, retrieval_config, collaboration_config = load_configurations(args)
    backend_kwargs = {}

    # Get ollama_host from model config if specified (defaults to UKP server in OllamaEngine)
    if hasattr(model_config, "ollama_host") and model_config.ollama_host:
        backend_kwargs["ollama_host"] = model_config.ollama_host

    if hasattr(args, "device") and args.device:
        backend_kwargs["device"] = args.device
    ConfigContext.initialize(
        model_config=model_config,
        retrieval_config=retrieval_config,
        collaboration_config=collaboration_config,
        backend=args.backend,
        **backend_kwargs,
    )

    # Setup output directory
    output_dir = setup_output_directory(args)
    output_manager = OutputManager(str(output_dir), debug_mode=args.debug)

    # Register output_dir in ConfigContext so methods can access it
    ConfigContext.set_output_dir(str(output_dir))

    # Initialize state manager for resume/checkpoint functionality
    from src.utils.experiment import ExperimentStateManager

    state_manager = ExperimentStateManager(output_dir, args.methods)
    state_manager.load_checkpoint()

    # Save initial checkpoint to ensure it exists from the start
    state_manager.save_checkpoint()

    runner = Runner(
        output_manager=output_manager,
    )
    runner.set_state_manager(state_manager)

    # Load topics
    freshwiki = FreshWikiLoader()
    topics = [entry.topic for entry in freshwiki.load_topics(args.num_topics)]

    # Run experiment
    run_start = time.time()
    runner.run(topics, args.methods)

    # Persist results.json based on generated articles
    try:
        total_time = time.time() - run_start
        save_final_results(
            output_dir, topics, args.methods, total_time, backend=args.backend
        )
    except Exception as e:
        logger.warning(f"Failed to save final results.json after generation: {e}")


if __name__ == "__main__":
    exit(main())
