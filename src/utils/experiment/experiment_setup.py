"""
Experiment setup and directory management utilities.
"""

from pathlib import Path

import logging

logger = logging.getLogger(__name__)


def setup_output_directory(args) -> Path:
    """Setup output directory for new or resumed experiments."""
    from src.utils.io import OutputManager

    if args.resume_dir:
        # Resume from specific directory
        resume_dir = OutputManager.verify_resume_dir(args.resume_dir)
        logger.info(f"📂 Resuming from specified directory: {resume_dir}")
        return Path(resume_dir)

    # Check if custom output directory is specified
    if hasattr(args, "output_dir") and args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📂 Using custom output directory: {output_dir}")
        return output_dir

    # If no custom output_dir specified, create new experiment directory using OutputManager
    custom_name = (
        getattr(args, "experiment_name", None)
        if hasattr(args, "experiment_name")
        else None
    )
    output_path = OutputManager.create_output_dir(
        args.backend, args.methods, args.num_topics, custom_name=custom_name
    )
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if custom_name:
        logger.info(f"📂 Created new run directory with custom name: {output_dir}")
    else:
        logger.info(f"📂 Created new run directory: {output_dir}")
    return output_dir


def find_project_root() -> str:
    """Find the project root directory."""
    current = Path(__file__).parent.parent.parent.absolute()

    # Look for indicators of project root
    indicators = ["requirements.txt", "README.md", ".git", "src"]

    while current != current.parent:
        if any((current / indicator).exists() for indicator in indicators):
            return str(current)
        current = current.parent

    # Fallback to current directory
    return str(Path.cwd())
