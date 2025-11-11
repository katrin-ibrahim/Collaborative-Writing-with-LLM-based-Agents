"""
Experiment setup and directory management utilities.
"""

from pathlib import Path

import logging

logger = logging.getLogger(__name__)


def setup_output_directory(args) -> Path:
    """
    Setup output directory for new or resumed experiments.
    Automatically detects if checkpoint exists when using --experiment_name and resumes.
    """
    from src.utils.io import OutputManager

    if getattr(args, "resume_dir", None):
        resume_dir = OutputManager.verify_resume_dir(args.resume_dir)
        logger.info(f"📂 Resuming from specified directory: {resume_dir}")
        return Path(resume_dir)

    if getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📂 Using custom output directory: {output_dir}")
        return output_dir

    custom_name = getattr(args, "experiment_name", None)

    if custom_name:
        potential_dir = Path("results") / args.backend / custom_name
        checkpoint_file = potential_dir / "checkpoint.json"

        if checkpoint_file.exists():
            logger.info(
                f"📂 Found existing checkpoint, resuming experiment: {potential_dir}"
            )
            return potential_dir
        else:
            logger.info(f"📂 Creating new experiment with custom name: {custom_name}")

    # Call create_output_dir with custom_name only when it's a non-empty str to satisfy type checkers
    if isinstance(custom_name, str) and custom_name:
        output_path = OutputManager.create_output_dir(
            args.backend, args.methods, args.num_topics, custom_name=custom_name
        )
    else:
        output_path = OutputManager.create_output_dir(
            args.backend, args.methods, args.num_topics
        )

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
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
