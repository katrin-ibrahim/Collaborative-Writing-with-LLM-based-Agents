"""
Utility functions for local baseline experiments.
Uses shared utilities to avoid baselines module imports.
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import shared utilities (avoiding baselines module for SLURM compatibility)
from shared.prompt_utils import (
    build_direct_prompt,
    count_words,
    error_article,
    post_process_article,
)

# Re-export for convenience
__all__ = [
    "build_direct_prompt",
    "post_process_article",
    "error_article",
    "count_words",
]
