#!/usr/bin/env python3
"""
Entry point for Ollama model baseline experiments.
This is a wrapper around the unified entry point.
"""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import the main entry point
from src.baselines.__main__ import main

if __name__ == "__main__":
    # Force backend to ollama
    if "--backend" not in sys.argv:
        sys.argv.extend(["--backend", "ollama"])
    exit(main())
