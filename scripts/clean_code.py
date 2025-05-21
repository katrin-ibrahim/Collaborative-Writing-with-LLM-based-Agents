#!/usr/bin/env python3
"""
Script to clean up Python code by:
1. Removing unused imports
2. Sorting imports
3. Ensuring consistent formatting
"""

import os
import subprocess
import sys
from pathlib import Path

def clean_file(file_path):
    """Clean a single file."""
    print(f"Cleaning {file_path}...")
    
    # Remove unused imports
    subprocess.run([
        "autoflake",
        "--in-place",
        "--remove-all-unused-imports",
        "--remove-unused-variables",
        file_path
    ], check=False)
    
    # Sort imports
    subprocess.run([
        "isort",
        file_path
    ], check=False)
    
    print(f"✓ Cleaned {file_path}")

def clean_directory(directory, extensions=['.py']):
    """Clean all files with specified extensions in a directory (recursive)."""
    path = Path(directory)
    for file_path in path.glob('**/*'):
        if file_path.is_file() and file_path.suffix in extensions:
            clean_file(str(file_path))

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Clean specific files or directories
        for path in sys.argv[1:]:
            if os.path.isfile(path) and path.endswith('.py'):
                clean_file(path)
            elif os.path.isdir(path):
                clean_directory(path)
    else:
        # Clean the entire project by default
        project_root = os.path.dirname(os.path.abspath(__file__))
        clean_directory(project_root)

if __name__ == "__main__":
    main()
