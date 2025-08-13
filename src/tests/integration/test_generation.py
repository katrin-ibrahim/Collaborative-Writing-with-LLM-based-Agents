"""
Integration test for stepping through baseline generation using main entry point.
Run with: pytest tests/integration/test_generation.py -s -v --pdb
"""

import sys
from unittest.mock import patch


class TestGenerationIntegration:
    """Step through actual baseline generation using main entry point."""

    def test_baselines(self):
        """Step through generation using main entry point - set breakpoints to debug."""
        # Mock command line arguments
        test_args = [
            "-b",
            "ollama",
            "-m",
            # "rag",
            "storm",
            # "direct",
            # "collaborative",
            "-oh",
            "http://localhost:11434",
            "-n",
            "20",
        ]

        print(f"Running main with args: {test_args}")

        # Import and run main with mocked sys.argv
        with patch.object(sys, "argv", ["main.py"] + test_args):
            from src.baselines.__main__ import main

            result = main()

        print(f"Main execution completed with result: {result}")
        # Assert that the result is as expected
        assert result == 0, "Main execution did not complete successfully"
