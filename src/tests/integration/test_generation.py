"""
Integration test for stepping through baseline generation using main entry point.
Run with: pytest tests/integration/test_generation.py -s -v --pdb
"""

import sys
from unittest.mock import patch


class TestGenerationIntegration:
    """Step through actual baseline generation using main entry point."""

    # python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_localhost --retrieval_manager wiki --semantic_filtering true --experiment_name exp_001_n5__c-ollama_localhost__rm-wiki__sf_20-08_10:09
    def test_baselines(self):
        """Step through generation using main entry point - set breakpoints to debug."""
        # Mock command line arguments
        test_args = [
            "-b",
            "ollama",
            "-m",
            # "rag",
            # "storm",
            # "direct",
            # "writer_only",
            "writer_reviewer",
            "--model_config",
            "balanced_writer",
            "--retrieval_manager",
            "wiki",
            "-n",
            "1",
            "--semantic_filtering",
            "true",
            "--experiment_name",
            "test_writer",
            # "-r",
            # "/Users/katrin/Documents/Repos/Collaborative-Writing-with-LLM-based-Agents/results/ollama/exp_001_n5__c-ollama_localhost__rm-wiki__sf"
            # "-om",
            # "gemma3:1b",
        ]

        print(f"Running main with args: {test_args}")

        # Import and run main with mocked sys.argv
        with patch.object(sys, "argv", ["main.py"] + test_args):
            from src.main.__main__ import main

            result = main()

        print(f"Main execution completed with result: {result}")
        # Assert that the result is as expected
        assert result == 0, "Main execution did not complete successfully"
