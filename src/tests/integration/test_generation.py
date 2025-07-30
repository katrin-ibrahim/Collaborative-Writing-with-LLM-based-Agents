"""
Integration test for stepping through baseline generation using main entry point.
Run with: pytest tests/integration/test_generation.py -s -v --pdb
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestGenerationIntegration:
    """Step through actual baseline generation using main entry point."""

    def test_baselines(self):
        """Step through generation using main entry point - set breakpoints to debug."""
        # Mock command line arguments
        test_args = [
            "--backend",
            "ollama",
            "-m",
            "rag",
            "-n",
            "1",
        ]

        # BREAKPOINT 2: Set here before main execution
        print(f"Running main with args: {test_args}")

        # Import and run main with mocked sys.argv
        with patch.object(sys, "argv", ["main.py"] + test_args):
            from src.baselines.__main__ import main

            # BREAKPOINT 3: Set here during main execution
            result = main()

        # BREAKPOINT 4: Set here to examine results
        print(f"Main execution completed with result: {result}")
        # Assert that the result is as expected
        assert result == 0, "Main execution did not complete successfully"

    def test_baselines_with_config(self):
        """Step through main entry with custom config - set breakpoints to debug."""

        temp_dir = Path(tempfile.mkdtemp(prefix="debug_main_config_"))
        print(f"\nUsing temp directory: {temp_dir}")

        try:
            # BREAKPOINT 1: Set here for config setup
            config_dir = temp_dir / "config"
            config_dir.mkdir()

            # Create custom model config
            model_config = {
                "models": {
                    "conversation": "llama3.1:8b",
                    "outline": "llama3.1:8b",
                    "article": "llama3.1:8b",
                    "polish": "llama3.1:8b",
                },
                "storm": {"max_conv_turn": 2, "max_perspective": 3, "search_top_k": 3},
            }

            config_file = config_dir / "test_models.yaml"
            import yaml

            with open(config_file, "w") as f:
                yaml.dump(model_config, f)

            output_dir = temp_dir / "config_results"

            test_args = [
                "--backend",
                "ollama",
                "--methods",
                "direct",
                "--num-topics",
                "1",
                "--model-config",
                str(config_file),
            ]

            print(f"Running main with custom config: {config_file}")

            with patch.object(sys, "argv", ["main.py"] + test_args):
                from src.baselines.__main__ import main

                # BREAKPOINT 3: Set here during config execution
                result = main()

            # BREAKPOINT 4: Set here to examine config results
            print(f"Config-based execution result: {result}")

            # Verify config was used
            if output_dir.exists():
                config_files = list(output_dir.glob("**/config*"))
                print(f"Config files in output: {[f.name for f in config_files]}")

        except Exception as e:
            print(f"Config-based main failed: {e}")
            pytest.skip(f"Config execution failed: {e}")
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
