#!/usr/bin/env python3
"""
Test script for local STORM runner.
Tests STORM with local models before SLURM deployment.
"""
import sys
from pathlib import Path

import logging

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from local.storm_runner import LocalSTORMRunner
from utils.logging_setup import setup_logging
from utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


def test_local_storm():
    """Test the local STORM implementation."""
    # Setup logging
    setup_logging("DEBUG")
    logger.info("🌪️ Testing Local STORM Runner")

    # Test topic
    topic = "Artificial Intelligence"

    try:
        # Check if models are available
        models_dir = Path("models")
        if not models_dir.exists():
            logger.error(
                "❌ No models directory found. Please ensure local models are available."
            )
            return False

        # Create output manager
        output_manager = OutputManager(base_dir="results/local_storm_test")

        # Initialize STORM runner
        logger.info("🔧 Initializing Local STORM Runner...")
        storm_runner = LocalSTORMRunner(
            model_path="models/", output_manager=output_manager, device="auto"
        )

        # Test STORM configuration (lightweight for testing)
        test_config = {
            "max_conv_turn": 1,  # Minimal for quick test
            "max_perspective": 1,  # Minimal for quick test
            "search_top_k": 2,  # Minimal for quick test
            "max_thread_num": 1,  # Single thread
        }

        # Run STORM
        logger.info(f"🌪️ Running STORM for: {topic}")
        article = storm_runner.run_storm(topic, storm_config=test_config)

        # Check results
        if article and article.content and len(article.content) > 100:
            logger.info("✅ Local STORM test successful!")
            logger.info(f"📄 Generated article: {len(article.content)} characters")
            logger.info(
                f"⏱️ Generation time: {article.metadata.get('generation_time', 0):.1f}s"
            )

            # Print a preview
            preview = (
                article.content[:300] + "..."
                if len(article.content) > 300
                else article.content
            )
            logger.info(f"📝 Preview:\n{preview}")

            return True
        else:
            logger.error("❌ Local STORM test failed - no content generated")
            return False

    except ImportError as e:
        if "knowledge_storm" in str(e):
            logger.error(
                "❌ knowledge-storm package not installed. Please install with:"
            )
            logger.error("   pip install knowledge-storm")
        elif "dspy" in str(e):
            logger.error("❌ dspy package not installed. Please install with:")
            logger.error("   pip install dspy-ai")
        else:
            logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Local STORM test failed: {e}")
        return False


def main():
    """Main test function."""
    success = test_local_storm()
    if success:
        print("\n🎉 Local STORM test completed successfully!")
        print("You can now develop and test STORM locally before SLURM deployment.")
    else:
        print("\n❌ Local STORM test failed. Check the logs above.")
        print("Make sure you have:")
        print("1. Local models in the 'models/' directory")
        print("2. knowledge-storm package installed")
        print("3. All dependencies satisfied")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
