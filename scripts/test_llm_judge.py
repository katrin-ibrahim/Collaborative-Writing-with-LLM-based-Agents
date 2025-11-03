"""
Test script for LLM Judge (SLURM version)

This script tests the llm_judge_slurm.py implementation to ensure:
1. Model loads correctly
2. Articles are evaluated
3. Output format is correct
4. All rubric criteria are scored
"""

import subprocess
import sys
from pathlib import Path

import json

TEST_ARTICLE = "results/ollama/3_methods_3n/articles/direct_2022_AFL_Grand_Final.md"
SCRIPT_PATH = "src/evaluation/llm_judge_slurm.py"
OUTPUT_DIR = Path("test_llm_judge_output")


def run_judge(article_path: str) -> dict:
    """
    Run the LLM judge on an article.

    Args:
        article_path: Path to article file

    Returns:
        Parsed JSON result
    """
    print(f"Running judge on: {article_path}")

    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, article_path],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        print(f"Error running judge:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Judge failed with exit code {result.returncode}")

    return json.loads(result.stdout)


def validate_result(result: dict) -> bool:
    """
    Validate that the result has the expected structure.

    Args:
        result: Result dictionary from judge

    Returns:
        True if valid, raises exception otherwise
    """
    required_fields = [
        "interest_level",
        "coherence_organization",
        "relevance_focus",
        "broad_coverage",
        "justification",
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing required field: {field}")

    # Check scores are in valid range
    score_fields = [f for f in required_fields if f != "justification"]
    for field in score_fields:
        score = result[field]
        if not isinstance(score, int) or score < 1 or score > 5:
            raise ValueError(f"Invalid score for {field}: {score} (must be 1-5)")

    # Check justification is a string
    if not isinstance(result["justification"], str):
        raise ValueError("Justification must be a string")

    return True


def main():
    """Run tests."""
    print("=" * 60)
    print("Testing LLM Judge (SLURM Version)")
    print("=" * 60)
    print()

    # Check test article exists
    article_path = Path(TEST_ARTICLE)
    if not article_path.exists():
        print(f"Error: Test article not found: {TEST_ARTICLE}")
        print("Please ensure you have generated articles first.")
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Test Configuration:")
    print(f"  Script: {SCRIPT_PATH}")
    print(f"  Test Article: {TEST_ARTICLE}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print()

    # Test 1: Single article evaluation
    print("Test 1: Evaluating single article...")
    print("-" * 60)

    try:
        results = run_judge(str(article_path))

        # Should return a list with one result
        if not isinstance(results, list) or len(results) != 1:
            raise ValueError(f"Expected list with 1 result, got: {type(results)}")

        result = results[0]

        # Check for errors in result
        if "error" in result:
            print(f"✗ Evaluation returned error: {result['error']}")
            if "raw_output" in result:
                print(f"  Raw output: {result['raw_output']}")
            sys.exit(1)

        # Validate result structure
        validate_result(result)

        # Save result
        output_file = OUTPUT_DIR / "single_article_result.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print("✓ Single article evaluation completed successfully")
        print(f"  Result saved to: {output_file}")
        print()
        print("  Evaluation Scores:")
        print(f"    Interest Level: {result['interest_level']}/5")
        print(f"    Coherence & Organization: {result['coherence_organization']}/5")
        print(f"    Relevance & Focus: {result['relevance_focus']}/5")
        print(f"    Broad Coverage: {result['broad_coverage']}/5")
        print()
        print(f"  Justification:")
        print(f"    {result['justification'][:200]}...")

    except subprocess.TimeoutExpired:
        print("✗ Test timed out after 600 seconds")
        print("  This may indicate:")
        print("  - Model download in progress (first run)")
        print("  - Insufficient GPU memory")
        print("  - Model loading issues")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print()
    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    print()
    print("Output files:")
    for file in OUTPUT_DIR.iterdir():
        print(f"  {file}")


if __name__ == "__main__":
    main()
