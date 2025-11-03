#!/bin/bash

# Test script for LLM Judge (SLURM version)
# This script tests the llm_judge_slurm.py with sample articles

set -e

echo "============================================"
echo "Testing LLM Judge (SLURM Version)"
echo "============================================"
echo ""

# Configuration
TEST_ARTICLE="results/ollama/3_methods_3n/articles/direct_2022_AFL_Grand_Final.md"
OUTPUT_DIR="test_llm_judge_output"
SCRIPT_PATH="src/evaluation/llm_judge_slurm.py"

# Check if test article exists
if [ ! -f "$TEST_ARTICLE" ]; then
    echo "Error: Test article not found: $TEST_ARTICLE"
    echo "Please ensure you have generated articles first."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Test Configuration:"
echo "  Script: $SCRIPT_PATH"
echo "  Test Article: $TEST_ARTICLE"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Test 1: Single article evaluation
echo "Test 1: Evaluating single article..."
echo "---------------------------------------"
python "$SCRIPT_PATH" "$TEST_ARTICLE" > "$OUTPUT_DIR/single_article_result.json" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Single article evaluation completed successfully"
    echo "  Result saved to: $OUTPUT_DIR/single_article_result.json"
    echo ""
    echo "  Sample output:"
    head -20 "$OUTPUT_DIR/single_article_result.json"
else
    echo "✗ Single article evaluation failed"
    echo "  Check logs in: $OUTPUT_DIR/single_article_result.json"
    exit 1
fi

echo ""
echo "============================================"
echo "All tests completed successfully!"
echo "============================================"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"
