#!/usr/bin/env python3
"""
Direct prompting baseline experiment script.
This baseline uses a simple direct prompt approach without any agent structure.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the src directory to the path so we can import from there
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src"))

from workflows.direct_prompting import DirectPromptingWorkflow
from evaluation.evaluator import Evaluator
from utils.config import load_config


def main():
    """Run the direct prompting baseline experiment."""
    parser = argparse.ArgumentParser(description="Run direct prompting baseline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/baseline_configs/direct_prompting.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="experiments/results/direct_prompting",
        help="Directory to save results"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the workflow
    workflow = DirectPromptingWorkflow(config)
    
    # Initialize the evaluator
    evaluator = Evaluator(config)
    
    # Run experiments
    print("Running direct prompting baseline experiments...")
    results = workflow.run()
    
    # Evaluate results
    print("Evaluating results...")
    evaluation_results = evaluator.evaluate(results)
    
    # Save results
    print(f"Saving results to {args.output_dir}")
    evaluator.save_results(evaluation_results, args.output_dir)
    
    print("Direct prompting baseline experiments complete!")


if __name__ == "__main__":
    main()
