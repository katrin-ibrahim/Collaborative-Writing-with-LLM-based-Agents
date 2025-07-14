#!/usr/bin/env python3
"""
Main entry point for the analysis module.
Run with: python -m analysis
"""

import argparse
import sys
from pathlib import Path

from . import analyze_results


def main():
    """Main entry point for the analysis module."""
    parser = argparse.ArgumentParser(description="Run analysis on experiment results")
    parser.add_argument(
        "results_file",
        nargs="?",
        default="results.json",
        help="Path to results.json file (default: results.json)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="analysis_output",
        help="Output directory for analysis results (default: analysis_output)",
    )

    args = parser.parse_args()

    # Check if results file exists
    if not Path(args.results_file).exists():
        print(f"Error: {args.results_file} not found!")
        sys.exit(1)

    print(f"Running analysis on {args.results_file}...")
    print(f"Output will be saved to {args.output_dir}/")

    try:
        # Run the analysis
        results = analyze_results(args.results_file, args.output_dir)

        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)

        # Print summary
        data = results["data"]
        aggregated = results["aggregated"]
        charts = results["charts"]

        print(f"\nData Summary:")
        print(f"- Timestamp: {data.get('timestamp', 'N/A')}")
        print(f"- Total topics processed: {aggregated['summary']['total_topics']}")
        print(
            f"- Total successful results: {aggregated['summary']['total_successful']}"
        )
        print(f"- Methods: {', '.join(aggregated['summary']['methods'])}")

        print(f"\nGenerated Charts:")
        for chart_name, chart_path in charts.items():
            if chart_path:
                print(f"- {chart_name}: {chart_path}")

        print(f"\nDetailed results saved to: {args.output_dir}/")

    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
