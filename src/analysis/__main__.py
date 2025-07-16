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
        "results_dir", help="Path to results directory (must contain results.json)"
    )

    args = parser.parse_args()

    # Check if input is a directory and contains results.json
    input_path = Path(args.results_dir)
    if not input_path.is_dir():
        print(f"Error: {args.results_dir} is not a directory!")
        sys.exit(1)

    results_file = input_path / "results.json"
    if not results_file.exists():
        print(f"Error: {results_file} not found!")
        sys.exit(1)

    print(f"Running analysis on {results_file}...")

    # Output directory will be created inside the results directory
    output_dir = input_path / "analysis_output"
    print(f"Output will be saved to {output_dir}/")

    try:
        # Run the analysis
        results = analyze_results(args.results_dir)

        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)

        # Print summary
        data = results["data"]
        aggregated = results["aggregated"]
        charts = results["charts"]
        aggregated_file = results.get("aggregated_file", "")

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

        print(f"\nOutput Files:")
        if aggregated_file:
            print(f"- Aggregated metrics: {aggregated_file}")
        print(f"- Detailed results saved to: {output_dir}/")

    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
