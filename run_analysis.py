#!/usr/bin/env python3
"""
Script to run the analysis module on the results.json file.
"""

import sys
from pathlib import Path

import os

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from analysis import analyze_results


def main():
    """Run the complete analysis pipeline."""
    results_file = "results.json"
    output_dir = "analysis_output"

    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!")
        return

    print(f"Running analysis on {results_file}...")
    print(f"Output will be saved to {output_dir}/")

    try:
        # Run the analysis
        results = analyze_results(results_file, output_dir)

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


if __name__ == "__main__":
    main()
