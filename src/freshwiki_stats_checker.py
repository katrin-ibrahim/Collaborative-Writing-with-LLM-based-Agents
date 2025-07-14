#!/usr/bin/env python3
"""
Generate comprehensive FreshWiki dataset statistics for sharing and documentation.
"""

import sys
from datetime import datetime
from pathlib import Path

import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.freshwiki_loader import FreshWikiLoader


def generate_comprehensive_stats(loader: FreshWikiLoader, output_file: str = None):
    """Generate comprehensive dataset statistics."""
    stats = loader.get_stats()

    if stats["status"] != "loaded":
        print("❌ No FreshWiki data found!")
        return None

    # Get detailed statistics
    entries = loader.entries
    word_counts = [len(entry.reference_content.split()) for entry in entries]
    section_counts = [len(entry.reference_outline) for entry in entries]

    # Calculate percentiles
    import numpy as np

    word_percentiles = {
        "25th": int(np.percentile(word_counts, 25)),
        "50th": int(np.percentile(word_counts, 50)),
        "75th": int(np.percentile(word_counts, 75)),
        "90th": int(np.percentile(word_counts, 90)),
    }

    section_percentiles = {
        "25th": int(np.percentile(section_counts, 25)),
        "50th": int(np.percentile(section_counts, 50)),
        "75th": int(np.percentile(section_counts, 75)),
        "90th": int(np.percentile(section_counts, 90)),
    }

    # Analyze topic diversity instead of forced categorization
    topic_analysis = analyze_topic_diversity([entry.topic for entry in entries])

    # Get quality distribution
    quality_bins = {
        "short": len([w for w in word_counts if w < 1000]),
        "medium": len([w for w in word_counts if 1000 <= w < 2500]),
        "long": len([w for w in word_counts if w >= 2500]),
    }

    comprehensive_stats = {
        "dataset_info": {
            "generated_at": datetime.now().isoformat(),
            "total_entries": len(entries),
            "data_path": str(loader.data_path.resolve()),
            "quality_filtered": True,
            "extraction_source": "FreshWiki Repository",
        },
        "word_count_analysis": {
            "mean": round(np.mean(word_counts), 1),
            "median": int(np.median(word_counts)),
            "std_dev": round(np.std(word_counts), 1),
            "min": min(word_counts),
            "max": max(word_counts),
            "percentiles": word_percentiles,
            "distribution": quality_bins,
        },
        "section_analysis": {
            "mean": round(np.mean(section_counts), 1),
            "median": int(np.median(section_counts)),
            "std_dev": round(np.std(section_counts), 1),
            "min": min(section_counts),
            "max": max(section_counts),
            "percentiles": section_percentiles,
            "interpretation": {
                "percentile_meaning": "Distribution of article structure complexity - higher percentiles indicate more detailed articles",
                "evaluation_relevance": "Section count affects outline evaluation metrics (HSR) and content generation complexity",
                "25th_percentile": f'{section_percentiles["25th"]} sections - simpler articles with basic structure',
                "75th_percentile": f'{section_percentiles["75th"]} sections - detailed articles with comprehensive coverage',
                "diversity_assessment": f"Range of {min(section_counts)}-{max(section_counts)} sections provides varied complexity for evaluation",
            },
        },
        "topic_analysis": topic_analysis,
        "sample_topics": [
            {
                "topic": entry.topic,
                "word_count": len(entry.reference_content.split()),
                "sections": len(entry.reference_outline),
                "outline_preview": (
                    entry.reference_outline[:3]
                    if len(entry.reference_outline) > 3
                    else entry.reference_outline
                ),
            }
            for entry in entries[:10]
        ],
        "quality_assessment": {
            "suitable_for_evaluation": len(entries),
            "avg_content_depth": round(
                np.mean([len(e.reference_outline) for e in entries]), 1
            ),
            "content_completeness": "100%",  # All entries are quality-filtered
            "structural_diversity": len(set(len(e.reference_outline) for e in entries)),
        },
    }

    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comprehensive_stats, f, indent=2, ensure_ascii=False)
        print(f"📁 Statistics saved to: {output_path.resolve()}")

    return comprehensive_stats


def analyze_topic_diversity(topics):
    """Analyze topic diversity without forced categorization."""
    # Analyze topic structure patterns
    patterns = {
        "with_year": len(
            [t for t in topics if any(str(year) in t for year in range(2020, 2025))]
        ),
        "with_season": len(
            [
                t
                for t in topics
                if any(word in t.lower() for word in ["season", "series"])
            ]
        ),
        "event_based": len(
            [
                t
                for t in topics
                if any(
                    word in t.lower()
                    for word in [
                        "ceremony",
                        "festival",
                        "championship",
                        "olympics",
                        "cup",
                    ]
                )
            ]
        ),
        "person_biography": len(
            [
                t
                for t in topics
                if ("_" in t or "(" in t)
                and not any(str(year) in t for year in range(2020, 2025))
            ]
        ),
        "location_based": len(
            [
                t
                for t in topics
                if any(
                    word in t.lower()
                    for word in ["city", "bridge", "building", "airport", "station"]
                )
            ]
        ),
        "media_content": len(
            [
                t
                for t in topics
                if any(
                    word in t.lower()
                    for word in ["film", "album", "song", "book", "show"]
                )
            ]
        ),
    }

    # Calculate topic length distribution
    title_lengths = [len(topic) for topic in topics]
    import numpy as np

    return {
        "topic_patterns": patterns,
        "title_length_stats": {
            "mean_chars": round(np.mean(title_lengths), 1),
            "median_chars": int(np.median(title_lengths)),
            "shortest": min(title_lengths),
            "longest": max(title_lengths),
        },
        "topic_complexity": {
            "simple_titles": len([t for t in topics if len(t.split()) <= 3]),
            "complex_titles": len([t for t in topics if len(t.split()) > 6]),
            "parenthetical_info": len([t for t in topics if "(" in t]),
        },
        "diversity_note": "Topics analyzed by structural patterns rather than semantic categories to avoid misclassification",
    }


def print_stats_summary(stats):
    """Print a human-readable summary."""
    print("🔍 FreshWiki Dataset Statistics")
    print("=" * 50)

    info = stats["dataset_info"]
    word_stats = stats["word_count_analysis"]
    section_stats = stats["section_analysis"]

    print(f"📊 Dataset Overview:")
    print(f"  Total Topics: {info['total_entries']}")
    print(f"  Quality Filtered: {info['quality_filtered']}")
    print(f"  Generated: {info['generated_at'][:19]}")

    print(f"\n📝 Content Analysis:")
    print(f"  Word Count - Mean: {word_stats['mean']}, Median: {word_stats['median']}")
    print(f"  Word Range: {word_stats['min']} - {word_stats['max']} words")
    print(
        f"  Section Count - Mean: {section_stats['mean']}, Median: {section_stats['median']}"
    )
    print(f"  Section Range: {section_stats['min']} - {section_stats['max']} sections")
    print(
        f"    → 25th percentile: {section_stats['percentiles']['25th']} sections (simpler articles)"
    )
    print(
        f"    → 75th percentile: {section_stats['percentiles']['75th']} sections (detailed articles)"
    )

    print(f"\n🎯 Topic Structure Analysis:")
    patterns = stats["topic_analysis"]["topic_patterns"]
    complexity = stats["topic_analysis"]["topic_complexity"]
    for pattern, count in patterns.items():
        if count > 0:
            pct = round((count / info["total_entries"]) * 100, 1)
            pattern_name = pattern.replace("_", " ").title()
            print(f"  {pattern_name}: {count} ({pct}%)")

    print(f"\n📏 Title Complexity:")
    title_stats = stats["topic_analysis"]["title_length_stats"]
    print(f"  Simple titles (≤3 words): {complexity['simple_titles']}")
    print(f"  Complex titles (>6 words): {complexity['complex_titles']}")
    print(f"  With clarifying info: {complexity['parenthetical_info']}")
    print(f"  Average title length: {title_stats['mean_chars']} characters")

    print(f"\n✅ Quality Assessment:")
    qa = stats["quality_assessment"]
    print(f"  Evaluation Ready: {qa['suitable_for_evaluation']} topics")
    print(f"  Content Depth: {qa['avg_content_depth']} sections/topic")
    print(f"  Structural Diversity: {qa['structural_diversity']} unique section counts")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate FreshWiki dataset statistics"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/freshwiki_stats.json",
        help="Output file for statistics (default: data/freshwiki_stats.json)",
    )
    parser.add_argument(
        "--console-only",
        "-c",
        action="store_true",
        help="Only print to console, don't save file",
    )

    args = parser.parse_args()

    # Load data and generate stats
    loader = FreshWikiLoader()

    output_file = None if args.console_only else args.output
    stats = generate_comprehensive_stats(loader, output_file)

    if stats:
        print_stats_summary(stats)

        if not args.console_only:
            print(f"   Statistics file: {Path(args.output).resolve()}")
    else:
        print("❌ Could not generate statistics - no data found")


if __name__ == "__main__":
    main()
