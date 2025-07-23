#!/usr/bin/env python3
"""
Debug script to analyze entity extraction and metric calculations.
"""
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from evaluation.metrics.entity_metrics import EntityMetrics
from evaluation.metrics.heading_metrics import HeadingMetrics
from utils.freshwiki_loader import FreshWikiLoader


def debug_entity_extraction():
    """Debug entity extraction with detailed logging."""

    # Load a test case from your results
    freshwiki = FreshWikiLoader()

    # Try to get the Michelle Yeoh entry as an example
    entry = None
    for e in freshwiki.entries:
        if "Michelle_Yeoh" in e.topic or "Michelle Yeoh" in e.topic:
            entry = e
            break

    if not entry:
        print("❌ Could not find Michelle Yeoh entry. Using first available entry.")
        entry = freshwiki.entries[0] if freshwiki.entries else None

    if not entry:
        print("❌ No FreshWiki entries available!")
        return

    print(f"🔍 Debugging entity extraction for: {entry.topic}")
    print("=" * 80)

    # Load the corresponding generated article
    article_path = Path("articles") / f"storm_{entry.topic}.md"
    if not article_path.exists():
        # Try alternative naming
        normalized_name = entry.topic.replace("/", "_").replace(" ", "_")
        article_path = Path("articles") / f"storm_{normalized_name}.md"

    if article_path.exists():
        with open(article_path, "r") as f:
            generated_content = f.read()
    else:
        print(f"❌ Could not find generated article at {article_path}")
        # Use dummy content for testing
        generated_content = """# Michelle Yeoh

## Early Life
Michelle Yeoh was born in Malaysia.

## Career
She became a famous actress in Hong Kong cinema.

## Awards
Michelle won an Academy Award for Everything Everywhere All at Once.
"""

    print("\n📊 CONTENT ANALYSIS")
    print("-" * 40)
    print(f"Reference content length: {len(entry.reference_content)} chars")
    print(f"Generated content length: {len(generated_content)} chars")

    # Debug entity extraction
    entity_metrics = EntityMetrics()

    print("\n🔍 ENTITY EXTRACTION DEBUG")
    print("-" * 40)

    try:
        print("Extracting entities from REFERENCE content...")
        ref_entities = entity_metrics.extract_entities(entry.reference_content)
        print(
            f"✅ Reference entities ({len(ref_entities)}): {sorted(list(ref_entities))}"
        )

        print("\nExtracting entities from GENERATED content...")
        gen_entities = entity_metrics.extract_entities(generated_content)
        print(
            f"✅ Generated entities ({len(gen_entities)}): {sorted(list(gen_entities))}"
        )

        # Show intersection
        common = ref_entities.intersection(gen_entities)
        print(f"\n🎯 Common entities ({len(common)}): {sorted(list(common))}")

        # Calculate recall manually
        if ref_entities:
            recall = len(common) / len(ref_entities)
            print(
                f"📊 Manual AER calculation: {len(common)}/{len(ref_entities)} = {recall:.4f} ({recall*100:.2f}%)"
            )

        # Test the actual method
        print(f"\n🔧 Testing calculate_overall_entity_recall method...")
        aer_result = entity_metrics.calculate_overall_entity_recall(
            generated_content, entry.reference_content
        )
        print(f"📊 Method result: {aer_result:.4f} ({aer_result*100:.2f}%)")

    except Exception as e:
        print(f"❌ Entity extraction failed: {e}")
        import traceback

        traceback.print_exc()


def debug_heading_extraction():
    """Debug heading extraction and HER calculation."""

    print("\n\n🔍 HEADING EXTRACTION DEBUG")
    print("=" * 80)

    # Test data
    reference_headings = [
        "Early Life",
        "Career",
        "Personal Life",
        "Awards and Recognition",
    ]
    generated_content = """# Michelle Yeoh

## Early Life and Background
Michelle Yeoh was born in Malaysia.

## Acting Career
She became famous in Hong Kong cinema.

## Awards and Achievements
She won an Academy Award.
"""

    heading_metrics = HeadingMetrics()

    # Extract headings from generated content
    generated_headings = heading_metrics.extract_headings_from_content(
        generated_content
    )

    print(f"📋 Reference headings: {reference_headings}")
    print(f"📋 Generated headings: {generated_headings}")

    # Debug HSR calculation
    try:
        hsr = heading_metrics.calculate_heading_soft_recall(
            generated_headings, reference_headings
        )
        print(f"📊 Heading Soft Recall: {hsr:.4f} ({hsr*100:.2f}%)")
    except Exception as e:
        print(f"❌ HSR calculation failed: {e}")
        import traceback

        traceback.print_exc()

    # Debug HER calculation
    entity_metrics = EntityMetrics()

    try:
        print(f"\n🔍 Extracting entities from reference headings...")
        ref_heading_entities = set()
        for heading in reference_headings:
            entities = entity_metrics.extract_entities(heading)
            ref_heading_entities.update(entities)
            print(f"  '{heading}' -> {entities}")

        print(f"\n🔍 Extracting entities from generated headings...")
        gen_heading_entities = set()
        for heading in generated_headings:
            entities = entity_metrics.extract_entities(heading)
            gen_heading_entities.update(entities)
            print(f"  '{heading}' -> {entities}")

        print(f"\n📊 Reference heading entities: {sorted(list(ref_heading_entities))}")
        print(f"📊 Generated heading entities: {sorted(list(gen_heading_entities))}")

        common_heading_entities = ref_heading_entities.intersection(
            gen_heading_entities
        )
        print(f"🎯 Common heading entities: {sorted(list(common_heading_entities))}")

        if ref_heading_entities:
            her = len(common_heading_entities) / len(ref_heading_entities)
            print(
                f"📊 Manual HER calculation: {len(common_heading_entities)}/{len(ref_heading_entities)} = {her:.4f} ({her*100:.2f}%)"
            )

    except Exception as e:
        print(f"❌ HER calculation failed: {e}")
        import traceback

        traceback.print_exc()


def debug_flair_directly():
    """Test FLAIR NER directly to see what it extracts."""

    print("\n\n🔍 DIRECT FLAIR NER TEST")
    print("=" * 80)

    try:
        from flair.data import Sentence
        from flair.models import SequenceTagger

        # Load FLAIR model
        tagger = SequenceTagger.load("ner")
        print("✅ FLAIR NER model loaded successfully")

        # Test sentences
        test_sentences = [
            "Michelle Yeoh was born in Malaysia.",
            "She won an Academy Award for Everything Everywhere All at Once.",
            "Early Life and Background",
            "Awards and Recognition",
        ]

        for i, text in enumerate(test_sentences, 1):
            print(f"\n🔍 Test {i}: '{text}'")
            sentence = Sentence(text)
            tagger.predict(sentence)

            print(f"  Entities found:")
            for entity in sentence.get_spans("ner"):
                print(
                    f"    - '{entity.text}' ({entity.tag}, confidence: {entity.score:.3f})"
                )

            # Show what our extraction would get
            entities_over_threshold = [
                entity.text.lower().strip()
                for entity in sentence.get_spans("ner")
                if entity.score > 0.5
            ]
            print(f"  Our extraction (>0.5 conf): {entities_over_threshold}")

    except ImportError:
        print("❌ FLAIR not available")
    except Exception as e:
        print(f"❌ FLAIR test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🐛 ENTITY EXTRACTION DEBUG ANALYSIS")
    print("=" * 80)

    # Run all debug functions
    debug_flair_directly()
    debug_entity_extraction()
    debug_heading_extraction()

    print("\n\n✅ Debug analysis complete!")
    print("Check the output above to identify where the entity extraction is failing.")
