# tests/unit/test_metrics.py
"""
Unit tests for metrics functions and integration test with evaluator main.

Structure:
1. Integration test: Run evaluation main on real results directory
2. Unit tests: Verify correctness of individual metric calculations
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation.evaluator import ArticleEvaluator

# Import metrics functions
from src.evaluation.metrics import (  # Core metric functions; Utility functions; Text preprocessing; Constants
    METRIC_DESCRIPTIONS,
    STORM_METRICS,
    calculate_all_rouge_metrics,
    calculate_composite_score,
    calculate_entity_recall,
    calculate_heading_entity_recall,
    calculate_heading_metrics,
    calculate_heading_soft_recall,
    calculate_rouge_1,
    calculate_rouge_l,
    evaluate_article_metrics,
    extract_entities,
    extract_headings_from_content,
    format_metrics_for_display,
    preprocess_text_for_rouge,
    validate_metrics,
)

# Import real data models
from src.utils.data_models import Article
from src.utils.freshwiki_loader import FreshWikiEntry


class TestIntegrationWithMain:
    """Integration test that runs the evaluation main module on a results directory."""

    def test_evaluation_main_with_results_dir(
        self,
    ):
        """Test running evaluation main module on a sample results directory."""
        # Create a sample results directory structure
        results_dir = Path(
            "/Users/katrin/Documents/Repos/Collaborative-Writing-with-LLM-based-Agents/results/ollama/storm_N=1_T=19.07_23:11"
        )

        """Step through generation using main entry point - set breakpoints to debug."""
        # Mock command line arguments
        test_args = [str(results_dir), "--force"]  # Path to the results directory

        print(f"Running main with args: {test_args}")

        # Import and run main with mocked sys.argv
        with patch.object(sys, "argv", ["main.py"] + test_args):
            from src.evaluation.__main__ import main

            result = main()

        print(f"Main execution completed with result: {result}")
        # Assert that the result is as expected
        assert result == 0, "Main execution did not complete successfully"


class TestROUGEMetricsCorrectness:
    """Test correctness of ROUGE metric calculations."""

    def test_rouge_1_exact_calculation(self):
        """Test ROUGE-1 calculation with known expected values."""
        # Test case with known overlap
        generated = "the quick brown fox jumps over lazy dog"
        reference = "the brown fox jumps over the lazy cat"

        # Expected: overlap = ["the", "brown", "fox", "jumps", "over", "lazy"] = 6 words
        # Reference has 8 words total
        # ROUGE-1 = 6/8 = 0.75
        score = calculate_rouge_1(generated, reference)
        assert abs(score - 0.75) < 0.01

    def test_rouge_1_no_overlap(self):
        """Test ROUGE-1 with completely different texts."""
        generated = "artificial intelligence machine learning"
        reference = "quantum computing blockchain technology"
        score = calculate_rouge_1(generated, reference)
        assert score == 0.0

    def test_rouge_1_perfect_match(self):
        """Test ROUGE-1 with identical texts."""
        text = "machine learning artificial intelligence"
        score = calculate_rouge_1(text, text)
        assert score == 1.0

    def test_rouge_l_longest_common_subsequence(self):
        """Test ROUGE-L calculation with known LCS."""
        # Generated: "A B C D E"
        # Reference: "A X B Y C"
        # LCS: "A B C" = 3 tokens
        # Reference length: 5 tokens
        # ROUGE-L = 3/5 = 0.6
        generated = "alpha beta gamma delta epsilon"
        reference = "alpha xenon beta yttrium gamma"

        score = calculate_rouge_l(generated, reference)
        assert abs(score - 0.6) < 0.01

    def test_rouge_l_identical_texts(self):
        """Test ROUGE-L with identical texts."""
        text = "the quick brown fox"
        score = calculate_rouge_l(text, text)
        assert score == 1.0

    def test_rouge_l_no_common_subsequence(self):
        """Test ROUGE-L with no common subsequence."""
        generated = "alpha beta gamma"
        reference = "delta epsilon zeta"
        score = calculate_rouge_l(generated, reference)
        assert score == 0.0

    def test_text_preprocessing_correctness(self):
        """Test that text preprocessing works as expected."""
        text = "Machine-Learning in 2023: A comprehensive study!"
        processed = preprocess_text_for_rouge(text)

        # Should convert to lowercase
        assert all(word.islower() or word.isdigit() for word in processed)

        # Should preserve years
        assert "2023" in processed

        # Should handle hyphens
        assert "machine_learning" in processed

        # Should filter short words
        assert "a" not in processed  # Too short

        # Should preserve meaningful words
        assert "comprehensive" in processed
        assert "study" in processed


class TestEntityMetricsCorrectness:
    """Test correctness of entity extraction and recall."""

    def test_entity_extraction_basic(self):
        """Test basic entity extraction functionality."""
        text = "John Smith works at Microsoft in Seattle."
        entities = extract_entities(text)

        # Should return a set
        assert isinstance(entities, set)

        # If FLAIR is available, should find entities; if not, should return empty set
        # Both are valid behaviors depending on setup

    def test_entity_recall_identical_texts(self):
        """Test entity recall with identical texts should be 1.0."""
        text = "Apple Inc. was founded by Steve Jobs in California."
        recall = calculate_entity_recall(text, text)
        assert recall == 1.0

    def test_entity_recall_no_reference_entities(self):
        """Test entity recall when reference has no entities."""
        generated = "Apple Inc. and Microsoft are technology companies."
        reference = "these are simple words without any named entities"

        recall = calculate_entity_recall(generated, reference)
        # When reference has no entities, recall should be 1.0
        assert recall == 1.0

    def test_entity_recall_empty_generated(self):
        """Test entity recall with empty generated text."""
        generated = ""
        reference = "Apple Inc. is based in California."

        recall = calculate_entity_recall(generated, reference)
        # Should be 0.0 (no entities found in empty text)
        assert recall == 0.0


class TestHeadingMetricsCorrectness:
    """Test correctness of heading extraction and similarity."""

    def test_heading_extraction_markdown(self):
        """Test extraction of markdown headings."""
        content = """
        # First Heading
        Some content here.

        ## Second Heading
        More content.

        ### Third Heading
        Even more content.
        """

        headings = extract_headings_from_content(content)

        assert "First Heading" in headings
        assert "Second Heading" in headings
        assert "Third Heading" in headings
        assert len(headings) == 3

    def test_heading_extraction_colon_format(self):
        """Test extraction of colon-terminated headings."""
        content = """
        Introduction:
        This is the introduction section.

        Methods:
        This describes the methodology.

        Regular sentence with colon: this should not be a heading.

        Results:
        The results are presented here.
        """

        headings = extract_headings_from_content(content)

        assert "Introduction" in headings
        assert "Methods" in headings
        assert "Results" in headings
        # Should not include the regular sentence
        assert "Regular sentence with colon" not in headings

    def test_heading_soft_recall_identical(self):
        """Test HSR with identical headings should be 1.0."""
        headings = ["Introduction", "Methods", "Results", "Conclusion"]
        hsr = calculate_heading_soft_recall(headings, headings)
        assert hsr == 1.0

    def test_heading_soft_recall_empty_inputs(self):
        """Test HSR with empty inputs."""
        assert calculate_heading_soft_recall([], ["ref1", "ref2"]) == 0.0
        assert calculate_heading_soft_recall(["gen1", "gen2"], []) == 0.0
        assert calculate_heading_soft_recall([], []) == 0.0

    def test_heading_entity_recall_empty_inputs(self):
        """Test HER with empty inputs."""
        assert calculate_heading_entity_recall([], ["ref1", "ref2"]) == 0.0
        assert calculate_heading_entity_recall(["gen1", "gen2"], []) == 0.0
        assert calculate_heading_entity_recall([], []) == 0.0


class TestCompleteEvaluationCorrectness:
    """Test correctness of complete article evaluation."""

    def test_evaluate_article_metrics_structure(self):
        """Test that complete evaluation returns correct structure."""
        article_content = """
        # Introduction to Machine Learning
        Machine learning is a subset of artificial intelligence.

        ## Applications
        ML has applications in healthcare and finance.
        """

        reference_content = """
        Machine learning represents a branch of artificial intelligence.
        Applications include healthcare and financial services.
        """

        reference_headings = ["Introduction", "Applications", "Conclusion"]

        results = evaluate_article_metrics(
            article_content, reference_content, reference_headings
        )

        # Check all STORM metrics are present
        for metric in STORM_METRICS:
            assert metric in results, f"Missing metric: {metric}"

        # Check values are in valid range (0-100)
        for metric, value in results.items():
            assert 0.0 <= value <= 100.0, f"Metric {metric} value {value} out of range"

        # Check that metrics make sense
        # Should have some ROUGE overlap due to similar content
        assert results["rouge_1"] > 0.0

        # Should have some heading similarity
        assert results["heading_soft_recall"] > 0.0

    def test_evaluate_article_metrics_empty_content(self):
        """Test evaluation with empty content."""
        results = evaluate_article_metrics("", "", [])

        # Should return all metrics with 0.0 values
        for metric in STORM_METRICS:
            assert metric in results
            assert results[metric] == 0.0


class TestUtilityFunctionsCorrectness:
    """Test correctness of utility functions."""

    def test_validate_metrics_valid_range(self):
        """Test metrics validation with values in valid range."""
        valid_metrics = {
            "rouge_1": 75.0,
            "rouge_l": 68.5,
            "heading_soft_recall": 82.3,
            "heading_entity_recall": 90.0,
            "article_entity_recall": 77.8,
        }

        assert validate_metrics(valid_metrics) is True

    def test_validate_metrics_invalid_range(self):
        """Test metrics validation with values outside valid range."""
        invalid_metrics = {
            "rouge_1": 105.0,  # > 100
            "rouge_l": -5.0,  # < 0
        }

        assert validate_metrics(invalid_metrics) is False

    def test_format_metrics_for_display_precision(self):
        """Test metric formatting with different precision levels."""
        metrics = {
            "rouge_1": 75.6789,
            "rouge_l": 68.1234,
        }

        formatted_2 = format_metrics_for_display(metrics, precision=2)
        assert formatted_2["rouge_1"] == "75.68%"
        assert formatted_2["rouge_l"] == "68.12%"

        formatted_1 = format_metrics_for_display(metrics, precision=1)
        assert formatted_1["rouge_1"] == "75.7%"
        assert formatted_1["rouge_l"] == "68.1%"

    def test_calculate_composite_score_equal_weights(self):
        """Test composite score calculation with equal weights."""
        metrics = {
            "rouge_1": 80.0,
            "rouge_l": 70.0,
            "heading_soft_recall": 90.0,
            "heading_entity_recall": 85.0,
            "article_entity_recall": 75.0,
        }

        score = calculate_composite_score(metrics)
        expected = (80.0 + 70.0 + 90.0 + 85.0 + 75.0) / 5
        assert abs(score - expected) < 0.01

    def test_calculate_composite_score_custom_weights(self):
        """Test composite score with custom weights."""
        metrics = {
            "rouge_1": 80.0,
            "rouge_l": 60.0,
        }

        weights = {
            "rouge_1": 0.7,  # 70% weight
            "rouge_l": 0.3,  # 30% weight
        }

        score = calculate_composite_score(metrics, weights)
        expected = (80.0 * 0.7 + 60.0 * 0.3) / (0.7 + 0.3)
        assert abs(score - expected) < 0.01


class TestEvaluatorIntegration:
    """Test ArticleEvaluator with real data models."""

    def test_evaluator_with_real_data_models(self):
        """Test evaluator using actual Article and FreshWikiEntry objects."""
        # Create real Article object
        article = Article(
            title="Artificial Intelligence Applications",
            content="""
            # Introduction to AI
            Artificial intelligence is transforming multiple industries.

            ## Healthcare Applications
            AI assists in medical diagnosis and drug discovery.

            ## Financial Services
            Machine learning powers fraud detection systems.
            """,
            metadata={"generation_time": 45.2, "word_count": 67},
        )

        # Create real FreshWikiEntry object
        reference = FreshWikiEntry(
            topic="Artificial Intelligence Applications",
            reference_content="""
            Artificial intelligence has widespread applications across industries.
            In healthcare, AI supports diagnostic processes and pharmaceutical research.
            Financial institutions use AI for fraud prevention and risk assessment.
            """,
            reference_outline=[
                "Introduction to Artificial Intelligence",
                "Healthcare Applications",
                "Financial Applications",
                "Future Developments",
            ],
            metadata={"source": "freshwiki", "quality": "high"},
        )

        # Test evaluation
        evaluator = ArticleEvaluator()
        results = evaluator.evaluate_article(article, reference)

        # Verify results structure and values
        for metric in STORM_METRICS:
            assert metric in results
            assert 0.0 <= results[metric] <= 100.0

        # Should have reasonable similarity due to overlapping content
        assert results["rouge_1"] > 10.0  # Some word overlap expected
        assert results["heading_soft_recall"] > 20.0  # Some heading similarity expected

        # Test summary generation
        summary = evaluator.get_evaluation_summary(results)
        assert "composite_score" in summary
        assert "overall_quality" in summary
        assert "%" in summary["composite_score"]


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
