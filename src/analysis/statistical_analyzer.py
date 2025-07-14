# src/analysis/statistical_analyzer.py
"""
Statistical analysis module for hypothesis testing and significance.
"""

import warnings

import logging
import numpy as np
from dataclasses import dataclass
from scipy import stats
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTest:
    """Results of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""


@dataclass
class MetricComparison:
    """Statistical comparison between two methods for one metric."""

    metric: str
    method1: str
    method2: str
    method1_stats: Dict[str, float]
    method2_stats: Dict[str, float]
    tests: List[StatisticalTest]
    recommendation: str


class StatisticalAnalyzer:
    """Perform statistical analysis on aggregated experimental results."""

    def __init__(self, aggregated_data: Dict[str, Any], alpha: float = 0.05):
        self.aggregated_data = aggregated_data
        self.alpha = alpha
        self.raw_aggregations = aggregated_data.get("raw_aggregations", {})

    def analyze_all(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        if not self.raw_aggregations:
            logger.warning("No raw aggregations found for statistical analysis")
            return {}

        methods = list(self.raw_aggregations.keys())

        analysis_results = {
            "overview": self._generate_overview(),
            "method_comparisons": {},
            "effect_sizes": {},
            "recommendations": {},
        }

        # Pairwise method comparisons
        if len(methods) >= 2:
            for i, method1 in enumerate(methods):
                for method2 in methods[i + 1 :]:
                    comparison_key = f"{method1}_vs_{method2}"
                    comparison = self._compare_methods_statistically(method1, method2)
                    analysis_results["method_comparisons"][comparison_key] = comparison

        # Overall effect sizes
        analysis_results["effect_sizes"] = self._calculate_effect_sizes()

        # Generate recommendations
        analysis_results["recommendations"] = self._generate_recommendations(
            analysis_results
        )

        return analysis_results

    def _generate_overview(self) -> Dict[str, Any]:
        """Generate statistical overview of the experiment."""
        methods = list(self.raw_aggregations.keys())

        overview = {
            "methods_analyzed": methods,
            "alpha_level": self.alpha,
            "sample_sizes": {},
            "power_analysis": {},
        }

        for method, agg in self.raw_aggregations.items():
            overview["sample_sizes"][method] = agg.topic_count

            # Basic power analysis (Cohen's conventions)
            n = agg.topic_count
            if n < 10:
                power_comment = (
                    "Very low sample size - results should be interpreted cautiously"
                )
            elif n < 30:
                power_comment = "Small sample size - limited statistical power"
            elif n < 100:
                power_comment = "Moderate sample size - reasonable statistical power"
            else:
                power_comment = "Large sample size - good statistical power"

            overview["power_analysis"][method] = {
                "sample_size": n,
                "power_assessment": power_comment,
            }

        return overview

    def _compare_methods_statistically(
        self, method1: str, method2: str
    ) -> Dict[str, Any]:
        """Perform statistical comparison between two methods."""
        if method1 not in self.raw_aggregations or method2 not in self.raw_aggregations:
            raise ValueError(f"Methods {method1} or {method2} not found")

        agg1 = self.raw_aggregations[method1]
        agg2 = self.raw_aggregations[method2]

        comparison = {
            "method1": method1,
            "method2": method2,
            "metric_comparisons": {},
            "overall_assessment": "",
        }

        significant_improvements = []
        significant_decreases = []

        # Compare each metric
        for metric in [
            "rouge_1",
            "rouge_2",
            "rouge_l",
            "heading_soft_recall",
            "heading_entity_recall",
            "article_entity_recall",
        ]:
            if metric in agg1.metrics and metric in agg2.metrics:
                metric_comp = self._compare_metric_statistically(
                    metric, method1, method2, agg1.metrics[metric], agg2.metrics[metric]
                )
                comparison["metric_comparisons"][metric] = metric_comp

                # Track significant changes
                for test in metric_comp.tests:
                    if test.significant:
                        if agg2.metrics[metric].mean > agg1.metrics[metric].mean:
                            significant_improvements.append(metric)
                        else:
                            significant_decreases.append(metric)

        # Overall assessment
        if significant_improvements and not significant_decreases:
            comparison["overall_assessment"] = (
                f"{method2} shows significant improvements over {method1}"
            )
        elif significant_decreases and not significant_improvements:
            comparison["overall_assessment"] = (
                f"{method2} shows significant decreases compared to {method1}"
            )
        elif significant_improvements and significant_decreases:
            comparison["overall_assessment"] = (
                f"Mixed results: {method2} shows both improvements and decreases compared to {method1}"
            )
        else:
            comparison["overall_assessment"] = (
                f"No significant differences found between {method1} and {method2}"
            )

        return comparison

    def _compare_metric_statistically(
        self, metric: str, method1: str, method2: str, stats1: Any, stats2: Any
    ) -> MetricComparison:
        """Perform statistical tests for a single metric comparison."""
        values1 = np.array(stats1.values)
        values2 = np.array(stats2.values)

        tests = []

        # Check normality (for small samples)
        if len(values1) >= 3 and len(values2) >= 3:
            # Shapiro-Wilk test for normality
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p1 = stats.shapiro(values1)
                    _, p2 = stats.shapiro(values2)
                normal_data = p1 > 0.05 and p2 > 0.05
            except:
                normal_data = True  # Assume normal if test fails
        else:
            normal_data = True  # Assume normal for very small samples

        # Choose appropriate test
        if normal_data and len(values1) >= 3 and len(values2) >= 3:
            # Independent t-test
            try:
                t_stat, p_val = stats.ttest_ind(values1, values2, equal_var=False)

                # Cohen's d effect size
                pooled_std = np.sqrt(
                    (
                        (len(values1) - 1) * np.var(values1, ddof=1)
                        + (len(values2) - 1) * np.var(values2, ddof=1)
                    )
                    / (len(values1) + len(values2) - 2)
                )
                cohens_d = (
                    (np.mean(values2) - np.mean(values1)) / pooled_std
                    if pooled_std > 0
                    else 0
                )

                # Confidence interval for difference in means
                diff_mean = np.mean(values2) - np.mean(values1)
                se_diff = np.sqrt(
                    np.var(values1, ddof=1) / len(values1)
                    + np.var(values2, ddof=1) / len(values2)
                )
                df = len(values1) + len(values2) - 2
                t_critical = stats.t.ppf(1 - self.alpha / 2, df)
                ci_lower = diff_mean - t_critical * se_diff
                ci_upper = diff_mean + t_critical * se_diff

                # Interpretation
                if abs(cohens_d) < 0.2:
                    effect_interp = "negligible effect"
                elif abs(cohens_d) < 0.5:
                    effect_interp = "small effect"
                elif abs(cohens_d) < 0.8:
                    effect_interp = "medium effect"
                else:
                    effect_interp = "large effect"

                tests.append(
                    StatisticalTest(
                        test_name="Independent t-test",
                        statistic=t_stat,
                        p_value=p_val,
                        significant=p_val < self.alpha,
                        effect_size=cohens_d,
                        confidence_interval=(ci_lower, ci_upper),
                        interpretation=f"Cohen's d = {cohens_d:.3f} ({effect_interp})",
                    )
                )
            except Exception as e:
                logger.warning(f"T-test failed for {metric}: {e}")

        # Mann-Whitney U test (non-parametric alternative)
        if len(values1) >= 3 and len(values2) >= 3:
            try:
                u_stat, p_val = stats.mannwhitneyu(
                    values1, values2, alternative="two-sided"
                )

                # Rank-biserial correlation as effect size
                n1, n2 = len(values1), len(values2)
                r = 1 - (2 * u_stat) / (n1 * n2)

                tests.append(
                    StatisticalTest(
                        test_name="Mann-Whitney U test",
                        statistic=u_stat,
                        p_value=p_val,
                        significant=p_val < self.alpha,
                        effect_size=r,
                        interpretation=f"Rank-biserial correlation = {r:.3f}",
                    )
                )
            except Exception as e:
                logger.warning(f"Mann-Whitney U test failed for {metric}: {e}")

        # Generate recommendation
        if not tests:
            recommendation = "Insufficient data for statistical testing"
        elif any(test.significant for test in tests):
            if stats2.mean > stats1.mean:
                recommendation = f"{method2} performs significantly better than {method1} on {metric}"
            else:
                recommendation = f"{method1} performs significantly better than {method2} on {metric}"
        else:
            recommendation = (
                f"No significant difference between {method1} and {method2} on {metric}"
            )

        return MetricComparison(
            metric=metric,
            method1=method1,
            method2=method2,
            method1_stats=stats1.to_dict(),
            method2_stats=stats2.to_dict(),
            tests=tests,
            recommendation=recommendation,
        )

    def _calculate_effect_sizes(self) -> Dict[str, Any]:
        """Calculate effect sizes across all metrics and methods."""
        if len(self.raw_aggregations) < 2:
            return {}

        methods = list(self.raw_aggregations.keys())
        effect_sizes = {}

        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1 :]:
                comparison_key = f"{method1}_vs_{method2}"
                effect_sizes[comparison_key] = {}

                agg1 = self.raw_aggregations[method1]
                agg2 = self.raw_aggregations[method2]

                for metric in [
                    "rouge_1",
                    "rouge_2",
                    "rouge_l",
                    "heading_soft_recall",
                    "heading_entity_recall",
                    "article_entity_recall",
                ]:
                    if metric in agg1.metrics and metric in agg2.metrics:
                        values1 = np.array(agg1.metrics[metric].values)
                        values2 = np.array(agg2.metrics[metric].values)

                        # Cohen's d
                        pooled_std = np.sqrt(
                            (
                                (len(values1) - 1) * np.var(values1, ddof=1)
                                + (len(values2) - 1) * np.var(values2, ddof=1)
                            )
                            / (len(values1) + len(values2) - 2)
                        )
                        cohens_d = (
                            (np.mean(values2) - np.mean(values1)) / pooled_std
                            if pooled_std > 0
                            else 0
                        )

                        # Hedge's g (bias-corrected)
                        j = 1 - (3 / (4 * (len(values1) + len(values2)) - 9))
                        hedges_g = cohens_d * j

                        effect_sizes[comparison_key][metric] = {
                            "cohens_d": cohens_d,
                            "hedges_g": hedges_g,
                            "interpretation": self._interpret_effect_size(
                                abs(cohens_d)
                            ),
                        }

        return effect_sizes

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude using Cohen's conventions."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"

    def _generate_recommendations(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate actionable recommendations based on statistical analysis."""
        recommendations = {
            "methodology": [],
            "future_research": [],
            "interpretation_caveats": [],
        }

        # Sample size recommendations
        overview = analysis_results.get("overview", {})
        sample_sizes = overview.get("sample_sizes", {})

        min_sample_size = min(sample_sizes.values()) if sample_sizes else 0
        if min_sample_size < 30:
            recommendations["methodology"].append(
                f"Consider increasing sample size (current min: {min_sample_size}). "
                "Samples of 30+ topics provide better statistical power."
            )

        # Method comparison recommendations
        comparisons = analysis_results.get("method_comparisons", {})
        significant_findings = []
        mixed_results = []

        for comp_key, comp_data in comparisons.items():
            assessment = comp_data.get("overall_assessment", "")
            if "significant improvements" in assessment:
                significant_findings.append(comp_data)
            elif "Mixed results" in assessment:
                mixed_results.append(comp_data)

        if significant_findings:
            recommendations["methodology"].append(
                "Strong evidence for method differences found. Consider focusing on the better-performing method."
            )
        elif mixed_results:
            recommendations["methodology"].append(
                "Mixed results suggest method performance may depend on specific metrics or contexts. "
                "Consider analyzing performance by topic characteristics."
            )
        else:
            recommendations["methodology"].append(
                "No significant differences found. Consider investigating: "
                "1) Different prompting strategies, 2) Larger sample sizes, 3) Alternative evaluation metrics."
            )

        # Effect size recommendations
        effect_sizes = analysis_results.get("effect_sizes", {})
        large_effects = []
        for comp_key, metrics in effect_sizes.items():
            for metric, effect_data in metrics.items():
                if effect_data.get("interpretation") == "large":
                    large_effects.append((comp_key, metric, effect_data["cohens_d"]))

        if large_effects:
            recommendations["future_research"].append(
                f"Large effect sizes found for: {', '.join([f'{metric} ({comp})' for comp, metric, _ in large_effects])}. "
                "These metrics show the most promise for detecting method differences."
            )

        # Interpretation caveats
        if min_sample_size < 10:
            recommendations["interpretation_caveats"].append(
                "Very small sample sizes limit generalizability of findings."
            )

        if len(sample_sizes) < 2:
            recommendations["interpretation_caveats"].append(
                "Only one method analyzed - no baseline comparison available."
            )

        recommendations["interpretation_caveats"].append(
            "Results are specific to the FreshWiki dataset and current model configuration. "
            "Validation on other datasets recommended."
        )

        return recommendations

    def get_significance_summary(self) -> Dict[str, Any]:
        """Get a summary of significant findings across all comparisons."""
        analysis = self.analyze_all()

        summary = {
            "significant_comparisons": [],
            "non_significant_comparisons": [],
            "metrics_with_differences": set(),
            "overall_conclusion": "",
        }

        comparisons = analysis.get("method_comparisons", {})

        for comp_key, comp_data in comparisons.items():
            has_significant = False
            significant_metrics = []

            for metric, metric_comp in comp_data.get("metric_comparisons", {}).items():
                for test in metric_comp.get("tests", []):
                    if test.get("significant", False):
                        has_significant = True
                        significant_metrics.append(metric)
                        summary["metrics_with_differences"].add(metric)

            if has_significant:
                summary["significant_comparisons"].append(
                    {
                        "comparison": comp_key,
                        "significant_metrics": significant_metrics,
                        "assessment": comp_data.get("overall_assessment", ""),
                    }
                )
            else:
                summary["non_significant_comparisons"].append(comp_key)

        # Overall conclusion
        if summary["significant_comparisons"]:
            summary["overall_conclusion"] = (
                f"Found significant differences in {len(summary['significant_comparisons'])} "
                f"method comparison(s) across {len(summary['metrics_with_differences'])} metric(s)."
            )
        else:
            summary["overall_conclusion"] = (
                "No significant differences found between methods."
            )

        return summary
