# src/methods/writer_reviewer_v2_method.py

import time

import logging

from src.collaborative.agents.reviewer_v3 import ReviewerV3
from src.collaborative.agents.writer_v4 import WriterV4
from src.collaborative.memory.convergence import ConvergenceChecker
from src.collaborative.memory.memory import SharedMemory
from src.config.config_context import ConfigContext
from src.methods.base_method import BaseMethod
from src.utils.data import Article
from src.utils.text_processing import remove_citation_tags

logger = logging.getLogger(__name__)


class WriterReviewerV2Method(BaseMethod):
    """
    Simplified collaborative method using WriterV2 and ReviewerV2 agents.

    Args:
        tom_enabled: Whether to enable Theory of Mind for inter-agent reasoning

    Workflow:
    1. WriterV2 creates initial draft with research
    2. ReviewerV2 provides fact-checked feedback
    3. WriterV2 revises based on feedback
    4. Repeat until convergence or max iterations
    """

    def __init__(self, tom_enabled: bool = False):
        super().__init__()

        # Initialize collaboration config
        self.collaboration_config = ConfigContext.get_collaboration_config()
        if self.collaboration_config is None:
            raise RuntimeError(
                "ConvergenceChecker: collaboration_config is None. "
                "Ensure ConfigContext is properly initialized before using ConvergenceChecker."
            )

        # Extract convergence parameters from config
        self.resolution_rate_threshold = getattr(
            self.collaboration_config, "resolution_rate_threshold", 0.95
        )
        self.max_iterations = getattr(self.collaboration_config, "max_iterations", 5)
        self.min_iterations = getattr(self.collaboration_config, "min_iterations", 1)
        self.stall_tolerance = getattr(self.collaboration_config, "stall_tolerance", 2)
        self.min_improvement = getattr(
            self.collaboration_config, "min_improvement", 0.01
        )

        self.priority_weights = {"high": 3, "medium": 2, "low": 1}
        self.small_tail_max = getattr(self.collaboration_config, "small_tail_max", 5)

        # Adaptive iteration parameters
        self.adaptive_iterations = getattr(
            self.collaboration_config, "adaptive_iterations", False
        )
        self.adaptive_extension_max = getattr(
            self.collaboration_config, "adaptive_extension_max", self.max_iterations
        )
        self.adaptive_improvement_threshold = getattr(
            self.collaboration_config, "adaptive_improvement_threshold", 0.02
        )
        self.adaptive_check_iteration = getattr(
            self.collaboration_config, "adaptive_check_iteration", 2
        )

        # Instantiate convergence checker with required arguments
        self.convergence_checker = ConvergenceChecker(
            max_iterations=self.max_iterations,
            resolution_rate_threshold=self.resolution_rate_threshold,
            min_iterations=self.min_iterations,
            stall_tolerance=self.stall_tolerance,
            min_improvement=self.min_improvement,
        )

        self.tom_enabled = tom_enabled
        self.shared_memory: SharedMemory = None  # type: ignore

    def run(self, topic: str) -> Article:
        """
        Generate article using WriterV2-ReviewerV2 collaboration.

        Args:
            topic: Topic to write about

        Returns:
            Final article after collaboration with comprehensive metadata
        """
        logger.info(f"Running WriterV2-ReviewerV2 collaboration for: {topic}")
        logger.info(f"Theory of Mind enabled: {self.tom_enabled}")

        # Reset usage counters at start
        task_models = self._get_task_models_for_method()
        self._reset_all_client_usage(task_models)

        start_time = time.time()
        writer_time = 0
        reviewer_time = 0

        try:
            # Get experiment output directory from ConfigContext
            output_dir = ConfigContext.get_output_dir()

            # Determine experiment name based on ToM setting
            experiment_name = (
                "writer_reviewer_tom" if self.tom_enabled else "writer_reviewer"
            )

            # Initialize shared memory with Theory of Mind support
            memory = SharedMemory(
                topic=topic,
                storage_dir=output_dir,  # Store memory files in experiment dir
                tom_enabled=self.tom_enabled,
                experiment_name=experiment_name,
            )

            # Set memory instance for tools and agents to access
            ConfigContext.set_memory_instance(memory)
            self.shared_memory = memory
            logger.info(
                f"Initialized shared memory for collaboration with key: {memory.session_id}"
            )

            # Initialize V3 agents
            writer = WriterV4()
            reviewer = ReviewerV3()

            # Check if we have an existing draft, otherwise create initial draft
            current_draft = memory.get_current_draft()
            if not current_draft:
                logger.info(f"WriterV2 creating initial draft for: {topic}")
                writer_start = time.time()

                # Initialize article state in memory
                seed_article = Article(
                    title=topic,
                    content=f"# {topic}\n\n",
                    sections={},
                    metadata=memory.state.get("metadata", {}).copy(),
                )
                memory.update_article_state(seed_article)

            # Collaborative improvement loop - let convergence checker handle all termination logic
            resolution_progress_history = []  # track resolution rate per iteration
            while True:
                # WriterV2 processes (initial draft or revision based on iteration)
                try:
                    current_iteration = memory.get_iteration()
                    if current_iteration == 0:
                        logger.info("WriterV2 creating initial draft")
                    else:
                        logger.info("WriterV2 addressing feedback and revising")

                    writer_start = time.time()
                    writer.process()
                    writer_time += time.time() - writer_start
                    current_draft = memory.get_current_draft()

                    if not current_draft:
                        raise RuntimeError("WriterV2 failed to create draft")

                    # Update ToM belief state: Reviewer observes Writer's behavior
                    if self.tom_enabled and memory.tom_module.enabled:
                        self._update_tom_after_writer(memory, current_iteration)

                    if current_iteration == 0:
                        logger.info(
                            f"WriterV2 completed initial draft: {len(current_draft)} characters"
                        )
                    else:
                        logger.info(
                            f"WriterV2 completed revision: {len(current_draft)} characters"
                        )

                    # ReviewerV2 analyzes and provides feedback
                    logger.info("ReviewerV2 analyzing article and generating feedback")
                    reviewer_start = time.time()

                    reviewer.process()
                    reviewer_time += time.time() - reviewer_start
                    logger.info("ReviewerV2 completed analysis and feedback generation")

                    # Update ToM belief state: Writer observes Reviewer's behavior
                    if self.tom_enabled and memory.tom_module.enabled:
                        self._update_tom_after_reviewer(memory)
                except Exception as e:
                    current_iteration = memory.get_iteration()
                    logger.error(f"V2 failed in iteration {current_iteration}: {e}")
                    raise RuntimeError(f"Workflow failed: {e}") from e

                # Check for convergence BEFORE incrementing iteration
                logger.info("Checking convergence criteria")

                # Collect ALL feedback items across ALL iterations
                all_items = []
                pending_items = []

                for iter_num in range(current_iteration + 1):
                    items_by_section = (
                        self.shared_memory.get_feedback_items_for_iteration(
                            iteration=iter_num
                        )
                    )

                    for _, items in items_by_section.items():
                        for item in items:
                            all_items.append(item)

                            status = getattr(item, "status", None)
                            if status is None and isinstance(item, dict):
                                status = item.get("status")

                            status_str = (
                                status.value
                                if (status and hasattr(status, "value"))
                                else status
                            ) or "pending"
                            if status_str.lower() != "verified_addressed":
                                pending_items.append(item)

                logger.info(
                    f"Convergence check: {len(pending_items)} pending items out of {len(all_items)} total across {current_iteration + 1} iterations"
                )

                # Calculate current resolution rate and store
                current_resolution_rate = self.convergence_checker.resolution_rate(
                    pending_items, all_items
                )
                resolution_progress_history.append(current_resolution_rate)
                if current_iteration > 0:
                    prev_rate = resolution_progress_history[current_iteration - 1]
                    improvement = current_resolution_rate - prev_rate
                    logger.info(
                        f"Resolution rate: {current_resolution_rate:.3f} (Δ {improvement:+.3f} from iteration {current_iteration - 1})"
                    )
                else:
                    logger.info(
                        f"Resolution rate: {current_resolution_rate:.3f} (initial)"
                    )

                converged, convergence_reason = (
                    self.convergence_checker.check_convergence(
                        iteration=current_iteration,
                        pending_items=pending_items,
                        all_items=all_items,
                    )
                )
                if converged:
                    logger.info(f"Convergence achieved: {convergence_reason}")
                    break

                # Adaptive extension check (only once at specified iteration)
                if (
                    self.adaptive_iterations
                    and current_iteration == self.adaptive_check_iteration
                    and self.max_iterations < self.adaptive_extension_max
                    and current_iteration > 0
                ):
                    prev_rate = resolution_progress_history[current_iteration - 1]
                    improvement = current_resolution_rate - prev_rate
                    if improvement >= self.adaptive_improvement_threshold:
                        old_max = self.max_iterations
                        self.max_iterations = self.adaptive_extension_max
                        self.convergence_checker.max_iterations = self.max_iterations
                        logger.info(
                            f"Adaptive extension triggered: improvement {improvement:.3f} >= threshold {self.adaptive_improvement_threshold:.3f}. max_iterations {old_max} -> {self.max_iterations}"
                        )
                    else:
                        logger.info(
                            f"Adaptive extension skipped: improvement {improvement:.3f} < threshold {self.adaptive_improvement_threshold:.3f}"
                        )

                # Move to next iteration for writer revision
                memory.next_iteration()
                writer_start = time.time()

            # Prepare final article with comprehensive metadata
            final_article = memory.get_current_draft_as_article()
            if not final_article:
                raise RuntimeError("No final article available")

            # Enhanced metadata for WriterV2-ReviewerV2 method
            total_time = time.time() - start_time

            # Collect token usage statistics
            token_usage = self._collect_token_usage(task_models)

            # Get collaboration metrics
            collab_metrics = self._calculate_collaboration_metrics(
                memory, writer_time, reviewer_time, total_time
            )

            # Enhanced metadata
            final_iterations = memory.get_iteration()
            final_article.metadata.update(
                {
                    # Standard metadata fields expected by output_manager
                    "generation_time": total_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    # Method-specific metadata
                    "method": "writer_reviewer_v2",
                    "tom_enabled": self.tom_enabled,
                    "total_iterations": final_iterations,
                    "execution_time": {
                        "total_time": total_time,
                        "writer_time": writer_time,
                        "reviewer_time": reviewer_time,
                        "overhead_time": total_time - writer_time - reviewer_time,
                    },
                    "collaboration_metrics": collab_metrics,
                    "theory_of_mind": (
                        self._get_tom_metrics(memory) if self.tom_enabled else None
                    ),
                    "token_usage": token_usage,
                }
            )

            logger.info("WriterV2-ReviewerV2 collaboration completed successfully")
            logger.info(
                f"Final article: {len(final_article.content)} characters, {len(final_article.sections)} sections"
            )
            logger.info(
                f"Total time: {total_time:.2f}s, Iterations: {final_iterations}, Tokens: {token_usage['total_tokens']}"
            )

            # Log ToM statistics if enabled
            if self.tom_enabled and memory.tom_module.enabled:
                tom_metrics = final_article.metadata.get("theory_of_mind", {})
                prediction_stats = tom_metrics.get("prediction_stats", {})
                quality_metrics = tom_metrics.get("prediction_quality", {})

                logger.info("=" * 60)
                logger.info("THEORY OF MIND STATISTICS")
                logger.info("=" * 60)
                logger.info(
                    f"Total predictions made: {prediction_stats.get('total_predictions', 0)}"
                )
                logger.info(
                    f"Accurate predictions: {prediction_stats.get('correct_predictions', 0)}"
                )
                logger.info(
                    f"Prediction accuracy rate: {prediction_stats.get('accuracy_rate', 0.0):.1%}"
                )

                quality_indicators = quality_metrics.get("quality_indicators", {})
                logger.info(
                    f"Average confidence: {quality_indicators.get('avg_confidence', 0.0):.3f}"
                )
                logger.info(
                    f"Confidence std dev: {quality_indicators.get('std_confidence', 0.0):.3f}"
                )
                logger.info(
                    f"Confidence range: {quality_indicators.get('min_confidence', 0.0):.3f} - {quality_indicators.get('max_confidence', 0.0):.3f}"
                )

                action_diversity = quality_metrics.get("action_diversity", {})
                logger.info(
                    f"Unique predicted actions: {action_diversity.get('unique_actions', 0)}"
                )
                logger.info(
                    f"Action diversity score: {action_diversity.get('diversity_score', 0.0):.3f}"
                )

                conf_dist = quality_metrics.get("confidence_distribution", {})
                logger.info("Confidence distribution:")
                for bucket, count in conf_dist.items():
                    logger.info(f"  {bucket}: {count} predictions")

                # Log behavioral impact metrics
                behavioral_impact = tom_metrics.get("behavioral_impact", {})
                if behavioral_impact:
                    logger.info("=" * 60)
                    logger.info("BEHAVIORAL IMPACT ANALYSIS")
                    logger.info("=" * 60)
                    logger.info(
                        f"Feedback volumes per iteration: {behavioral_impact.get('feedback_volumes', [])}"
                    )
                    logger.info(
                        f"ToM confidence progression: {[f'{c:.2f}' for c in behavioral_impact.get('tom_confidence_progression', [])]}"
                    )
                    logger.info(
                        f"Acceptance rate progression: {[f'{r:.2%}' for r in behavioral_impact.get('acceptance_rate_progression', [])]}"
                    )
                    logger.info(
                        f"Confidence-feedback correlation: {behavioral_impact.get('confidence_feedback_correlation', 0.0):.3f}"
                    )
                    logger.info(
                        f"Behavioral adaptation detected: {behavioral_impact.get('behavioral_adaptation_detected', False)}"
                    )

                logger.info("=" * 60)

            # Post-processing: Remove citation tags before final output
            final_article.content = remove_citation_tags(final_article.content)

            return final_article

        except Exception as e:
            logger.error(
                f"WriterV2-ReviewerV2 collaboration failed: {e}", exc_info=True
            )
            raise RuntimeError(f"WriterReviewerV2Method failed: {e}") from e

    def _calculate_collaboration_metrics(
        self,
        memory: SharedMemory,
        writer_time: float,
        reviewer_time: float,
        total_time: float,
    ) -> dict:
        """Calculate comprehensive collaboration metrics."""

        # Collect all feedback items from all iterations
        feedback_by_iteration = {}
        all_feedback = []
        max_iter = memory.get_iteration()
        for i in range(max_iter + 1):
            iteration_count = 0
            items_by_section = memory.get_feedback_items_for_iteration(i)
            for items in items_by_section.values():
                iteration_count += len(items)
                all_feedback.extend(items)

            # Store the final count for this iteration
            feedback_by_iteration[str(i)] = iteration_count

        # Calculate feedback metrics
        def get_status(item):
            status = getattr(item, "status", None)
            if status is None and isinstance(item, dict):
                status = item.get("status")
            if status is None:
                return None
            return status.value if hasattr(status, "value") else status

        total_feedback = len(all_feedback)
        addressed_feedback = len(
            [f for f in all_feedback if get_status(f) == "verified_addressed"]
        )

        # Count *everything* not verified as pending ---
        pending_feedback = len(
            [f for f in all_feedback if get_status(f) != "verified_addressed"]
        )

        # Calculate convergence metrics using existing checker
        convergence_metrics = {}
        try:
            iteration = memory.get_iteration()
            # Use the same logic for "pending" above
            pending_items = [
                f for f in all_feedback if get_status(f) != "verified_addressed"
            ]
            all_items = all_feedback
            converged, reason = self.convergence_checker.check_convergence(
                iteration,
                pending_items,
                all_items,
            )
            # Calculate weighted resolution rate as convergence_score
            convergence_score = self.convergence_checker.resolution_rate(
                pending_items, all_items
            )
            convergence_metrics = {
                "feedback_addressed_percentage": convergence_score,
                "convergence_score": convergence_score,
                "convergence_criteria_met": converged,
                "convergence_reason": reason,
            }
        except Exception as e:
            logger.warning(f"Could not calculate convergence metrics: {e}")

        return {
            "total_feedback_items": total_feedback,
            "addressed_feedback": addressed_feedback,
            "pending_feedback": pending_feedback,
            "feedback_resolution_rate": (
                addressed_feedback / total_feedback if total_feedback > 0 else 0
            ),
            "feedback_by_iteration": feedback_by_iteration,
            "average_feedback_per_iteration": total_feedback
            / max(1, len(feedback_by_iteration)),
            "time_efficiency": {
                "writer_time_percentage": (
                    writer_time / total_time if total_time > 0 else 0
                ),
                "reviewer_time_percentage": (
                    reviewer_time / total_time if total_time > 0 else 0
                ),
                "seconds_per_iteration": total_time / max(1, memory.get_iteration()),
            },
            "convergence": convergence_metrics,
        }

    def _get_tom_metrics(self, memory: SharedMemory) -> dict:
        """Get Theory of Mind metrics if available."""
        if not memory.tom_module or not memory.tom_module.enabled:
            return {}

        try:
            # Use the ToM module's built-in stats methods
            prediction_stats = memory.tom_module.get_prediction_accuracy_stats()
            prediction_quality = memory.tom_module.get_prediction_quality_metrics()

            # Calculate behavioral impact metrics
            behavioral_impact = self._calculate_behavioral_impact_metrics(memory)

            return {
                "prediction_stats": prediction_stats,
                "prediction_quality": prediction_quality,
                "behavioral_impact": behavioral_impact,
            }

        except Exception as e:
            logger.warning(f"Could not calculate ToM metrics: {e}")
            return {"error": str(e)}

    def _calculate_behavioral_impact_metrics(self, memory: SharedMemory) -> dict:
        """Calculate metrics to assess if ToM predictions actually influence agent behavior."""
        try:
            # Track feedback volume per iteration with ToM predictions
            feedback_volumes_by_iteration = []
            tom_confidence_by_iteration = []
            acceptance_rates_by_iteration = []

            max_iter = memory.get_iteration()

            for i in range(max_iter + 1):
                # Get feedback volume
                items_by_section = memory.get_feedback_items_for_iteration(i)
                all_items = []
                for items in items_by_section.values():
                    all_items.extend(items)
                feedback_volumes_by_iteration.append(len(all_items))

                # Get ToM prediction confidence for this iteration
                iteration_predictions = [
                    p
                    for p in memory.tom_module.prediction_history
                    if getattr(p, "iteration", None) == i
                ]
                avg_confidence = (
                    sum(p.confidence for p in iteration_predictions)
                    / len(iteration_predictions)
                    if iteration_predictions
                    else 0.0
                )
                tom_confidence_by_iteration.append(avg_confidence)

                # Calculate acceptance rate if iteration > 0
                if i > 0 and all_items:
                    addressed_count = sum(
                        1
                        for item in all_items
                        if getattr(item, "status", None)
                        and getattr(item, "status").value
                        in ["addressed", "verified_addressed"]
                    )
                    acceptance_rates_by_iteration.append(
                        addressed_count / len(all_items)
                    )
                elif i > 0:
                    acceptance_rates_by_iteration.append(0.0)

            # Calculate correlation metrics
            behavioral_adaptation_detected = False
            confidence_feedback_correlation = 0.0

            if (
                len(tom_confidence_by_iteration) > 2
                and len(feedback_volumes_by_iteration) > 2
            ):
                # Check if high confidence correlates with adjusted feedback volume
                import numpy as np

                if len(tom_confidence_by_iteration) == len(
                    feedback_volumes_by_iteration
                ):
                    try:
                        correlation = np.corrcoef(
                            tom_confidence_by_iteration, feedback_volumes_by_iteration
                        )[0, 1]
                        confidence_feedback_correlation = (
                            float(correlation) if not np.isnan(correlation) else 0.0
                        )
                    except Exception:
                        pass

                # Detect behavioral adaptation: if acceptance rate changes align with ToM predictions
                if len(acceptance_rates_by_iteration) > 1:
                    # Check if acceptance rates converge over time (sign of adaptation)
                    rate_changes = [
                        abs(
                            acceptance_rates_by_iteration[i + 1]
                            - acceptance_rates_by_iteration[i]
                        )
                        for i in range(len(acceptance_rates_by_iteration) - 1)
                    ]
                    if rate_changes:
                        avg_change = sum(rate_changes) / len(rate_changes)
                        # If changes are decreasing (stabilizing), adaptation likely occurred
                        behavioral_adaptation_detected = (
                            avg_change < 0.2 and len(rate_changes) > 1
                        )

            return {
                "feedback_volumes": feedback_volumes_by_iteration,
                "tom_confidence_progression": tom_confidence_by_iteration,
                "acceptance_rate_progression": acceptance_rates_by_iteration,
                "confidence_feedback_correlation": confidence_feedback_correlation,
                "behavioral_adaptation_detected": behavioral_adaptation_detected,
                "total_iterations": max_iter + 1,
            }

        except Exception as e:
            logger.warning(f"Could not calculate behavioral impact metrics: {e}")
            return {}

    def _update_tom_after_writer(self, memory: SharedMemory, iteration: int) -> None:
        """Calculate writer's observable action from feedback acceptance rate and store it."""
        if not memory.tom_module or not memory.tom_module.enabled:
            return

        try:
            from src.collaborative.tom.theory_of_mind import AgentRole, WriterAction

            if iteration == 0:
                logger.info(
                    "ToM: Skipping writer evaluation for iteration 0 (no feedback yet)"
                )
                return

            # Get feedback from previous iteration
            items_by_section = memory.get_feedback_items_for_iteration(iteration - 1)
            all_items = []
            for items in items_by_section.values():
                all_items.extend(items)

            if not all_items:
                logger.info(f"ToM: No feedback items from iteration {iteration-1}")
                return

            # Calculate actual acceptance rate from addressed feedback
            addressed_count = sum(
                1
                for item in all_items
                if getattr(item, "status", None)
                and str(getattr(item, "status")).lower() == "addressed"
            )
            acceptance_rate = addressed_count / len(all_items)

            # Map acceptance rate to WriterAction enum
            if acceptance_rate >= 0.7:
                observed_action = WriterAction.ACCEPT_MOST.value
            elif acceptance_rate >= 0.4:
                observed_action = WriterAction.PARTIALLY_ACCEPT.value
            else:
                observed_action = WriterAction.CONTEST_SOME.value

            # Append structured observation to history
            try:
                memory.append_tom_observation(
                    agent="writer",
                    observed_action=observed_action,
                    iteration=iteration,
                    details={
                        "acceptance_rate": acceptance_rate,
                        "addressed_count": addressed_count,
                        "total_feedback": len(all_items),
                    },
                )
            except Exception as obs_exc:
                logger.warning(f"ToM: Failed to append writer observation: {obs_exc}")

            logger.info(
                f"ToM: Writer action observed - {observed_action} "
                f"(acceptance rate: {acceptance_rate:.2%}, "
                f"{addressed_count}/{len(all_items)} addressed)"
            )

            # Evaluate pending predictions about writer
            recent_predictions = [
                p
                for p in memory.tom_module.prediction_history
                if p.predictor_role == AgentRole.REVIEWER
                and p.target_role == AgentRole.WRITER
                and p.accuracy.value == "unknown"
            ]

            for pred in recent_predictions:
                memory.tom_module.evaluate_prediction_accuracy(
                    prediction_id=pred.id, actual_outcome=observed_action
                )

        except Exception as e:
            logger.warning(f"Failed to update ToM after writer: {e}")

    def _update_tom_after_reviewer(self, memory: SharedMemory) -> None:
        """Calculate reviewer's observable action from feedback type distribution and store it."""
        if not memory.tom_module or not memory.tom_module.enabled:
            return

        try:
            from src.collaborative.tom.theory_of_mind import AgentRole, ReviewerAction

            current_iteration = memory.get_iteration()
            items_by_section = memory.get_feedback_items_for_iteration(
                current_iteration
            )
            all_items = []
            for items in items_by_section.values():
                all_items.extend(items)

            if not all_items:
                logger.info(
                    f"ToM: No feedback items from iteration {current_iteration}"
                )
                return

            # Count feedback by type
            type_counts = {}
            for item in all_items:
                feedback_type = getattr(item, "type", None)
                if feedback_type:
                    type_str = (
                        str(feedback_type)
                        if not hasattr(feedback_type, "value")
                        else feedback_type.value
                    )
                    type_counts[type_str] = type_counts.get(type_str, 0) + 1

            # Calculate percentages
            total = len(all_items)
            type_percentages = {k: v / total for k, v in type_counts.items()}
            logger.info(f"ToM: Reviewer feedback type distribution: {type_percentages}")

            # Determine dominant focus
            if not type_percentages:
                observed_action = ReviewerAction.BALANCED_FEEDBACK.value
            else:
                max_type = max(type_percentages.items(), key=lambda x: x[1])[0]
                max_percentage = type_percentages[max_type]

                # If one type dominates (>40%), use it as focus
                if max_percentage > 0.4:
                    if max_type == "accuracy":
                        observed_action = ReviewerAction.FOCUS_ACCURACY.value
                    elif max_type == "content_expansion":
                        observed_action = ReviewerAction.FOCUS_EXPANSION.value
                    elif max_type == "structure":
                        observed_action = ReviewerAction.FOCUS_STRUCTURE.value
                    elif max_type == "clarity":
                        observed_action = ReviewerAction.FOCUS_CLARITY.value
                    elif max_type == "style":
                        observed_action = ReviewerAction.FOCUS_STYLE.value
                    else:
                        observed_action = ReviewerAction.BALANCED_FEEDBACK.value
                else:
                    observed_action = ReviewerAction.BALANCED_FEEDBACK.value

            # Append structured observation to history
            try:
                memory.append_tom_observation(
                    agent="reviewer",
                    observed_action=observed_action,
                    iteration=current_iteration,
                    details={
                        "total_feedback": total,
                        "type_counts": type_counts,
                        "type_percentages": type_percentages,
                    },
                )
            except Exception as obs_exc:
                logger.warning(f"ToM: Failed to append reviewer observation: {obs_exc}")

            logger.info(
                f"ToM: Reviewer action observed - {observed_action} "
                f"({total} feedback items, type distribution: {type_counts})"
            )

            # Evaluate pending predictions about reviewer
            recent_predictions = [
                p
                for p in memory.tom_module.prediction_history
                if p.predictor_role == AgentRole.WRITER
                and p.target_role == AgentRole.REVIEWER
                and p.accuracy.value == "unknown"
            ]

            for pred in recent_predictions:
                memory.tom_module.evaluate_prediction_accuracy(
                    prediction_id=pred.id, actual_outcome=observed_action
                )

        except Exception as e:
            logger.warning(f"Failed to update ToM after reviewer: {e}")
