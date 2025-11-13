# src/collaborative/theory_of_mind.py
"""
Theory of Mind Module for Collaborative AI Writing

This module enables agents to model each other's mental states, predict behaviors,
and adapt their strategies based on implicit understanding rather than explicit communication.

True ToM involves:
1. Intent Prediction - predicting what the other agent will do
2. Belief Modeling - modeling what the other agent thinks/believes
3. Adaptation - changing behavior based on predicted mental states
"""

import time
from enum import Enum

import logging
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    WRITER = "writer"
    REVIEWER = "reviewer"


class PredictionAccuracy(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"


class ToMPredictionModel(BaseModel):
    """Structured LLM output for Theory of Mind predictions."""

    predicted_action: str = Field(
        description="What the target agent will likely do next (e.g., 'accept_most_feedback', 'request_major_expansion')"
    )
    confidence: float = Field(
        description="Confidence level in this prediction (0.0 to 1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Detailed reasoning explaining why this prediction makes sense based on the context and belief state"
    )


@dataclass
class BeliefState:
    """Model of what one agent believes about another agent's mental state."""

    agent_role: AgentRole
    beliefs_about_other: Dict[str, Any] = field(default_factory=dict)
    confidence_level: float = 0.5  # 0.0 to 1.0
    last_updated: float = field(default_factory=time.time)

    # Specific belief categories
    predicted_priorities: List[str] = field(
        default_factory=list
    )  # What they care about most
    predicted_satisfaction_level: float = (
        0.5  # How satisfied they are with current work
    )
    predicted_next_action: Optional[str] = None  # What they'll do next
    predicted_feedback_acceptance_rate: float = 0.5  # How likely to accept feedback


@dataclass
class IntentPrediction:
    """Prediction about what another agent intends to do."""

    predictor_role: AgentRole
    target_role: AgentRole
    predicted_action: str
    confidence: float
    reasoning: str
    id: str = field(default_factory=lambda: str(int(time.time() * 1000000)))
    timestamp: float = field(default_factory=time.time)
    actual_outcome: Optional[str] = None
    accuracy: PredictionAccuracy = PredictionAccuracy.UNKNOWN


@dataclass
class ToMInteraction:
    """Record of a Theory of Mind interaction between agents."""

    interaction_id: str
    writer_belief_state: BeliefState
    reviewer_belief_state: BeliefState
    writer_predictions: List[IntentPrediction] = field(default_factory=list)
    reviewer_predictions: List[IntentPrediction] = field(default_factory=list)
    interaction_outcome: Optional[str] = None
    learning_occurred: bool = False
    timestamp: float = field(default_factory=time.time)


class TheoryOfMindModule:
    """
    Core Theory of Mind module for collaborative writing agents.

    Enables agents to:
    1. Model each other's mental states and beliefs
    2. Predict each other's actions and responses using LLM reasoning
    3. Adapt behavior based on predicted mental states
    4. Learn from prediction accuracy over time
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._llm_client = None  # Lazy-loaded LLM client
        self.interactions: List[ToMInteraction] = []
        self.belief_states: Dict[AgentRole, BeliefState] = {
            AgentRole.WRITER: BeliefState(AgentRole.WRITER),
            AgentRole.REVIEWER: BeliefState(AgentRole.REVIEWER),
        }
        self.prediction_history: List[IntentPrediction] = []

        if self.enabled:
            logger.info("Theory of Mind module initialized and ENABLED")
        else:
            logger.info("Theory of Mind module initialized but DISABLED")

    @property
    def llm_client(self):
        """Lazy load LLM client when first needed."""
        if self._llm_client is None and self.enabled:
            from src.config.config_context import ConfigContext

            try:
                self._llm_client = ConfigContext.get_client("theory_of_mind")
                logger.info("ToM: Lazy-loaded LLM client for predictions")
            except Exception as e:
                logger.warning(
                    f"ToM: Failed to get LLM client, will use heuristics: {e}"
                )
        return self._llm_client

    def predict_agent_response(
        self, predictor: AgentRole, target: AgentRole, context: Dict[str, Any]
    ) -> IntentPrediction:
        """
        Predict how the target agent will respond in the given context using LLM reasoning.
        This is the core ToM function - predicting without explicit communication.
        """
        if not self.enabled:
            logger.debug("ToM: Module disabled, returning default prediction")
            return self._create_default_prediction(predictor, target, context)

        logger.info(f"ToM: {predictor.value} predicting {target.value}'s response")

        # Get current belief state about the target agent
        belief_state = self.belief_states[predictor]

        # Use LLM for prediction if available, otherwise fall back to heuristics
        if self.llm_client:
            try:
                logger.info("ToM: Using LLM-based prediction")
                prediction = self._llm_predict(predictor, target, context, belief_state)
                logger.info(
                    f"ToM: LLM predicted action: {prediction.predicted_action} "
                    f"(confidence: {prediction.confidence:.2f})"
                )
            except Exception as e:
                logger.warning(
                    f"ToM: LLM prediction failed, using fallback heuristics: {e}"
                )
                prediction = self._heuristic_predict(
                    predictor, target, context, belief_state
                )
        else:
            logger.info("ToM: No LLM client available, using heuristic prediction")
            prediction = self._heuristic_predict(
                predictor, target, context, belief_state
            )

        # Store prediction for later accuracy evaluation
        self.prediction_history.append(prediction)
        logger.info(
            f"ToM: Final prediction - {predictor.value} predicts {target.value} will: "
            f"{prediction.predicted_action} (confidence: {prediction.confidence:.2f})"
        )

        return prediction

    def _llm_predict(
        self,
        predictor: AgentRole,
        target: AgentRole,
        context: Dict[str, Any],
        belief_state: BeliefState,
    ) -> IntentPrediction:
        """Use LLM to make Theory of Mind predictions."""
        # Build prompt based on predictor and target roles
        if predictor == AgentRole.WRITER and target == AgentRole.REVIEWER:
            prompt = self._build_writer_predicts_reviewer_prompt(context, belief_state)
        elif predictor == AgentRole.REVIEWER and target == AgentRole.WRITER:
            prompt = self._build_reviewer_predicts_writer_prompt(context, belief_state)
        else:
            return self._create_default_prediction(predictor, target, context)

        # Call LLM with structured output
        if self.llm_client is None:
            logger.warning("ToM: LLM client not available, falling back to heuristics")
            return self._create_default_prediction(predictor, target, context)

        response: ToMPredictionModel = self.llm_client.call_structured_api(
            prompt=prompt,
            output_schema=ToMPredictionModel,
            temperature=0.3,
        )

        return IntentPrediction(
            predictor_role=predictor,
            target_role=target,
            predicted_action=response.predicted_action,
            confidence=response.confidence,
            reasoning=response.reasoning,
        )

    def _build_writer_predicts_reviewer_prompt(
        self, context: Dict[str, Any], belief_state: BeliefState
    ) -> str:
        """Build prompt for writer predicting reviewer behavior."""
        article_metrics = context.get("article_metrics", {})
        word_count = article_metrics.get("word_count", "unknown")
        section_count = article_metrics.get("section_count", "unknown")

        predicted_priorities = belief_state.predicted_priorities or [
            "accuracy",
            "completeness",
            "structure",
        ]
        predicted_satisfaction = belief_state.predicted_satisfaction_level

        return f"""You are modeling the mental state of a REVIEWER agent based on the WRITER's perspective.

CURRENT CONTEXT:
- Article word count: {word_count}
- Article sections: {section_count}
- Action: {context.get('action', 'review')}

WRITER'S BELIEFS ABOUT REVIEWER:
- Reviewer priorities: {', '.join(predicted_priorities)}
- Reviewer satisfaction level: {predicted_satisfaction:.2f} (0.0 = very unsatisfied, 1.0 = very satisfied)
- Confidence in these beliefs: {belief_state.confidence_level:.2f}

PREDICTION TASK: Based on the reviewer's priorities and satisfaction level, predict their SPECIFIC next action.

DECISION-MAKING PROCESS:
1. **Assess Article Quality Against Reviewer Priorities:**
   - If reviewer prioritizes "accuracy" → check if word count/sections suggest thorough coverage
   - If reviewer prioritizes "completeness" → check if section count is comprehensive
   - If reviewer prioritizes "structure" → evaluate if organization seems logical

2. **Map Satisfaction to Likely Feedback Intensity:**
   - Satisfaction 0.0-0.3 → predict "provide_extensive_feedback" or "request_major_revision"
   - Satisfaction 0.4-0.6 → predict "focus_on_specific_improvements" or "request_targeted_expansion"
   - Satisfaction 0.7-0.9 → predict "provide_minor_refinements" or "approve_with_small_suggestions"
   - Satisfaction 0.9-1.0 → predict "approve_article" or "minimal_feedback"

3. **Consider Action Context:**
   - First iteration → expect thorough initial review
   - Later iterations → expect focused verification of previous feedback
   - Research available → expect fact-checking and citation review
   - No research → expect structural and clarity focus

CONFIDENCE CALIBRATION BASED ON EVIDENCE STRENGTH:
HIGH CONFIDENCE (0.8-1.0):
- Clear historical pattern of reviewer behavior exists
- Satisfaction level strongly indicates specific action
- Article metrics clearly meet or fail reviewer priorities
- Consistent alignment between priorities and metrics

MEDIUM CONFIDENCE (0.5-0.7):
- Some historical patterns but not fully consistent
- Satisfaction level is moderate (neither very high nor very low)
- Article metrics partially meet reviewer priorities
- Mixed signals between different priority dimensions

LOW CONFIDENCE (0.1-0.4):
- No clear historical patterns
- Satisfaction level is uncertain or contradictory
- Article metrics don't clearly relate to reviewer priorities
- First interaction or insufficient context

CRITICAL: Your confidence should reflect HOW CERTAIN you are about the prediction based on available evidence.
DO NOT use 0.5 as a default - use the full range 0.1-1.0 based on evidence strength.

OUTPUT REQUIREMENTS:
- Action: A specific, concrete action (e.g., "provide_extensive_accuracy_feedback", "request_expansion_of_3_sections")
- Reasoning: 2-3 sentences explaining WHY you chose this action based on priorities and satisfaction
- Confidence: A value between 0.1 and 1.0 reflecting evidence strength (NOT a guess)
"""

    def _build_reviewer_predicts_writer_prompt(
        self, context: Dict[str, Any], belief_state: BeliefState
    ) -> str:
        """Build prompt for reviewer predicting writer behavior."""
        feedback_context = context.get("current_feedback", [])
        feedback_count = (
            len(feedback_context) if isinstance(feedback_context, list) else 0
        )
        feedback_severity = context.get("feedback_severity", "medium")

        predicted_acceptance_rate = belief_state.predicted_feedback_acceptance_rate
        writer_priorities = belief_state.predicted_priorities or [
            "creativity",
            "flow",
            "completeness",
        ]

        return f"""You are modeling the mental state of a WRITER agent based on the REVIEWER's perspective.

CURRENT CONTEXT:
- Number of feedback items: {feedback_count}
- Feedback severity: {feedback_severity}
- Review strategy: {context.get('review_strategy', 'holistic')}
- Iteration: {context.get('iteration', 0)}

REVIEWER'S BELIEFS ABOUT WRITER:
- Writer priorities: {', '.join(writer_priorities)}
- Predicted feedback acceptance rate: {predicted_acceptance_rate:.0%}
- Confidence in these beliefs: {belief_state.confidence_level:.2f}

PREDICTION TASK: Based on writer priorities, acceptance rate, and feedback context, predict their SPECIFIC response action.

DECISION-MAKING PROCESS:
1. **Assess Feedback Alignment with Writer Priorities:**
   - If feedback conflicts with writer's "creativity" priority → expect resistance or partial acceptance
   - If feedback aligns with "completeness" priority → expect high acceptance
   - If feedback requires major structural changes vs writer's "flow" priority → expect negotiation

2. **Map Acceptance Rate + Feedback Volume to Response:**
   - High acceptance (>70%) + Low feedback count (<5) → "accept_all_feedback"
   - High acceptance + High feedback count (>10) → "accept_most_with_minor_pushback"
   - Medium acceptance (40-70%) + Medium feedback → "selectively_accept_prioritize_high"
   - Low acceptance (<40%) + High severity → "contest_feedback_request_clarification"

3. **Consider Severity and Writer Confidence:**
   - Low severity + confident writer → quick acceptance
   - High severity + defensive writer → request justification
   - First iteration → more receptive to feedback
   - Late iteration → more resistant if already addressed similar issues

CONFIDENCE CALIBRATION BASED ON EVIDENCE STRENGTH:
HIGH CONFIDENCE (0.8-1.0):
- Strong historical acceptance rate pattern (e.g., consistently >80% or <20%)
- Clear priority conflicts or alignments with feedback
- Consistent past behavior in similar situations
- Feedback type matches known writer strengths/weaknesses

MEDIUM CONFIDENCE (0.5-0.7):
- Moderate acceptance rate (40-70%) with some variance
- Some priority conflicts but not severe
- Mixed historical responses to similar feedback
- Moderate amount of contextual information

LOW CONFIDENCE (0.1-0.4):
- No historical acceptance data or highly variable
- Unclear priority conflicts
- First-time collaboration or novel feedback type
- Contradictory signals from different context elements

CRITICAL: Confidence should reflect PREDICTIVE CERTAINTY based on:
1. Consistency of historical acceptance patterns
2. Clarity of priority alignments/conflicts
3. Amount and quality of contextual information
4. Strength of causal reasoning from context to prediction

DO NOT use 0.5 as a default - assess actual evidence strength and use the full range 0.1-1.0.

OUTPUT REQUIREMENTS:
- Action: A specific, concrete response (e.g., "accept_80_percent_contest_structure_feedback", "request_examples_for_clarity_items")
- Reasoning: 2-3 sentences explaining WHY based on priorities, acceptance rate, and feedback context
- Confidence: A value between 0.1 and 1.0 reflecting evidence strength and predictive certainty
"""

    def _heuristic_predict(
        self,
        predictor: AgentRole,
        target: AgentRole,
        context: Dict[str, Any],
        belief_state: BeliefState,
    ) -> IntentPrediction:
        """Fallback heuristic-based predictions when LLM is unavailable."""
        if predictor == AgentRole.WRITER and target == AgentRole.REVIEWER:
            return self._writer_predicts_reviewer_heuristic(context, belief_state)
        elif predictor == AgentRole.REVIEWER and target == AgentRole.WRITER:
            return self._reviewer_predicts_writer_heuristic(context, belief_state)
        else:
            return self._create_default_prediction(predictor, target, context)

    def _writer_predicts_reviewer_heuristic(
        self, context: Dict[str, Any], belief_state: BeliefState
    ) -> IntentPrediction:
        """Writer predicts how reviewer will behave (heuristic fallback)."""

        # Analyze context to predict reviewer behavior
        article_quality_indicators = context.get("article_metrics", {})
        context.get("previous_feedback_severity", "medium")

        # Predict based on believed reviewer priorities
        predicted_priorities = belief_state.predicted_priorities or [
            "accuracy",
            "completeness",
            "structure",
        ]

        predicted_action = "provide_moderate_feedback"
        confidence = 0.6
        reasoning = f"Based on reviewer's apparent priorities: {predicted_priorities}"

        # Adjust prediction based on article quality
        word_count = article_quality_indicators.get("word_count", 500)
        if word_count < 300:
            predicted_action = "request_major_expansion"
            confidence = 0.8
            reasoning += ". Article appears too short for reviewer standards."
        elif word_count > 1500:
            predicted_action = "focus_on_accuracy_refinement"
            confidence = 0.7
            reasoning += (
                ". Article length adequate, reviewer likely to focus on quality."
            )

        # Factor in prediction of reviewer's satisfaction level
        if belief_state.predicted_satisfaction_level < 0.3:
            predicted_action = "provide_extensive_feedback"
            confidence = 0.75
            reasoning += f" Reviewer seems unsatisfied (predicted satisfaction: {belief_state.predicted_satisfaction_level:.2f})."

        return IntentPrediction(
            predictor_role=AgentRole.WRITER,
            target_role=AgentRole.REVIEWER,
            predicted_action=predicted_action,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _reviewer_predicts_writer_heuristic(
        self, context: Dict[str, Any], belief_state: BeliefState
    ) -> IntentPrediction:
        """Reviewer predicts how writer will respond to feedback (heuristic fallback)."""

        # Analyze feedback acceptance patterns
        feedback_context = context.get("current_feedback", [])
        feedback_severity = context.get("feedback_severity", "medium")

        predicted_acceptance_rate = belief_state.predicted_feedback_acceptance_rate

        predicted_action = "accept_most_feedback"
        confidence = 0.6
        reasoning = (
            f"Writer typically accepts {predicted_acceptance_rate:.0%} of feedback"
        )

        # Predict based on feedback characteristics
        if feedback_severity == "high" or len(feedback_context) > 5:
            if predicted_acceptance_rate > 0.7:
                predicted_action = "accept_feedback_but_pushback_on_scope"
                confidence = 0.7
                reasoning += ". Heavy feedback load may cause selective acceptance."
            else:
                predicted_action = "contest_some_feedback"
                confidence = 0.8
                reasoning += ". Writer shows low acceptance rate and may resist extensive feedback."

        # Factor in writer's predicted priorities
        writer_priorities = belief_state.predicted_priorities or [
            "creativity",
            "flow",
            "completeness",
        ]
        if "creativity" in writer_priorities and feedback_severity == "high":
            predicted_action = "partially_accept_maintain_creative_vision"
            reasoning += f" Writer priorities ({writer_priorities}) may conflict with extensive feedback."

        return IntentPrediction(
            predictor_role=AgentRole.REVIEWER,
            target_role=AgentRole.WRITER,
            predicted_action=predicted_action,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _create_default_prediction(
        self, predictor: AgentRole, target: AgentRole, context: Dict[str, Any]
    ) -> IntentPrediction:
        """Create default prediction when ToM is disabled."""
        return IntentPrediction(
            predictor_role=predictor,
            target_role=target,
            predicted_action="default_behavior",
            confidence=0.5,
            reasoning="ToM disabled - using default prediction",
        )

    def update_belief_state(
        self, observer: AgentRole, observed_behavior: Dict[str, Any]
    ):
        """Update belief state based on observed behavior of the other agent."""
        if not self.enabled:
            logger.debug("ToM: Module disabled, skipping belief state update")
            return

        logger.info(
            f"ToM: {observer.value} updating beliefs based on observed behavior"
        )

        belief_state = self.belief_states[observer]
        updates_made = []

        # Update beliefs based on observed behavior
        if "feedback_acceptance_rate" in observed_behavior:
            old_rate = belief_state.predicted_feedback_acceptance_rate
            new_rate = observed_behavior["feedback_acceptance_rate"]
            belief_state.predicted_feedback_acceptance_rate = (
                old_rate * 0.7 + new_rate * 0.3
            )
            updates_made.append(
                f"acceptance_rate: {old_rate:.2f} -> {belief_state.predicted_feedback_acceptance_rate:.2f}"
            )

        if "priorities_shown" in observed_behavior:
            new_priorities = observed_behavior["priorities_shown"]
            belief_state.predicted_priorities = new_priorities
            updates_made.append(f"priorities: {new_priorities}")

        if "satisfaction_indicators" in observed_behavior:
            satisfaction_signals = observed_behavior["satisfaction_indicators"]
            old_sat = belief_state.predicted_satisfaction_level
            new_sat = satisfaction_signals.get("level", 0.5)
            belief_state.predicted_satisfaction_level = new_sat
            updates_made.append(f"satisfaction: {old_sat:.2f} -> {new_sat:.2f}")

        belief_state.last_updated = time.time()
        old_confidence = belief_state.confidence_level
        belief_state.confidence_level = min(1.0, old_confidence + 0.1)

        logger.info(
            f"ToM: {observer.value} belief updates: {', '.join(updates_made) if updates_made else 'none'}. "
            f"Confidence: {old_confidence:.2f} -> {belief_state.confidence_level:.2f}"
        )

    def evaluate_prediction_accuracy(
        self, prediction_id: str, actual_outcome: str
    ) -> PredictionAccuracy:
        """Evaluate how accurate a prediction was after seeing the actual outcome."""
        if not self.enabled:
            logger.debug("ToM: Module disabled, skipping prediction evaluation")
            return PredictionAccuracy.UNKNOWN

        logger.info(f"ToM: Evaluating prediction {prediction_id}")

        # Find the prediction
        prediction = None
        for pred in self.prediction_history:
            if hasattr(pred, "id") and pred.id == prediction_id:
                prediction = pred
                break

        if not prediction:
            logger.warning(f"ToM: Prediction {prediction_id} not found in history")
            return PredictionAccuracy.UNKNOWN

        # Simple accuracy check
        prediction.actual_outcome = actual_outcome

        if (
            prediction.predicted_action in actual_outcome
            or actual_outcome in prediction.predicted_action
        ):
            prediction.accuracy = PredictionAccuracy.CORRECT
            logger.info(
                f"ToM: ✓ Correct prediction! "
                f"Predicted: {prediction.predicted_action}, Actual: {actual_outcome}"
            )
            self._reinforce_successful_beliefs(prediction)
        else:
            prediction.accuracy = PredictionAccuracy.INCORRECT
            logger.info(
                f"ToM: ✗ Incorrect prediction. "
                f"Predicted: {prediction.predicted_action}, Actual: {actual_outcome}"
            )
            self._adjust_failed_beliefs(prediction)

        return prediction.accuracy

    def _reinforce_successful_beliefs(self, correct_prediction: IntentPrediction):
        """Strengthen beliefs that led to correct predictions."""
        predictor_beliefs = self.belief_states[correct_prediction.predictor_role]
        predictor_beliefs.confidence_level = min(
            1.0, predictor_beliefs.confidence_level + 0.05
        )

    def _adjust_failed_beliefs(self, incorrect_prediction: IntentPrediction):
        """Adjust beliefs that led to incorrect predictions."""
        predictor_beliefs = self.belief_states[incorrect_prediction.predictor_role]
        predictor_beliefs.confidence_level = max(
            0.0, predictor_beliefs.confidence_level - 0.05
        )

        # More sophisticated belief adjustment could happen here
        # e.g., questioning assumptions about other agent's priorities

    def get_prediction_accuracy_stats(self) -> Dict[str, Any]:
        """Get statistics on prediction accuracy for research analysis."""
        if not self.prediction_history:
            return {"total_predictions": 0, "accuracy_rate": 0.0}

        total = len(self.prediction_history)
        correct = sum(
            1
            for p in self.prediction_history
            if p.accuracy == PredictionAccuracy.CORRECT
        )

        return {
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy_rate": correct / total if total > 0 else 0.0,
            "writer_predictions": len(
                [
                    p
                    for p in self.prediction_history
                    if p.predictor_role == AgentRole.WRITER
                ]
            ),
            "reviewer_predictions": len(
                [
                    p
                    for p in self.prediction_history
                    if p.predictor_role == AgentRole.REVIEWER
                ]
            ),
        }

    def get_belief_evolution(self) -> Dict[str, Any]:
        """Get data on how beliefs have evolved over time for research analysis."""
        return {
            "writer_beliefs": {
                "confidence": self.belief_states[AgentRole.WRITER].confidence_level,
                "predicted_reviewer_priorities": self.belief_states[
                    AgentRole.WRITER
                ].predicted_priorities,
                "predicted_reviewer_satisfaction": self.belief_states[
                    AgentRole.WRITER
                ].predicted_satisfaction_level,
            },
            "reviewer_beliefs": {
                "confidence": self.belief_states[AgentRole.REVIEWER].confidence_level,
                "predicted_writer_priorities": self.belief_states[
                    AgentRole.REVIEWER
                ].predicted_priorities,
                "predicted_writer_acceptance_rate": self.belief_states[
                    AgentRole.REVIEWER
                ].predicted_feedback_acceptance_rate,
            },
        }

    def get_prediction_quality_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics on prediction quality for research analysis."""
        if not self.prediction_history:
            return {
                "total_predictions": 0,
                "confidence_distribution": {},
                "action_diversity": {},
                "quality_indicators": {},
            }

        # Confidence distribution
        confidence_buckets = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        confidence_values = []
        for pred in self.prediction_history:
            conf = pred.confidence
            confidence_values.append(conf)

            if conf < 0.2:
                confidence_buckets["0.0-0.2"] += 1
            elif conf < 0.4:
                confidence_buckets["0.2-0.4"] += 1
            elif conf < 0.6:
                confidence_buckets["0.4-0.6"] += 1
            elif conf < 0.8:
                confidence_buckets["0.6-0.8"] += 1
            else:
                confidence_buckets["0.8-1.0"] += 1

        # Action diversity
        action_counts = {}
        for pred in self.prediction_history:
            action = pred.predicted_action
            action_counts[action] = action_counts.get(action, 0) + 1

        # Calculate diversity score (normalized entropy)
        import math

        total = len(self.prediction_history)
        diversity_score = 0.0
        if total > 1:
            for count in action_counts.values():
                if count > 0:
                    p = count / total
                    diversity_score -= p * math.log2(p)

            max_entropy = (
                math.log2(len(action_counts)) if len(action_counts) > 1 else 1.0
            )
            diversity_score = diversity_score / max_entropy if max_entropy > 0 else 0.0

        # Quality indicators
        avg_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )
        std_confidence = 0.0
        if len(confidence_values) > 1:
            variance = sum((x - avg_confidence) ** 2 for x in confidence_values) / len(
                confidence_values
            )
            std_confidence = math.sqrt(variance)

        return {
            "total_predictions": len(self.prediction_history),
            "confidence_distribution": confidence_buckets,
            "action_diversity": {
                "unique_actions": len(action_counts),
                "action_counts": action_counts,
                "diversity_score": diversity_score,
            },
            "quality_indicators": {
                "avg_confidence": avg_confidence,
                "std_confidence": std_confidence,
                "min_confidence": min(confidence_values) if confidence_values else 0.0,
                "max_confidence": max(confidence_values) if confidence_values else 0.0,
            },
        }
