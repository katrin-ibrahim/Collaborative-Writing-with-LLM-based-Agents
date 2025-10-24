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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    WRITER = "writer"
    REVIEWER = "reviewer"


class PredictionAccuracy(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"


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
    2. Predict each other's actions and responses
    3. Adapt behavior based on predicted mental states
    4. Learn from prediction accuracy over time
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
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

    def predict_agent_response(
        self, predictor: AgentRole, target: AgentRole, context: Dict[str, Any]
    ) -> IntentPrediction:
        """
        Predict how the target agent will respond in the given context.
        This is the core ToM function - predicting without explicit communication.
        """
        if not self.enabled:
            return self._create_default_prediction(predictor, target, context)

        # Get current belief state about the target agent
        belief_state = self.belief_states[predictor]

        # Make prediction based on:
        # 1. Historical patterns
        # 2. Current context
        # 3. Believed mental state of target

        if predictor == AgentRole.WRITER and target == AgentRole.REVIEWER:
            prediction = self._writer_predicts_reviewer(context, belief_state)
        elif predictor == AgentRole.REVIEWER and target == AgentRole.WRITER:
            prediction = self._reviewer_predicts_writer(context, belief_state)
        else:
            prediction = self._create_default_prediction(predictor, target, context)

        # Store prediction for later accuracy evaluation
        self.prediction_history.append(prediction)
        logger.info(
            f"{predictor.value} predicts {target.value} will: {prediction.predicted_action}"
        )

        return prediction

    def _writer_predicts_reviewer(
        self, context: Dict[str, Any], belief_state: BeliefState
    ) -> IntentPrediction:
        """Writer predicts how reviewer will behave."""

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

    def _reviewer_predicts_writer(
        self, context: Dict[str, Any], belief_state: BeliefState
    ) -> IntentPrediction:
        """Reviewer predicts how writer will respond to feedback."""

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
            return

        belief_state = self.belief_states[observer]

        # Update beliefs based on observed behavior
        if "feedback_acceptance_rate" in observed_behavior:
            # Update prediction of acceptance rate
            old_rate = belief_state.predicted_feedback_acceptance_rate
            new_rate = observed_behavior["feedback_acceptance_rate"]
            # Weighted average - recent behavior gets more weight
            belief_state.predicted_feedback_acceptance_rate = (
                old_rate * 0.7 + new_rate * 0.3
            )

        if "priorities_shown" in observed_behavior:
            # Update understanding of other agent's priorities
            new_priorities = observed_behavior["priorities_shown"]
            belief_state.predicted_priorities = new_priorities

        if "satisfaction_indicators" in observed_behavior:
            # Update prediction of satisfaction level
            satisfaction_signals = observed_behavior["satisfaction_indicators"]
            belief_state.predicted_satisfaction_level = satisfaction_signals.get(
                "level", 0.5
            )

        belief_state.last_updated = time.time()
        belief_state.confidence_level = min(
            1.0, belief_state.confidence_level + 0.1
        )  # Increase confidence with more observations

        logger.info(
            f"{observer.value} updated beliefs about other agent. Confidence: {belief_state.confidence_level:.2f}"
        )

    def evaluate_prediction_accuracy(
        self, prediction_id: str, actual_outcome: str
    ) -> PredictionAccuracy:
        """Evaluate how accurate a prediction was after seeing the actual outcome."""
        if not self.enabled:
            return PredictionAccuracy.UNKNOWN

        # Find the prediction
        prediction = None
        for pred in self.prediction_history:
            if hasattr(pred, "id") and pred.id == prediction_id:
                prediction = pred
                break

        if not prediction:
            return PredictionAccuracy.UNKNOWN

        # Simple accuracy check (in real system, this would be more sophisticated)
        prediction.actual_outcome = actual_outcome

        if (
            prediction.predicted_action in actual_outcome
            or actual_outcome in prediction.predicted_action
        ):
            prediction.accuracy = PredictionAccuracy.CORRECT
            # Improve confidence in beliefs that led to correct predictions
            self._reinforce_successful_beliefs(prediction)
        else:
            prediction.accuracy = PredictionAccuracy.INCORRECT
            # Adjust beliefs that led to incorrect predictions
            self._adjust_failed_beliefs(prediction)

        logger.info(
            f"Prediction accuracy: {prediction.accuracy.value} (predicted: {prediction.predicted_action}, actual: {actual_outcome})"
        )
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

    def enable_tom(self):
        """Enable Theory of Mind functionality."""
        self.enabled = True
        logger.info("Theory of Mind module ENABLED")

    def disable_tom(self):
        """Disable Theory of Mind functionality."""
        self.enabled = False
        logger.info("Theory of Mind module DISABLED")
