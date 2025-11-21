# src/collaborative/tom/theory_of_mind.py
"""Theory of Mind Module for Collaborative AI Writing"""

import time
from enum import Enum

import logging
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    WRITER = "writer"
    REVIEWER = "reviewer"


class PredictionAccuracy(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"


class WriterAction(str, Enum):
    """Observable summary of writer's revision behavior based on feedback acceptance rate."""

    ACCEPT_MOST = "accept_most_feedback"
    PARTIALLY_ACCEPT = "partially_accept_feedback"
    CONTEST_SOME = "contest_some_feedback"


class ReviewerAction(str, Enum):
    """Observable reviewer feedback focus based on feedback type distribution."""

    FOCUS_ACCURACY = "focus_on_accuracy"
    FOCUS_EXPANSION = "focus_on_content_expansion"
    FOCUS_STRUCTURE = "focus_on_structure"
    FOCUS_CLARITY = "focus_on_clarity"
    FOCUS_STYLE = "focus_on_style"
    BALANCED_FEEDBACK = "balanced_feedback"


class ToMPredictionModel(BaseModel):
    """Structured LLM output for predicting an agent's next action/state.

    Valid predicted_action values:
    - Writer actions: accept_most_feedback, partially_accept_feedback, contest_some_feedback
    - Reviewer actions: focus_on_accuracy, focus_on_content_expansion, focus_on_structure,
                       focus_on_clarity, focus_on_style, balanced_feedback
    """

    predicted_action: Literal[
        "accept_most_feedback",
        "partially_accept_feedback",
        "contest_some_feedback",
        "focus_on_accuracy",
        "focus_on_content_expansion",
        "focus_on_structure",
        "focus_on_clarity",
        "focus_on_style",
        "balanced_feedback",
    ] = Field(description="The predicted action/state from the allowed enum values")
    confidence: float = Field(
        description="Confidence level in this prediction (0.0 to 1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Detailed reasoning explaining why this prediction makes sense"
    )


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
    iteration: Optional[int] = None  # Iteration when prediction was made


class TheoryOfMindModule:
    """Theory of Mind module for collaborative writing agents."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.prediction_history: List[IntentPrediction] = []
        if self.enabled:
            logger.info("Theory of Mind module initialized and ENABLED")
        else:
            logger.info("Theory of Mind module initialized but DISABLED")

    def predict_agent_response(
        self, llm_client: Any, prompt: str
    ) -> ToMPredictionModel:
        """Predict how an agent will respond using LLM client and prompt."""
        if not self.enabled:
            logger.debug("ToM: Module disabled, returning default prediction")
            return ToMPredictionModel(
                predicted_action="balanced_feedback",
                confidence=0.5,
                reasoning="ToM disabled - using default prediction",
            )

        logger.info("ToM: Making prediction using provided LLM client")
        try:
            response = llm_client.call_structured_api(
                prompt=prompt,
                output_schema=ToMPredictionModel,
                temperature=0.3,
            )
            logger.info(
                f"ToM: Predicted action: {response.predicted_action} (confidence: {response.confidence:.2f})"
            )
            return response
        except Exception as e:
            logger.warning(f"ToM: Prediction failed: {e}")
            return ToMPredictionModel(
                predicted_action="balanced_feedback",
                confidence=0.0,
                reasoning=f"Prediction failed: {str(e)}",
            )

    def store_prediction(
        self,
        predictor_role: AgentRole,
        target_role: AgentRole,
        prediction: ToMPredictionModel,
        iteration: Optional[int] = None,
    ) -> str:
        """Store a prediction for later accuracy evaluation."""
        intent_prediction = IntentPrediction(
            predictor_role=predictor_role,
            target_role=target_role,
            predicted_action=prediction.predicted_action,
            confidence=prediction.confidence,
            reasoning=prediction.reasoning,
            iteration=iteration,
        )
        self.prediction_history.append(intent_prediction)
        logger.info(f"ToM: Stored prediction {intent_prediction.id}")
        return intent_prediction.id

    def evaluate_prediction_accuracy(
        self, prediction_id: str, actual_outcome: str
    ) -> PredictionAccuracy:
        """Evaluate how accurate a prediction was after seeing the actual outcome."""
        if not self.enabled:
            return PredictionAccuracy.UNKNOWN

        prediction = None
        for pred in self.prediction_history:
            if hasattr(pred, "id") and pred.id == prediction_id:
                prediction = pred
                break

        if not prediction:
            logger.warning(f"ToM: Prediction {prediction_id} not found")
            return PredictionAccuracy.UNKNOWN

        prediction.actual_outcome = actual_outcome
        if prediction.predicted_action == actual_outcome:
            prediction.accuracy = PredictionAccuracy.CORRECT
            logger.info(
                f"ToM: ✓ Correct prediction! Predicted: {prediction.predicted_action}, Actual: {actual_outcome}"
            )
        else:
            prediction.accuracy = PredictionAccuracy.INCORRECT
            logger.info(
                f"ToM: ✗ Incorrect prediction. Predicted: {prediction.predicted_action}, Actual: {actual_outcome}"
            )

        return prediction.accuracy

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

    def get_prediction_quality_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics on prediction quality for research analysis."""
        if not self.prediction_history:
            return {
                "total_predictions": 0,
                "confidence_distribution": {},
                "action_diversity": {},
                "quality_indicators": {},
            }

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

        action_counts = {}
        for pred in self.prediction_history:
            action = pred.predicted_action
            action_counts[action] = action_counts.get(action, 0) + 1

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
