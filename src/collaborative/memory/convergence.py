class ConvergenceChecker:
    def __init__(
        self,
        max_iterations: int,
        min_feedback_threshold: int,
        feedback_addressed_threshold: float = 0.9,
    ):
        self.max_iterations = max_iterations
        self.min_feedback_threshold = min_feedback_threshold
        self.feedback_addressed_threshold = (
            feedback_addressed_threshold  # 90% by default
        )

    def check_convergence(
        self,
        iteration: int,
        current_feedback: list,
        feedback_history: list,
        structured_feedback: list = None,
    ) -> tuple[bool, str]:
        # Rule 1: Max iterations reached
        if iteration >= self.max_iterations:
            return True, "max_iterations"

        # Rule 2: No feedback for current iteration
        if len(current_feedback) == 0:
            return True, "no_feedback"

        # Rule 3: Feedback volume dropped below threshold
        if len(current_feedback) < self.min_feedback_threshold:
            return True, "low_feedback"

        # Rule 4: 90% of feedback items have been addressed
        if structured_feedback and len(structured_feedback) > 0:
            addressed_percentage = self._calculate_addressed_percentage(
                structured_feedback
            )
            if addressed_percentage >= self.feedback_addressed_threshold:
                return True, f"feedback_addressed_{addressed_percentage:.1%}"

        return False, None

    def _calculate_addressed_percentage(self, structured_feedback: list) -> float:
        """Calculate the percentage of feedback items that have been verified as addressed by reviewer."""
        if not structured_feedback:
            return 0.0

        total_feedback = len(structured_feedback)
        # Count feedback that reviewer has verified as addressed
        verified_addressed = sum(
            1
            for feedback in structured_feedback
            if feedback.get("reviewer_verification") == "verified_addressed"
            or feedback.get("status") == "applied"  # Legacy support
            or feedback.get("status")
            == "ignored"  # Writer disagreement accepted by reviewer
        )

        return verified_addressed / total_feedback
