class ConvergenceChecker:
    def __init__(self, max_iterations: int, min_feedback_threshold: int):
        self.max_iterations = max_iterations
        self.min_feedback_threshold = min_feedback_threshold

    def check_convergence(
        self, iteration: int, current_feedback: list, feedback_history: list
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

        return False, None
