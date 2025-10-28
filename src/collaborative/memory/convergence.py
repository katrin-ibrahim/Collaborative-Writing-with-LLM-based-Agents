from typing import Dict, List, Optional, Tuple


class ConvergenceChecker:
    """
    Weighted convergence based on VERIFIED_ADDRESSED only.

    Rules:
      1) Stop at max_iterations.
      2) If no feedback exists: iter 0 -> continue, else stop.
      3) If no pending remain -> stop ("All feedback items verified as addressed").
      4) If weighted verification rate >= threshold (after min_iterations) -> stop.
      5) If no high-priority pending and tail <= small_tail_max -> stop.
      6) Stall: if improvement < min_improvement for stall_tolerance consecutive iterations -> stop.

    Defaults are taken from CollaborationConfig when available.
    """

    def __init__(
        self,
        max_iterations: int,
        resolution_rate_threshold: float,
        min_iterations: int,
        stall_tolerance: int,
        min_improvement: float,
        priority_weights: Optional[
            Dict[str, int]
        ] = None,  # {"high":3,"medium":2,"low":1} = None,  # {"high":3,"medium":2,"low":1}
        small_tail_max: Optional[int] = None,  # e.g., 5 remaining low/medium items
    ):
        self.max_iterations = max_iterations
        self.resolution_rate_threshold = resolution_rate_threshold
        self.min_iterations = min_iterations
        self.stall_tolerance = stall_tolerance
        self.min_improvement = min_improvement
        self.small_tail_max = (
            small_tail_max if small_tail_max is not None else 5
        )  # default to 5 if not provided
        self.priority_weights = (
            priority_weights
            if priority_weights is not None
            else {"high": 3, "medium": 2, "low": 1}
        )

        # internal state for stall detection
        self._last_resolution_rate: Optional[float] = None
        self._stall_streak: int = 0

    # ---------- internals (weighted + verified-only) ----------
    @staticmethod
    def _get(item: dict, key: str, default=None):
        """Safe getter supporting dicts or pydantic objects."""
        if isinstance(item, dict):
            return item.get(key, default)
        # pydantic / attr-style
        return getattr(item, key, default)

    def _w(self, priority: Optional[str]) -> int:
        p = (priority or "medium").lower()
        return self.priority_weights.get(p, self.priority_weights["medium"])

    def _weighted_total(self, items: List[dict]) -> int:
        total = 0
        for it in items:
            total += self._w(self._get(it, "priority", "medium"))
        return total

    def _weighted_verified(self, items: List[dict]) -> int:
        total = 0
        for it in items:
            status = self._get(it, "status", "pending")
            status_str = getattr(status, "value", status)  # enum or str
            if (status_str or "").lower() == "verified_addressed":
                total += self._w(self._get(it, "priority", "medium"))
        return total

    def _verification_rate(self, all_items: List[dict]) -> float:
        """Weighted fraction of VERIFIED_ADDRESSED over all items."""
        if not all_items:
            return 0.0
        tot = self._weighted_total(all_items)
        if tot <= 0:
            return 0.0
        ver = self._weighted_verified(all_items)
        return ver / tot

    # ---------- public API ----------
    def check_convergence(
        self,
        iteration: int,
        pending_items: List[dict],
        all_items: List[dict],
    ) -> Tuple[bool, str]:
        # Rule 1: Max iterations
        if iteration >= self.max_iterations:
            return True, f"Maximum iterations ({self.max_iterations}) reached"

        # Rule 2: No feedback exists
        if len(all_items) == 0:
            if iteration == 0:
                return False, "First iteration — awaiting reviewer feedback"
            else:
                return True, "No feedback items found"

        # Rule 3a: All verified
        pending_count = len(pending_items)
        if pending_count == 0:
            return True, "All feedback items verified as addressed"

        # Weighted verification rate (verified-only)
        rate = self._verification_rate(all_items)

        # Rule 3b: High weighted verification rate
        if iteration >= self.min_iterations and rate >= self.resolution_rate_threshold:
            return True, f"High weighted verification rate: {rate:.1%} addressed"

        # Rule 4: Small tail (no high pending, few remain)
        high_pending = [
            it
            for it in pending_items
            if (self._get(it, "priority", "") or "").lower() == "high"
        ]
        if len(high_pending) == 0 and pending_count <= self.small_tail_max:
            return True, f"Only {pending_count} low/medium priority items remain"

        # Rule 5: Stall detection (tiny progress for N iters)
        if self._last_resolution_rate is not None:
            improvement = rate - self._last_resolution_rate
            if improvement < self.min_improvement:
                self._stall_streak += 1
            else:
                self._stall_streak = 0

            if (
                iteration >= self.min_iterations
                and self._stall_streak >= self.stall_tolerance
            ):
                self._last_resolution_rate = rate
                return True, (
                    f"Stalled: improvement < {self.min_improvement:.0%} for "
                    f"{self._stall_streak} consecutive iterations "
                    f"(weighted verification rate {rate:.1%})"
                )

        # update state for next call
        self._last_resolution_rate = rate

        # Not converged
        return (
            False,
            f"{pending_count} pending ({len(high_pending)} high); weighted verification rate {rate:.1%}",
        )

    def resolution_rate(
        self, pending_items: List[dict], all_items: List[dict]
    ) -> float:
        """Weighted fraction of VERIFIED_ADDRESSED over all items (for convergence score)."""
        if not all_items:
            return 0.0
        tot = self._weighted_total(all_items)
        if tot <= 0:
            return 0.0
        ver = self._weighted_verified(all_items)
        return ver / tot
