"""
Revision tracker for managing collaborative editing iterations.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime

from content_generation_system.src.memory.shared_memory import SharedMemory

class RevisionTracker:
    """
    Tracks the revision process between writer and reviewer agents.
    
    This class orchestrates the iterative revision process, tracking convergence
    and managing the interaction between writer and reviewer agents.
    """
    
    def __init__(
        self,
        shared_memory: Optional[SharedMemory] = None,
        max_iterations: int = 3,
        convergence_threshold: float = 0.9
    ):
        # Initialize shared memory
        self.shared_memory = shared_memory or SharedMemory()
        
        # Set parameters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize state
        self.current_iteration = 0
        self.convergence = 0.0
        self.start_time = None
        self.end_time = None
        self.iteration_history = []
        
    def start_revision_process(self) -> None:
        """Start the revision process and record the start time."""
        self.start_time = datetime.now()
        self.current_iteration = 0
        self.convergence = 0.0
        self.iteration_history = []
        
    def end_revision_process(self) -> None:
        """End the revision process and record the end time."""
        self.end_time = datetime.now()
        
    def record_iteration(self, iteration_data: Dict[str, Any]) -> None:
        """
        Record data from a revision iteration.
        
        Args:
            iteration_data: Dictionary containing iteration metrics and events
        """
        self.iteration_history.append({
            "iteration": self.current_iteration,
            "timestamp": datetime.now(),
            "data": iteration_data
        })
        
    def should_continue(self) -> bool:
        """
        Determine if the revision process should continue.
        
        Returns:
            True if the process should continue, False otherwise
        """
        # Check if we've reached the maximum number of iterations
        if self.current_iteration >= self.max_iterations:
            return False
            
        # Check if we've reached convergence
        if self.convergence >= self.convergence_threshold:
            return False
            
        return True
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the revision process.
        
        Returns:
            Dictionary containing revision process statistics
        """
        duration = None
        if self.start_time:
            end = self.end_time or datetime.now()
            duration = (end - self.start_time).total_seconds()
            
        return {
            "iterations_completed": self.current_iteration,
            "max_iterations": self.max_iterations,
            "convergence": self.convergence,
            "convergence_threshold": self.convergence_threshold,
            "duration_seconds": duration,
            "comments_total": self.shared_memory.total_comments,
            "comments_addressed": self.shared_memory.addressed_comments
        }
        
    def update_convergence(self) -> float:
        """
        Update and return the current convergence.
        
        Returns:
            The current convergence value
        """
        self.convergence = self.shared_memory.get_convergence()
        return self.convergence