"""
Shared memory module for agent communication.
"""
from typing import Dict, List, Any, Optional

class SharedMemory:
    """
    Shared memory for agent interaction.
    
    This class stores the shared state between agents, such as the draft content,
    review comments, and revision history.
    """
    
    def __init__(self):
        # Draft content
        self.draft_content = ""
        self.outline = {}
        
        # Review comments
        self.review_comments = []
        
        # Revision history
        self.revision_history = []
        
        # Convergence metrics
        self.addressed_comments = 0
        self.total_comments = 0
        
    def add_draft(self, draft_content: str) -> None:
        """
        Add a new draft to the shared memory.
        
        Args:
            draft_content: The content of the draft
        """
        self.draft_content = draft_content
        self.revision_history.append({
            "type": "draft",
            "content": draft_content
        })
        
    def add_outline(self, outline: Dict[str, Any]) -> None:
        """
        Add an outline to the shared memory.
        
        Args:
            outline: The outline dictionary
        """
        self.outline = outline
        self.revision_history.append({
            "type": "outline",
            "content": outline
        })
        
    def add_review_comment(self, comment: Dict[str, Any]) -> None:
        """
        Add a review comment to the shared memory.
        
        Args:
            comment: A dictionary with the following keys:
                - category: The category of the comment (fact, structure, etc.)
                - severity: The severity of the comment (high, medium, low)
                - content: The content of the comment
                - location: The location of the comment in the draft
                - status: The status of the comment (open, addressed, contested)
        """
        self.review_comments.append(comment)
        self.total_comments += 1
        
    def update_comment_status(self, comment_id: int, status: str) -> None:
        """
        Update the status of a review comment.
        
        Args:
            comment_id: The ID of the comment to update
            status: The new status of the comment
        """
        if 0 <= comment_id < len(self.review_comments):
            old_status = self.review_comments[comment_id]["status"]
            self.review_comments[comment_id]["status"] = status
            
            # Update convergence metrics
            if old_status == "open" and status == "addressed":
                self.addressed_comments += 1
            elif old_status == "addressed" and status == "open":
                self.addressed_comments -= 1
                
    def get_open_comments(self) -> List[Dict[str, Any]]:
        """
        Get all open review comments.
        
        Returns:
            A list of open review comments
        """
        return [c for c in self.review_comments if c["status"] == "open"]
        
    def get_comments_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all review comments in a specific category.
        
        Args:
            category: The category of comments to get
            
        Returns:
            A list of review comments in the specified category
        """
        return [c for c in self.review_comments if c["category"] == category]
        
    def clear_comments(self) -> None:
        """Clear all review comments and reset convergence metrics."""
        self.review_comments = []
        self.addressed_comments = 0
        self.total_comments = 0
        
    def get_convergence(self) -> float:
        """
        Calculate the convergence ratio.
        
        Returns:
            The ratio of addressed comments to total comments, or 1.0 if there are no comments
        """
        if self.total_comments == 0:
            return 1.0
        return self.addressed_comments / self.total_comments