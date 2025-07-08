from typing import Any, Dict

from agents.tools.content_toolkit import ContentToolkit
from agents.tools.evaluation_toolkit import EvaluationToolkit
from agents.tools.knowledge_toolkit import KnowledgeToolkit
from agents.tools.search_toolkit import SearchToolkit


class AgentToolkit:
    """
    Complete toolkit that combines all tool categories.

    Agents can access the specific toolkit categories they need:
    - Writer agents: All toolkits
    - Reviewer agents: Search, Knowledge (for verification), Evaluation
    - Future specialized agents: Subset based on their role
    """

    def __init__(self, config: Dict[str, Any]):
        self.search = SearchToolkit(config)
        self.knowledge = KnowledgeToolkit()
        self.content = ContentToolkit(config)
        self.evaluation = EvaluationToolkit()
