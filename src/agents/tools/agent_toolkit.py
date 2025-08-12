from typing import Any, Dict

from src.agents.tools.content_toolkit import ContentToolkit
from src.agents.tools.search_toolkit import SearchToolkit


class AgentToolkit:
    """
    Simplified toolkit for 3-node writer agent.
    Only includes essential search and content generation tools.
    """

    def __init__(self, config: Dict[str, Any]):
        self.search = SearchToolkit(config)
        self.content = ContentToolkit(config)
