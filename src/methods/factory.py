# src/methods/factory.py
"""
Factory for creating method instances.
"""

import logging

from src.methods.base_method import BaseMethod

logger = logging.getLogger(__name__)


def create_method(method_name: str) -> BaseMethod:
    """
    Create a method instance based on method name.

    Args:
        method_name: Name of the method to create

    Returns:
        Initialized method instance

    Raises:
        ValueError: If method_name is not supported
    """

    if method_name == "writer_v3":
        from src.methods.writer_only_v3_method import WriterOnlyV3Method

        return WriterOnlyV3Method()

    elif method_name == "writer_reviewer":
        from src.methods.writer_reviewer_v2_method import WriterReviewerV2Method

        return WriterReviewerV2Method(tom_enabled=False)

    elif method_name == "writer_reviewer_tom":
        from src.methods.writer_reviewer_v2_method import WriterReviewerV2Method

        return WriterReviewerV2Method(tom_enabled=True)

    elif method_name == "direct":
        from src.methods.direct_method import DirectMethod

        return DirectMethod()

    elif method_name == "rag":
        from src.methods.rag_method import RagMethod

        return RagMethod()

    elif method_name == "storm":
        from src.methods.storm_method import StormMethod

        return StormMethod()

    else:
        supported_methods = [
            "writer_only",
            "writer_v3",
            "writer_reviewer",
            "writer_reviewer_tom",
            "direct",
            "rag",
            "storm",
        ]
        raise ValueError(
            f"Unknown method: '{method_name}'. "
            f"Supported methods: {supported_methods}"
        )


def get_supported_methods() -> list[str]:
    """
    Get list of supported method names.

    Returns:
        List of supported method names
    """
    return [
        "writer_only",
        "writer_only_v2",
        "writer_reviewer",
        "writer_reviewer_tom",
        "writer_reviewer_v2",
        "writer_reviewer_v2_tom",
        "direct",
        "rag",
        "storm",
    ]
