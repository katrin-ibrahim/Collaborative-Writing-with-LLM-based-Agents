# FILE: utils/baseline_utils.py
"""
Shared utility functions for baseline experiments.
Used by both Ollama and local baseline implementations to maintain DRY principles.
"""
from pathlib import Path

import logging
from typing import Dict, List

from src.utils.data_models import Article

logger = logging.getLogger(__name__)


def build_direct_prompt(topic: str) -> str:
    """Build direct prompting prompt (shared between Ollama and local baselines)."""
    return f"""Write a comprehensive, well-structured Wikipedia-style article about \"{topic}\".

You are an expert encyclopedia writer. Create a detailed, factual article that captures the essential information about this topic.

CRITICAL REQUIREMENTS:
1. Start with a strong, informative introduction that summarizes the key facts
2. Create 4-6 main sections with specific, descriptive headings (NOT generic ones like "Overview")
3. Each section should contain 2-3 substantial paragraphs with specific details
4. Include dates, numbers, names, and concrete facts wherever possible
5. Use proper Wikipedia-style citations format [1], [2], etc. (even if hypothetical)
6. Maintain an encyclopedic, neutral tone throughout
7. Target 1200-1600 words for comprehensive coverage
8. Include entity-rich content with proper nouns, technical terms, and specific details

SECTION STRATEGY:
- Choose section headings that are specific to the topic domain
- For events: Background, Timeline, Key figures, Impact, Aftermath
- For organizations: History, Structure, Operations, Services, Controversies
- For people: Early life, Career, Major achievements, Legacy
- For places: Geography, History, Demographics, Economy, Culture
- For concepts: Definition, Development, Applications, Criticism

FORMAT:
# {topic}

[Write a comprehensive 2-3 paragraph introduction that defines the topic, explains its significance, and provides key contextual information. Include specific dates, locations, and quantitative details.]

## [Section 1 - Specific heading related to topic]

[2-3 detailed paragraphs with specific facts, dates, names, and quantitative information. Include proper citations.]

## [Section 2 - Another specific heading]

[2-3 detailed paragraphs continuing the comprehensive coverage.]

## [Section 3 - Third specific heading]

[2-3 detailed paragraphs with continued depth and specificity.]

## [Section 4 - Fourth specific heading]

[2-3 detailed paragraphs maintaining encyclopedic quality.]

## [Section 5 - Fifth specific heading if needed]

[2-3 detailed paragraphs for comprehensive coverage.]

## [Section 6 - Final specific heading if needed]

[2-3 detailed paragraphs completing the comprehensive article.]

Write the complete article now."""


# def post_process_article(content: str, topic: str) -> str:
#     """Post-process article content for better formatting (shared utility)."""
#     if not content:
#         return content
#
#     # Remove excessive whitespace
#     content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)
#     content = re.sub(r"[ \t]+", " ", content)
#
#     # Ensure proper heading format
#     lines = content.split("\n")
#     processed_lines = []
#
#     for line in lines:
#         line = line.strip()
#         if not line:
#             processed_lines.append("")
#             continue
#
#         # Fix heading formatting
#         if line.startswith("#"):
#             # Ensure space after hash
#             line = re.sub(r"^#+", lambda m: m.group() + " ", line)
#             line = re.sub(r"^# +", "# ", line)
#             line = re.sub(r"^## +", "## ", line)
#
#         processed_lines.append(line)
#
#     content = "\n".join(processed_lines)
#
#     # Ensure article starts with main heading
#     if not content.startswith(f"# {topic}"):
#         if content.startswith("# "):
#             content = f"# {topic}\n\n" + content[content.find("\n\n") + 2 :]
#         else:
#             content = f"# {topic}\n\n{content}"
#
#     return content.strip()


def error_article(topic: str, error_msg: str, method: str) -> Article:
    """Create error article when generation fails (shared utility)."""
    return Article(
        title=topic,
        content=f"# {topic}\n\nError generating article: {error_msg}",
        sections={},
        metadata={
            "method": method,
            "error": True,
            "error_message": error_msg,
            "word_count": 0,
            "generation_time": 0.0,
        },
    )


def extract_storm_output(storm_output_dir: Path, topic: str) -> tuple:
    """Extract STORM output content and metadata (shared utility)."""
    try:
        # Look for the main article file
        article_file = storm_output_dir / f"{topic.replace(' ', '_')}.md"
        if not article_file.exists():
            # Try alternative naming
            possible_files = list(storm_output_dir.glob("*.md"))
            if possible_files:
                article_file = possible_files[0]
            else:
                raise FileNotFoundError(f"No article file found in {storm_output_dir}")

        # Read content
        with open(article_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract metadata if available
        metadata = {"storm_output_dir": str(storm_output_dir)}

        # Look for metadata files
        meta_files = list(storm_output_dir.glob("*metadata*")) + list(
            storm_output_dir.glob("*config*")
        )
        for meta_file in meta_files:
            try:
                if meta_file.suffix == ".json":
                    import json

                    with open(meta_file, "r") as f:
                        file_meta = json.load(f)
                        metadata.update(file_meta)
            except Exception as e:
                logger.warning(f"Failed to read metadata from {meta_file}: {e}")

        return content, metadata

    except Exception as e:
        logger.error(f"Failed to extract STORM output: {e}")
        return f"# {topic}\n\nError reading STORM output: {e}", {"error": str(e)}


def build_rag_prompt(topic: str, context: str) -> str:
    """Build RAG prompt with retrieved context (shared utility)."""
    return f"""Write a comprehensive Wikipedia-style article about "{topic}" using the provided context.

Context Information:
{context}

Instructions:
1. Use the provided context as your primary source of information
2. Create a well-structured article with clear sections
3. Include specific facts, dates, and details from the context
4. Maintain a neutral, encyclopedic tone
5. Cite information appropriately
6. Target 1200-1600 words

Write a comprehensive article about {topic}:"""


def enhance_content_prompt(topic: str, content: str) -> str:
    """Build enhancement prompt for short content (shared utility)."""
    return f"""The following article about "{topic}" needs to be enhanced and expanded to meet Wikipedia standards.

Current article:
{content}

Please expand this article to be more comprehensive. Add more specific details, facts, dates, and sections. Target 1200-1600 words with 4-6 well-structured sections.

Enhanced article:"""


def validate_article_quality(content: str, min_words: int = 800) -> dict:
    """Validate article quality and return metrics (shared utility)."""
    if not content:
        return {"valid": False, "reason": "Empty content", "word_count": 0}

    word_count = len(content.split())

    # Check minimum word count
    if word_count < min_words:
        return {
            "valid": False,
            "reason": f"Too short ({word_count} words, minimum {min_words})",
            "word_count": word_count,
        }

    # Check for proper heading structure
    lines = content.split("\n")
    has_main_heading = any(line.strip().startswith("# ") for line in lines)
    has_sections = any(line.strip().startswith("## ") for line in lines)

    if not has_main_heading:
        return {
            "valid": False,
            "reason": "Missing main heading",
            "word_count": word_count,
        }

    if not has_sections:
        return {
            "valid": False,
            "reason": "Missing section headings",
            "word_count": word_count,
        }

    return {
        "valid": True,
        "word_count": word_count,
        "has_main_heading": has_main_heading,
        "has_sections": has_sections,
    }


def merge_results_with_existing(
    existing_results: Dict,
    all_topics: List[str],
    direct_results: List,
    storm_results: List,
    rag_results: List,
    methods: List[str],
) -> Dict:
    """Merge new results with existing completed results."""

    # Start with existing results structure
    all_results = existing_results.get("results", {})

    # Ensure all topics have entries
    for topic in all_topics:
        if topic not in all_results:
            all_results[topic] = {}

    # Process direct results
    if "direct" in methods:
        # Add new direct results
        for result in direct_results:
            topic = result.title
            all_results[topic]["direct"] = {
                "success": True,
                "word_count": result.metadata.get("word_count", 0),
                "article": result,
            }

        # Ensure all topics have direct entries (mark missing as not found)
        for topic in all_topics:
            if "direct" not in all_results[topic]:
                # Check if it should be completed (this handles the case where
                # the topic was completed but not in our current batch)
                all_results[topic]["direct"] = {
                    "success": False,
                    "error": "Direct result not found in current batch",
                }

    # Process storm results
    if "storm" in methods:
        # Add new storm results
        for result in storm_results:
            topic = result.title
            all_results[topic]["storm"] = {
                "success": True,
                "word_count": result.metadata.get("word_count", 0),
                "article": result,
            }

        # Ensure all topics have storm entries
        for topic in all_topics:
            if "storm" not in all_results[topic]:
                all_results[topic]["storm"] = {
                    "success": False,
                    "error": "STORM result not found in current batch",
                }

    # Process rag results
    if "rag" in methods:
        # Add new rag results
        for result in rag_results:
            topic = result.title
            all_results[topic]["rag"] = {
                "success": True,
                "word_count": result.metadata.get("word_count", 0),
                "article": result,
            }

        # Ensure all topics have rag entries
        for topic in all_topics:
            if "rag" not in all_results[topic]:
                all_results[topic]["rag"] = {
                    "success": False,
                    "error": "RAG result not found in current batch",
                }

    return all_results


def setup_output_directory(args) -> Path:
    """Setup output directory for new or resumed experiments."""
    from src.utils.output_manager import OutputManager

    if args.resume_dir:
        # Resume from specific directory
        resume_dir = OutputManager.verify_resume_dir(args.resume_dir)
        logger.info(f"📂 Resuming from specified directory: {resume_dir}")
        return Path(resume_dir)

    # If no resume_dir specified, create new experiment directory using OutputManager
    output_path = OutputManager.create_output_dir(
        args.backend, args.methods, args.num_topics
    )
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Created new run directory: {output_dir}")
    return output_dir


def make_serializable(obj):
    if hasattr(obj, "__dict__"):
        return {k: make_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        return obj
