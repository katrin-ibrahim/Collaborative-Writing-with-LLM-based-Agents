"""
LLM Judge for evaluating articles using Ollama backend.
"""

import sys
from pathlib import Path
from textwrap import dedent

import json
import logging
import re
import requests
from typing import Dict, List, Union

logging.basicConfig(level=logging.INFO)

RUBRIC_STRING = dedent(
    """
1. Interest Level:
- 1: Not engaging at all; no attempt to capture the reader's attention.
- 2: Fairly engaging with a basic narrative but lacking depth.
- 3: Moderately engaging with several interesting points.
- 4: Quite engaging with a well-structured narrative and noteworthy points that frequently capture and retain attention.
- 5: Exceptionally engaging throughout, with a compelling narrative that consistently stimulates interest.

2. Coherence and Organization:
- 1: Disorganized; lacks logical structure and coherence.
- 2: Fairly organized; a basic structure is present but not consistently followed.
- 3: Organized; a clear structure is mostly followed with some lapses in coherence.
- 4: Good organization; a clear structure with minor lapses in coherence.
- 5: Excellently organized; the article is logically structured with seamless transitions and a clear argument.

3. Relevance and Focus:
- 1: Off-topic; the content does not align with the headline or core subject.
- 2: Somewhat on topic but with several digressions; the core subject is evident but not consistently adhered to.
- 3: Generally on topic, despite a few unrelated details.
- 4: Mostly on topic and focused; the narrative has a consistent relevance to the core subject with infrequent digressions.
- 5: Exceptionally focused and entirely on topic; the article is tightly centered on the subject, with every piece of information contributing to a comprehensive understanding of the topic.

4. Broad Coverage:
- 1: Severely lacking; offers little to no coverage of the topic's primary aspects, resulting in a very narrow perspective.
- 2: Partial coverage; includes some of the topic's main aspects but misses others, resulting in an incomplete portrayal.
- 3: Acceptable breadth; covers most main aspects, though it may stray into minor unnecessary details or overlook some relevant points.
- 4: Good coverage; achieves broad coverage of the topic, hitting on all major points with minimal extraneous information.
- 5: Exemplary in breadth; delivers outstanding coverage, thoroughly detailing all crucial aspects of the topic without including irrelevant information.
"""
)


def create_evaluation_prompt(article: str) -> str:
    """
    Create the evaluation prompt for the LLM judge.

    Args:
        article: Article text to evaluate

    Returns:
        Formatted prompt string
    """
    return dedent(
        f"""
    You are an expert evaluator. Please evaluate the following article based on the rubric below.

    ARTICLE:
    {article[:5000]}

    RUBRIC:
    {RUBRIC_STRING}

    Return your evaluation as a JSON object with the following structure:
    {{
      "interest_level": <score 1-5>,
      "coherence_organization": <score 1-5>,
      "relevance_focus": <score 1-5>,
      "broad_coverage": <score 1-5>,
      "justification": "Brief explanation of your scores"
    }}

    Provide only the JSON object, no additional text.
    """
    )


def extract_json_from_text(text: str) -> Dict:
    """
    Extract JSON object from text, handling various formats.

    Args:
        text: Text potentially containing JSON

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If no valid JSON found
    """
    code_fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_fence_match:
        try:
            return json.loads(code_fence_match.group(1))
        except json.JSONDecodeError:
            pass

    json_matches = list(
        re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    )
    for match in reversed(json_matches):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON found in response")


def call_ollama(
    prompt: str,
    model: str = "qwen2.5:14b",
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Call Ollama API to generate text.

    Args:
        prompt: Input prompt
        model: Model name (e.g., 'qwen2.5:14b', 'llama3.2')
        host: Ollama server URL
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text

    Raises:
        requests.RequestException: If API call fails
    """
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    return response.json()["response"]


def score_articles(
    article_texts: Union[str, List[str]],
    model: str = "qwen2.5:14b",
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> List[Dict]:
    """
    Score articles using Ollama-based LLM judge.

    Args:
        article_texts: Single article string or list of article strings
        model: Ollama model name
        host: Ollama server URL
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        List of evaluation results as dictionaries
    """
    if isinstance(article_texts, str):
        article_texts = [article_texts]

    results = []
    for idx, article in enumerate(article_texts):
        logging.info(f"Evaluating article {idx + 1}/{len(article_texts)}...")

        prompt = create_evaluation_prompt(article)
        output = None

        try:
            output = call_ollama(
                prompt=prompt,
                model=model,
                host=host,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            result = extract_json_from_text(output)
            results.append(result)
            logging.info(f"Article {idx + 1} evaluated successfully")

        except Exception as e:
            logging.error(f"Failed to evaluate article {idx + 1}: {e}")
            results.append(
                {
                    "error": str(e),
                    "raw_output": output,
                }
            )

    return results


def main():
    """CLI entry point for LLM judge."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate articles using Ollama-based LLM judge"
    )
    parser.add_argument(
        "article_files",
        nargs="+",
        help="Path(s) to article text files",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="qwen2.5:14b",
        help="Ollama model to use (default: qwen2.5:14b)",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path (default: print to stdout)",
    )

    args = parser.parse_args()

    articles = []
    for path_str in args.article_files:
        path = Path(path_str)
        if not path.exists():
            logging.error(f"File not found: {path}")
            sys.exit(1)

        try:
            articles.append(path.read_text(encoding="utf-8"))
        except Exception as e:
            logging.error(f"Failed to read {path}: {e}")
            sys.exit(1)

    results = score_articles(
        article_texts=articles,
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    output_json = json.dumps(results, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        logging.info(f"Results written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
