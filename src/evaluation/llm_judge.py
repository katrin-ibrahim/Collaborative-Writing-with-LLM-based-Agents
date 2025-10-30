import sys
from textwrap import dedent

import json
import logging
from prometheus_eval import PrometheusEval
from transformers import AutoModelForCausalLM

# ----------------------
# Config
# ----------------------
MODEL_NAME = "prometheus-eval/prometheus-7b-v2.0"
MAX_TOKENS = 1024

# ----------------------
# Prompt Construction
# ----------------------
RUBRIC_STRING = dedent(
    """
1. Interest Level:
- 1: Not engaging at all; no attempt to capture the reader’s attention.
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
- 1: Severely lacking; offers little to no coverage of the topic’s primary aspects, resulting in a very narrow perspective.
- 2: Partial coverage; includes some of the topic’s main aspects but misses others, resulting in an incomplete portrayal.
- 3: Acceptable breadth; covers most main aspects, though it may stray into minor unnecessary details or overlook some relevant points.
- 4: Good coverage; achieves broad coverage of the topic, hitting on all major points with minimal extraneous information.
- 5: Exemplary in breadth; delivers outstanding coverage, thoroughly detailing all crucial aspects of the topic without including irrelevant information.
"""
)

# ----------------------
# Model + Scorer Setup
# ----------------------
logging.basicConfig(level=logging.INFO)


def load_judge():
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            load_in_4bit=True,
        )
        print("Model device:", next(model.parameters()).device)
        judge = PrometheusEval(model=model)
        return judge
    except Exception as e:
        logging.error(f"Failed to load model or PrometheusEval: {e}")
        sys.exit(1)


judge = load_judge()

# ----------------------
# Scoring Function
# ----------------------


def score_articles(article_texts):
    """
    Score one or more articles. Accepts a string or a list of strings.
    Returns a list of dicts (even for a single article).
    """
    if isinstance(article_texts, str):
        article_texts = [article_texts]
    logging.info(f"Scoring {len(article_texts)} article(s)...")
    try:
        scores, feedbacks = judge.absolute_grade(
            instructions=[
                "Evaluate the article below based on four criteria: Interest Level, Coherence and Organization, Relevance and Focus, Broad Coverage. Use the rubric provided."
            ],
            responses=article_texts,
            rubric=RUBRIC_STRING,
            reference_answers=[""] * len(article_texts),
        )
        results = []
        for i in range(len(article_texts)):
            results.append(
                {
                    "interest_level": scores[i][0],
                    "coherence_organization": scores[i][1],
                    "relevance_focus": scores[i][2],
                    "broad_coverage": scores[i][3],
                    "justification": feedbacks[i],
                }
            )
        return results
    except Exception as e:
        logging.error(f"Scoring failed: {e}")
        return None


# ----------------------
# Optional: CLI entry point
# ----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python llm_judge.py <article_file1> [<article_file2> ...]")
        sys.exit(1)
    articles = []
    for path in sys.argv[1:]:
        with open(path) as f:
            articles.append(f.read())
    results = score_articles(articles)
    if results is not None:
        print(json.dumps(results, indent=2))
    else:
        print("Scoring failed.")
