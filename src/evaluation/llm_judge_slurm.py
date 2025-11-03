import sys
from textwrap import dedent

import json
import logging
import re
import torch
from transformers import BitsAndBytesConfig, pipeline

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "prometheus-eval/prometheus-7b-v2.0"
MAX_TOKENS = 1024

USE_QUANTIZATION = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def create_pipeline():
    """
    Create the text generation pipeline with appropriate settings for SLURM.

    Returns:
        Hugging Face pipeline for text generation
    """
    model_kwargs = {
        "trust_remote_code": True,
    }

    if USE_QUANTIZATION and DEVICE == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
        device_map = "auto"
    elif DEVICE == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        device_map = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32
        device_map = "cpu"

    logging.info(
        f"Loading Prometheus model on {DEVICE} (quantization={USE_QUANTIZATION})..."
    )

    pipe = pipeline(
        task="text-generation",
        model=MODEL_NAME,
        device_map=device_map,
        model_kwargs=model_kwargs,
        return_full_text=False,
    )

    gen_config = getattr(pipe.model, "generation_config", None)
    if gen_config is not None and gen_config.pad_token_id is None:
        gen_config.pad_token_id = gen_config.eos_token_id

    logging.info("Model loaded successfully")
    return pipe


def extract_json_from_text(text: str) -> dict:
    """
    Extract JSON object from text, handling various formats.

    Args:
        text: Text potentially containing JSON

    Returns:
        Parsed JSON dictionary
    """
    # Try to find JSON block in code fence
    code_fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_fence_match:
        try:
            return json.loads(code_fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    json_matches = list(
        re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    )
    for match in reversed(json_matches):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON found in response")


def score_articles(article_texts, pipe):
    """
    Score articles using Prometheus model via transformers pipeline.

    Args:
        article_texts: Single article string or list of article strings
        pipe: Hugging Face pipeline

    Returns:
        List of evaluation results as dictionaries
    """
    if isinstance(article_texts, str):
        article_texts = [article_texts]

    prompt_template = dedent(
        """
    You are an expert evaluator. Please evaluate the following article based on the rubric below.

    ARTICLE:
    {article}

    RUBRIC:
    {rubric}

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

    results = []
    for idx, article in enumerate(article_texts):
        prompt = prompt_template.format(article=article[:5000], rubric=RUBRIC_STRING)

        logging.info(f"Evaluating article {idx + 1}/{len(article_texts)}...")

        gen_kwargs = {
            "max_new_tokens": MAX_TOKENS,
            "do_sample": False,
            "temperature": 0.0,
        }

        try:
            output = pipe(prompt, **gen_kwargs)[0]["generated_text"]
            result = extract_json_from_text(output)
            results.append(result)
            logging.info(f"Article {idx + 1} evaluated successfully")
        except Exception as e:
            logging.error(f"Failed to evaluate article {idx + 1}: {e}")
            results.append(
                {
                    "error": str(e),
                    "raw_output": output if "output" in locals() else None,
                }
            )

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python llm_judge_slurm.py <article_file1> [<article_file2> ...]")
        sys.exit(1)

    pipe = create_pipeline()

    articles = []
    for path in sys.argv[1:]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                articles.append(f.read())
        except FileNotFoundError:
            logging.error(f"File not found: {path}")
            sys.exit(1)

    results = score_articles(articles, pipe)
    print(json.dumps(results, indent=2, ensure_ascii=False))
