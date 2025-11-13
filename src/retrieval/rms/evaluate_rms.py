"""
Evaluate WikiRM (live) vs FaissRM (local) **without labels**.

Metrics per query variant:
- noise        : count of junky pages (disambiguation / list / category)
- diversity    : number of unique documents (doc_id collapsed from chunk_id)
- string_dups  : how many documents repeat in top-k
- time         : retrieval time in seconds

Per base query:
- stability (Jaccard overlap of doc_id sets across paraphrases)

Outputs:
- results/eval_rms.csv
- results/eval_rms_stability.csv
- results/eval_rms_summary.csv
- results/plots/*.png
- cache/retrieval/{faiss|wiki}.jsonl

Usage:
    python tools/eval/evaluate_rms_v4.py --num_topics 25 --top_k 20
"""

import argparse
import csv
import hashlib
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.config import CollaborationConfig, ModelConfig, RetrievalConfig, StormConfig
from src.config.config_context import ConfigContext
from src.retrieval.rms.faiss_rm import FaissRM
from src.retrieval.rms.wiki_rm import WikiRM
from src.utils.data.freshwiki_loader import FreshWikiLoader

# ------------------------- setup -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("eval_rms_v4")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def init_config():
    ConfigContext.initialize(
        model_config=ModelConfig.get_default(),
        retrieval_config=RetrievalConfig.get_default(),
        collaboration_config=CollaborationConfig.get_default(),
        storm_config=StormConfig.get_default(),
        backend="ollama",
    )


# ------------------------- helpers -------------------------
NOISE_PATTERNS = [
    r"\bdisambiguation\b",
    r"(^|\W)list of(\W|$)",
    r"(^|\W)category(\W|$)",
    r"(^|\W)index of(\W|$)",
    r"(^|\W)portal(\W|$)",
]


def paraphrase_query(q: str):
    return [q, f"What is {q}?", f"Overview of {q}", f"{q} article"]


def doc_id_from_chunk(chunk) -> str:
    cid = getattr(chunk, "chunk_id", "") or chunk.get("chunk_id", "")
    return cid.rsplit("_", 1)[0] if "_" in cid else cid


def is_noise(chunk) -> bool:
    """Accepts ResearchChunk or dict."""
    if isinstance(chunk, dict):
        desc = (chunk.get("description") or "").lower()
        url = (chunk.get("url") or "").lower()
    else:
        desc = (chunk.description or "").lower()
        url = (chunk.url or "").lower()
    if any(re.search(p, desc) for p in NOISE_PATTERNS):
        return True
    if "list_of" in url or "category:" in url or "disambiguation" in url:
        return True
    return False


def diversity_docs(ids):
    return len(set(ids))


def count_dups(ids):
    return sum(v > 1 for v in Counter(ids).values())


def jaccard_overlap_docids(list_of_lists):
    sets = [set(x) for x in list_of_lists if x]
    if not sets:
        return 0.0
    inter = set.intersection(*sets)
    uni = set.union(*sets)
    return len(inter) / max(1, len(uni))


# ------------------------- cache -------------------------
def _key(rm, q, k):
    return hashlib.md5(f"{rm}|{q}|{k}".encode()).hexdigest()


def load_cache(path):
    d = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d.update({json.loads(line)["key"]: json.loads(line)["hits"]})
    return d


def append_cache(path, key, hits):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "hits": hits}, ensure_ascii=False) + "\n")


def serialize_chunk(c):
    return {
        "chunk_id": c.chunk_id,
        "doc_id": doc_id_from_chunk(c),
        "description": c.description,
        "url": c.url,
        "rank": c.rank,
        "score": getattr(c, "relevance_score_normalized", None),
    }


# ------------------------- stats + plots -------------------------
def mean_ci_95(vals):
    arr = np.array(vals, float)
    if len(arr) == 0:
        return 0, 0, 0
    m = arr.mean()
    boot = [np.mean(np.random.choice(arr, len(arr), True)) for _ in range(500)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return m, lo, hi


def save_summary_and_plots(out_dir, plot_dir):
    det = list(csv.DictReader(open(out_dir / "eval_rms.csv", encoding="utf-8")))
    metrics = defaultdict(list)
    for r in det:
        rm = r["rm"]
        for k in ["noise", "diversity", "string_dups", "time"]:
            if r.get(k):
                metrics[(rm, k)].append(float(r[k]))

    rows = []
    for (rm, m), vals in metrics.items():
        mean, lo, hi = mean_ci_95(vals)
        rows.append(
            {
                "rm": rm,
                "metric": m,
                "mean": f"{mean:.3f}",
                "ci95_lo": f"{lo:.3f}",
                "ci95_hi": f"{hi:.3f}",
            }
        )
    with open(out_dir / "eval_rms_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rm", "metric", "mean", "ci95_lo", "ci95_hi"])
        w.writeheader()
        w.writerows(rows)

    names = {
        "noise": "Noise (junk pages)",
        "diversity": "Diversity (unique docs)",
        "string_dups": "Repeated Docs",
        "time": "Retrieval Time (s)",
    }
    for metric in ["noise", "diversity", "string_dups", "time"]:
        data = {
            rm: [
                float(x["mean"])
                for x in rows
                if x["metric"] == metric and x["rm"] == rm
            ][0]
            for rm in ["faiss", "wiki"]
            if any(x["metric"] == metric and x["rm"] == rm for x in rows)
        }
        if not data:
            continue
        x = np.arange(len(data))
        means = list(data.values())
        labels = list(data.keys())
        fig = plt.figure()
        plt.bar(
            x,
            means,
            color=["#4575b4", "#f46d43"],
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )
        plt.xticks(x, labels, fontsize=12)
        plt.title(names[metric], fontsize=14, fontweight="bold")
        plt.ylabel(names[metric], fontsize=12)
        plt.xlabel("Retriever (rm)", fontsize=12)
        for xi, m in zip(x, means):
            plt.text(xi, m, f"{m:.2f}", ha="center", va="bottom", fontsize=11)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}.png", dpi=200)
        plt.close(fig)
    log.info("Saved summary and labeled plots.")


# ------------------------- main -------------------------
def run_eval(num_topics=25, top_k=20, save_every=25, seed=42):
    set_seed(seed)
    init_config()
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    plot = out / "plots"
    plot.mkdir(parents=True, exist_ok=True)
    cache = Path("cache/retrieval")
    cache.mkdir(parents=True, exist_ok=True)

    loader = FreshWikiLoader()
    queries = [e.topic for e in loader.load_topics(num_topics)]
    faiss, wiki = FaissRM(), WikiRM()
    f_cache, w_cache = cache / "faiss.jsonl", cache / "wiki.jsonl"
    F, W = load_cache(f_cache), load_cache(w_cache)
    results, stability = [], []

    pbar = tqdm(total=len(queries), desc="Base queries", ncols=100)
    for i, base in enumerate(queries, 1):
        paraphrases = paraphrase_query(base)
        f_docids, w_docids = [], []
        for pq in paraphrases:
            # ---- Faiss ----
            k = _key("faiss", pq, top_k)
            t0 = time.time()
            if k in F:
                hits = F[k]
                t = 0
            else:
                chunks = faiss.search(query_or_queries=pq, max_results=top_k) or []
                t = time.time() - t0
                hits = [serialize_chunk(c) for c in chunks]
                append_cache(f_cache, k, hits)
                F[k] = hits
            ids = [h["doc_id"] for h in hits]
            results.append(
                {
                    "base_query": base,
                    "query": pq,
                    "rm": "faiss",
                    "k": top_k,
                    "time": t,
                    "noise": sum(is_noise(h) for h in hits),
                    "diversity": diversity_docs(ids),
                    "string_dups": count_dups(ids),
                }
            )
            f_docids.append(ids)
            # ---- Wiki ----
            k = _key("wiki", pq, top_k)
            t0 = time.time()
            if k in W:
                hits = W[k]
                t = 0
            else:
                chunks = wiki.search(query_or_queries=pq, max_results=top_k) or []
                t = time.time() - t0
                hits = [serialize_chunk(c) for c in chunks]
                append_cache(w_cache, k, hits)
                W[k] = hits
            ids = [h["doc_id"] for h in hits]
            results.append(
                {
                    "base_query": base,
                    "query": pq,
                    "rm": "wiki",
                    "k": top_k,
                    "time": t,
                    "noise": sum(is_noise(h) for h in hits),
                    "diversity": diversity_docs(ids),
                    "string_dups": count_dups(ids),
                }
            )
            w_docids.append(ids)
        stability.append(
            {
                "base_query": base,
                "faiss_stability_doc_overlap": jaccard_overlap_docids(f_docids),
                "wiki_stability_doc_overlap": jaccard_overlap_docids(w_docids),
            }
        )
        if i % save_every == 0 or i == len(queries):
            _save_csvs(results, stability, out)
            save_summary_and_plots(out, plot)
        pbar.update(1)
    pbar.close()
    log.info("Done.")


def _save_csvs(results, stability, out):
    f = out / "eval_rms.csv"
    fields = [
        "base_query",
        "query",
        "rm",
        "k",
        "time",
        "noise",
        "diversity",
        "string_dups",
    ]
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        [w.writerow({k: r.get(k) for k in fields}) for r in results]
    s = out / "eval_rms_stability.csv"
    fields = ["base_query", "faiss_stability_doc_overlap", "wiki_stability_doc_overlap"]
    with open(s, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        [w.writerow(r) for r in stability]
    log.info("Saved CSVs.")


# ------------------------- entry -------------------------
if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for v in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ.setdefault(v, "1")
    try:
        import faiss

        faiss.omp_set_num_threads(1)
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--num_topics", type=int, default=25)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run_eval(
        num_topics=a.num_topics, top_k=a.top_k, save_every=a.save_every, seed=a.seed
    )
