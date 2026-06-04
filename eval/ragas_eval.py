
"""
ragas_eval.py
-------------
Runs the full evaluation over held-out HotpotQA questions.

Features:
- 500-question evaluation by default
- 5 RAGAS metrics
- 95% bootstrap confidence intervals
- checkpointing and resume support
- true round-robin RAGAS judge LLM across:
  Gemini -> Groq -> Cohere -> Mistral -> ...

Usage:
    python eval/ragas_eval.py --threshold 0.4
    python eval/ragas_eval.py --threshold 0.6
    python eval/ragas_eval.py --threshold 0.4 --n 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_community.chat_models import ChatOpenAI
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

load_dotenv()

from src.ragas_router import get_ragas_llm
from langchain_core.outputs import Generation, LLMResult
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Optional provider packages. We import lazily in provider factories so the file
# can still be inspected/imported even if one extra provider package is missing.
# The eval loop will fail with a clear message only if that provider is actually used.

from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline import MultiHopQAPipeline
from src.prm import SEED

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
N_EVAL = 500
N_BOOT = 1000
BOOT_CI = 95
RESULTS_DIR = Path("results")
CHECKPOINT_EVERY = 25

PIPELINE_TIMEOUT_WARN = 0  # kept for clarity; no hard timeout in this script

# RAGAS judge rotation order.
PROVIDERS = ("groq", "openrouter")
PROVIDER_MAX_RETRIES = 3
RETRY_BASE = 2.0

# Reuse the same evaluation wrapper object for all metrics.
# It rotates once per LLM request, not once per metric.
_RAGAS_LLM = None


# ── Utilities ────────────────────────────────────────────────────────────────

def _atomic_json_dump(payload: Any, path: Path) -> None:
    """Write JSON atomically so a crash does not leave a partial checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _extract_text(message: Any) -> str:
    """Best-effort extraction of text from a LangChain chat response."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message.strip()

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()

    # Some providers may return a list of content blocks.
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        joined = "".join(parts).strip()
        if joined:
            return joined

    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text.strip()

    return str(message).strip()


def _prompt_to_text(prompt: Any) -> str:
    """Convert a LangChain PromptValue or plain object to text."""
    if hasattr(prompt, "to_string"):
        try:
            return prompt.to_string()
        except Exception:
            pass
    return str(prompt)


def _looks_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "429",
            "rate limit",
            "quota",
            "resource exhausted",
            "temporarily unavailable",
            "service unavailable",
            "timeout",
            "connection reset",
            "too many requests",
        )
    )


def _build_langchain_model(provider: str):
    """Create the LangChain chat model for a specific provider."""
    provider = provider.lower().strip()

    if provider == "groq":
        return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    elif provider == "openrouter":
        return ChatOpenAI(
        model="qwen/qwen3-32b",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )
    

    raise ValueError(f"Unknown provider: {provider}")



    """
    A RAGAS-compatible LLM wrapper that rotates providers on every request.

    This is intentionally duck-typed to work with RAGAS 0.2.x metric wrappers.
    It exposes generate_text/agenerate_text and a few compatibility aliases.
    """

    def __init__(
        self,
        providers: Sequence[str] = PROVIDERS,
        max_retries: int = PROVIDER_MAX_RETRIES,
        retry_base: float = RETRY_BASE,
    ) -> None:
        self.providers = tuple(providers)
        self.max_retries = max_retries
        self.retry_base = retry_base
        self._lock = threading.Lock()
        self._idx = 0
        self._clients: Dict[str, Any] = {}

    def _next_provider(self) -> str:
        with self._lock:
            provider = self.providers[self._idx]
            self._idx = (self._idx + 1) % len(self.providers)
            return provider

    def _get_client(self, provider: str):
        if provider not in self._clients:
            self._clients[provider] = _build_langchain_model(provider)
        return self._clients[provider]

    def _invoke_provider_once(
        self,
        provider: str,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        client = self._get_client(provider)
        prompt_text = _prompt_to_text(prompt)

        generations: List[List[Generation]] = []
        for _ in range(max(1, n)):
            kwargs = {}
            if stop:
                kwargs["stop"] = stop

            # Different LangChain models expose different signatures; keep this
            # defensive and simple.
            try:
                message = client.invoke(prompt_text, **kwargs)
            except TypeError:
                message = client.invoke(prompt_text)

            text = _extract_text(message)
            generations.append([Generation(text=text)])

        return LLMResult(
            generations=generations,
            llm_output={"provider": provider, "model": getattr(client, "model", None)},
        )

    def _call_with_failover(
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        last_error: Optional[Exception] = None

        for _ in range(len(self.providers)):
            provider = self._next_provider()

            for attempt in range(self.max_retries):
                try:
                    logger.info(f"RAGAS judge using provider: {provider}")
                    return self._invoke_provider_once(
                        provider=provider,
                        prompt=prompt,
                        n=n,
                        temperature=temperature,
                        stop=stop,
                    )
                except Exception as exc:
                    last_error = exc
                    if attempt < self.max_retries - 1 and _looks_transient(exc):
                        wait = self.retry_base ** attempt
                        logger.warning(
                            f"{provider} failed (attempt {attempt + 1}/{self.max_retries}): "
                            f"{exc}. Retrying in {wait:.1f}s..."
                        )
                        time.sleep(wait)
                        continue
                    logger.warning(f"{provider} failed: {exc}")
                    break

        raise RuntimeError("All RAGAS providers failed") from last_error

    # RAGAS 0.2.x compatibility hooks
    def get_temperature(self, n: int) -> float:
        return 0.01 if n <= 1 else 0.3

    def is_finished(self, response: LLMResult) -> bool:
        try:
            first = response.generations[0][0].text
            return bool(str(first).strip())
        except Exception:
            return True

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return self._call_with_failover(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
        )

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return await asyncio.to_thread(
            self._call_with_failover,
            prompt,
            n,
            temperature,
            stop,
        )

    # Newer/alternative aliases for extra compatibility.
    def generate(
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return self.generate_text(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

    async def agenerate(
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return await self.agenerate_text(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )





# ── RAGAS embeddings ─────────────────────────────────────────────────────────

_emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)

# Attach the shared rotating judge LLM to every metric.
# The same object is reused so rotation happens globally across all calls.
faithfulness.llm = get_ragas_llm()
faithfulness.embeddings = _emb

answer_relevancy.llm = get_ragas_llm()
answer_relevancy.embeddings = _emb

context_precision.llm = get_ragas_llm()
context_precision.embeddings = _emb

context_recall.llm = get_ragas_llm()
context_recall.embeddings = _emb

answer_correctness.llm = get_ragas_llm()
answer_correctness.embeddings = _emb


# ── Data loading ──────────────────────────────────────────────────────────────

def load_hotpotqa_eval(n: int = N_EVAL, seed: int = SEED):
    logger.info("Loading HotpotQA distractor validation split...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    indices = sorted(indices.tolist())
    subset = dataset.select(indices)

    logger.info(f"Sampled {len(subset)} questions")
    return subset


def extract_paragraphs(example: dict):
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]

    paragraphs = []
    ids = []
    for title, sents in zip(titles, sentences):
        para = " ".join(sents)
        paragraphs.append(para)
        ids.append(title)

    return paragraphs, ids


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(scores: List[float], n_boot: int = N_BOOT, ci: float = BOOT_CI):
    scores_arr = np.array(scores, dtype=float)
    if len(scores_arr) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(SEED)

    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(scores_arr, size=len(scores_arr), replace=True)
        boot_means.append(np.mean(sample))

    lo = (100 - ci) / 2
    hi = 100 - lo
    lower = np.percentile(boot_means, lo)
    upper = np.percentile(boot_means, hi)
    return float(np.mean(scores_arr)), float(lower), float(upper)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_path(threshold: float) -> Path:
    tag = str(threshold).replace(".", "")
    return RESULTS_DIR / f"checkpoint_t{tag}.json"


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "meta": {},
            "per_question_results": [],
            "pipeline_outputs": [],
        }

    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    # Backward-compatible support for older list-only checkpoints.
    if isinstance(loaded, list):
        return {
            "meta": {},
            "per_question_results": loaded,
            "pipeline_outputs": [],
        }

    return {
        "meta": loaded.get("meta", {}),
        "per_question_results": loaded.get("per_question_results", []),
        "pipeline_outputs": loaded.get("pipeline_outputs", []),
    }


def _save_checkpoint(
    path: Path,
    threshold: float,
    n_eval: int,
    per_question_results: List[Dict[str, Any]],
    pipeline_outputs: List[Dict[str, Any]],
) -> None:
    payload = {
        "meta": {
            "threshold": threshold,
            "seed": SEED,
            "n_eval": n_eval,
            "updated_at_unix": time.time(),
        },
        "per_question_results": per_question_results,
        "pipeline_outputs": pipeline_outputs,
    }
    _atomic_json_dump(payload, path)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(threshold: float, n_eval: int = N_EVAL, checkpoint_every: int = CHECKPOINT_EVERY):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    eval_data = load_hotpotqa_eval(n=n_eval)
    pipeline = MultiHopQAPipeline(threshold=threshold)
    logger.info(f"Running pipeline with PRM threshold t={threshold}")

    checkpoint_file = _checkpoint_path(threshold)
    checkpoint = _load_checkpoint(checkpoint_file)

    per_question_results: List[Dict[str, Any]] = checkpoint["per_question_results"]
    pipeline_outputs: List[Dict[str, Any]] = checkpoint["pipeline_outputs"]

    if checkpoint["meta"]:
        saved_n = checkpoint["meta"].get("n_eval")
        saved_threshold = checkpoint["meta"].get("threshold")
        if saved_n is not None and int(saved_n) != int(n_eval):
            logger.warning(
                f"Checkpoint n_eval={saved_n} differs from requested n_eval={n_eval}. "
                "Proceeding with the requested n_eval."
            )
        if saved_threshold is not None and float(saved_threshold) != float(threshold):
            logger.warning(
                f"Checkpoint threshold={saved_threshold} differs from requested threshold={threshold}. "
                "Proceeding with the requested threshold."
            )

    start_idx = min(len(per_question_results), len(eval_data))
    if start_idx > 0:
        logger.info(f"Resuming from checkpoint at question index {start_idx}")

    for idx in tqdm(range(start_idx, len(eval_data)), desc=f"Evaluating (t={threshold})"):
        example = eval_data[idx]
        question = example["question"]
        gold_answer = example["answer"]
        gold_titles = example["supporting_facts"]["title"]

        paragraphs, ids = extract_paragraphs(example)

        try:
            result = pipeline.run(question, paragraphs, paragraph_ids=ids)
        except Exception as e:
            logger.warning(f"Pipeline failed for '{question[:60]}': {e}")
            result = {
                "question": question,
                "answer": "Error",
                "contexts": [""],
                "kept": [],
                "pruned": [],
                "sub_questions": [],
            }

        per_question_results.append(
            {
                "question": question,
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truths": [gold_answer],
                "reference": gold_answer,
            }
        )

        pipeline_outputs.append(
            {
                "question": question,
                "answer": result["answer"],
                "gold_answer": gold_answer,
                "gold_titles": gold_titles,
                "kept": result.get("kept", []),
                "pruned": result.get("pruned", []),
                "sub_questions": result.get("sub_questions", []),
                "threshold": threshold,
            }
        )

        if len(per_question_results) % checkpoint_every == 0:
            _save_checkpoint(
                checkpoint_file,
                threshold=threshold,
                n_eval=n_eval,
                per_question_results=per_question_results,
                pipeline_outputs=pipeline_outputs,
            )

    # Save the final checkpoint before evaluation.
    _save_checkpoint(
        checkpoint_file,
        threshold=threshold,
        n_eval=n_eval,
        per_question_results=per_question_results,
        pipeline_outputs=pipeline_outputs,
    )

    # ── RAGAS evaluation ──────────────────────────────────────────────────────
    logger.info("Running RAGAS metrics...")

    ragas_dataset = Dataset.from_list(per_question_results)

    from ragas import RunConfig

    run_config = RunConfig(timeout=600, max_retries=3, max_workers=1)

    scores = evaluate(
    ragas_dataset,
    metrics=[
        faithfulness,
    ],
    run_config=run_config,
    )

    per_q_df = scores.to_pandas()

    metric_names = [
    "faithfulness",
    ]

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    ci_results = {}
    for metric in metric_names:
        if metric not in per_q_df.columns:
            logger.warning(f"Metric {metric} not found in RAGAS output")
            continue
        col = per_q_df[metric].dropna().tolist()
        mean, lo, hi = bootstrap_ci(col)
        ci_results[metric] = {
            "mean": round(mean, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "ci_str": f"{mean:.4f} [{lo:.4f}, {hi:.4f}]",
        }
        logger.info(f"{metric:25s}: {ci_results[metric]['ci_str']}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    tag = str(threshold).replace(".", "")
    output = {
        "threshold": threshold,
        "seed": SEED,
        "n_questions": len(per_question_results),
        "n_bootstrap": N_BOOT,
        "metrics": ci_results,
    }

    json_path = RESULTS_DIR / f"results_t{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    rows = []
    for metric, vals in ci_results.items():
        rows.append(
            {
                "system": f"PRM t={threshold}",
                "metric": metric,
                "mean": vals["mean"],
                "ci_lo": vals["ci_lo"],
                "ci_hi": vals["ci_hi"],
            }
        )

    csv_path = RESULTS_DIR / f"results_t{tag}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    raw_path = RESULTS_DIR / f"raw_scores_t{tag}.csv"
    per_q_df.to_csv(raw_path, index=False)

    outputs_path = RESULTS_DIR / f"pipeline_outputs_t{tag}.jsonl"
    with open(outputs_path, "w", encoding="utf-8") as f:
        for record in pipeline_outputs:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved: {json_path}, {csv_path}, {raw_path}, {outputs_path}")
    return output


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        choices=[0.4, 0.6],
    )
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Number of questions to evaluate (use 10 for testing)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help="Write a checkpoint every N processed questions",
    )
    args = parser.parse_args()
    run_evaluation(args.threshold, n_eval=args.n, checkpoint_every=args.checkpoint_every)
