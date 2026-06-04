"""
pipeline.py
-----------
End-to-end multi-hop QA pipeline:
  1. Decompose the question into sub-questions via an LLM
  2. For each hop: retrieve → PRM-score → threshold-prune
  3. Concatenate surviving context → synthesize final answer via LLM

The pipeline works with any OpenAI-compatible API endpoint.
Set OPENAI_API_KEY in a .env file or as an environment variable.

For local use without an API key, set USE_LOCAL_LLM=True in the config;
this falls back to google/flan-t5-base via HuggingFace transformers.
"""

import os
import logging
from typing import List, Dict, Tuple

from src.retriever import Retriever
from src.prm import ProcessRewardModel, SEED


logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
LLM_MODEL    = "gpt-3.5-turbo"    # cheap, fast, enough for this task
MAX_CONTEXT  = 3000               # max characters of context to pass to LLM
USE_LOCAL_LLM = True             # flip to True to avoid API costs


# ── LLM helper ───────────────────────────────────────────────────────────────

def _call_llm(prompt: str, system: str = "") -> str:
    """
    Thin wrapper around the OpenAI chat endpoint.
    Falls back to flan-t5 if USE_LOCAL_LLM is set.
    """
    if USE_LOCAL_LLM:
        return _call_local_llm(prompt)

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,    # deterministic — required for reproducibility
        seed=SEED,
    )
    return response.choices[0].message.content.strip()


def _call_local_llm(prompt: str) -> str:
    """Use Ollama Mistral locally."""
    import requests
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    return response.json()["response"].strip()


# ── Sub-question decomposition ────────────────────────────────────────────────

DECOMPOSE_SYSTEM = (
    "You are a question decomposition expert. "
    "Your ONLY job is to split a question into EXACTLY 2 sub-questions. "
    "Rules:\n"
    "1. Output EXACTLY 2 lines, nothing else.\n"
    "2. Each line must be a question ending with ?\n"
    "3. Do NOT answer the question.\n"
    "4. Do NOT explain anything.\n"
    "5. Do NOT add numbering or bullets.\n"
    "Example input: What nationality is the director of Inception?\n"
    "Example output:\n"
    "Who directed Inception?\n"
    "What is the nationality of Christopher Nolan?\n"
)


def decompose_question(question: str) -> List[str]:
    """
    Break a multi-hop question into 2 sequential sub-questions.

    Example:
        Q: "What nationality is the director of Inception?"
        Sub-Q1: "Who directed Inception?"
        Sub-Q2: "What nationality is Christopher Nolan?"

    We ask the LLM to do this — it knows the structure of bridging questions.
    """
    prompt = (
    f"Split this question into exactly 2 sub-questions. "
    f"Output only 2 lines, each a question ending with ?. "
    f"Do not answer. Do not explain.\n"
    f"Question: {question}\n"
    f"Sub-questions:"
)
    raw    = _call_llm(prompt, system=DECOMPOSE_SYSTEM)

    # Parse: take first 2 non-empty lines
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    sub_qs = lines[:2]

    # Fallback: if decomposition failed, use the original question twice
    if len(sub_qs) < 2:
        logger.warning(f"Decomposition returned < 2 sub-questions: {raw!r}")
        sub_qs = [question, question]

    logger.debug(f"Sub-questions: {sub_qs}")
    return sub_qs


# ── Answer synthesis ──────────────────────────────────────────────────────────

ANSWER_SYSTEM = (
    "You are a factual question answering system. "
    "Answer the question using ONLY the provided context. "
    "Be concise — answer in one sentence or a short phrase. "
    "If the context does not contain the answer, say 'Insufficient context'."
)


def synthesize_answer(question: str, context_chunks: List[str]) -> str:
    """
    Generate a final answer from the question + kept context paragraphs.

    We cap context length to avoid hitting token limits.
    Faithfulness (a RAGAS metric) measures whether the answer stays within
    the context — so we instruct the model to use ONLY the context.
    """
    # Join all context chunks, truncate to MAX_CONTEXT chars
    context_text = "\n\n".join(context_chunks)[:MAX_CONTEXT]

    prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return _call_llm(prompt, system=ANSWER_SYSTEM)


# ── Main pipeline ─────────────────────────────────────────────────────────────

class MultiHopQAPipeline:
    """
    Orchestrates the full retrieval-augmented multi-hop QA pipeline.

    Args:
        threshold: PRM pruning threshold (0.4 or 0.6 for the two ablations)
    """

    def __init__(self, threshold: float = 0.4):
        self.retriever = Retriever()
        self.prm       = ProcessRewardModel(threshold=threshold)
        self.threshold = threshold

    def run(
        self,
        question: str,
        paragraphs: List[str],
        paragraph_ids: List[str]  = None,
    ) -> Dict:
        """
        Full pipeline for one question.

        Args:
            question:      the original multi-hop question
            paragraphs:    the 10 distractor paragraphs from HotpotQA
            paragraph_ids: optional titles/ids for each paragraph

        Returns a dict with all intermediate results (for RAGAS + analysis):
            {
                "question":      str,
                "answer":        str,          ← the generated answer
                "contexts":      List[str],     ← paragraphs passed to LLM
                "sub_questions": List[str],
                "hop_results":   List[Dict],    ← all retrieved + PRM scores
                "kept":          List[Dict],
                "pruned":        List[Dict],
            }
        """
        # Step 1: Build FAISS index over this question's paragraphs
        self.retriever.build_index(paragraphs, ids=paragraph_ids)

        # Step 2: Decompose into sub-questions
        sub_questions = decompose_question(question)

        # Step 3: Multi-hop retrieval + PRM pruning
        all_kept   = []
        all_pruned = []
        all_retrieved = []

        for hop_idx, sub_q in enumerate(sub_questions):
            # Retrieve top-k for this sub-question
            retrieved = self.retriever.retrieve(sub_q)

            # PRM gate: score and prune
            kept, pruned = self.prm.prune(sub_q, retrieved)

            # Tag which hop each came from
            for r in kept:   r["hop"] = hop_idx + 1
            for r in pruned: r["hop"] = hop_idx + 1

            all_kept.extend(kept)
            all_pruned.extend(pruned)
            all_retrieved.extend(retrieved)

        # Deduplicate kept paragraphs by paragraph id
        seen      = set()
        kept_dedup = []
        for r in all_kept:
            if r["id"] not in seen:
                kept_dedup.append(r)
                seen.add(r["id"])

        # Step 4: Synthesize answer from kept context
        context_texts = [r["text"] for r in kept_dedup]
        if not context_texts:
            # Nothing survived pruning — fall back to top retrieval result
            logger.warning("All paragraphs pruned; using top-1 fallback")
            context_texts = [all_retrieved[0]["text"]] if all_retrieved else [""]

        answer = synthesize_answer(question, context_texts)

        return {
            "question":      question,
            "answer":        answer,
            "contexts":      context_texts,    # what RAGAS measures against
            "sub_questions": sub_questions,
            "hop_results":   all_retrieved,
            "kept":          kept_dedup,
            "pruned":        all_pruned,
        }
