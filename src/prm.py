"""
prm.py
------
Process Reward Model (PRM) that assigns a quality score in [0, 1] to each
retrieved paragraph at every reasoning hop.

Architecture choice: cross-encoder
  - A bi-encoder (like the retriever) encodes query and paragraph separately
    and computes similarity — fast but shallow.
  - A cross-encoder concatenates [query, paragraph] and runs them through a
    transformer jointly — it can see interactions between the two, so it's
    far better at relevance judgement.
  - We use ms-marco-MiniLM-L-6-v2, a cross-encoder fine-tuned on MS MARCO
    passage ranking. It outputs a raw logit; we sigmoid it to get [0, 1].

Threshold pruning:
  - t=0.4: lenient gate — passes most retrieved paragraphs
  - t=0.6: strict gate — passes only clearly relevant paragraphs
  Both are ablated so we can see how aggressiveness affects final metrics.

Fixed seed: SEED = 42 (used wherever randomness appears)
"""

import numpy as np
import torch
import logging
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
SEED       = 42
PRM_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # proven on MS MARCO
THRESHOLD_LOW  = 0.4   # lenient ablation
THRESHOLD_HIGH = 0.6   # strict ablation

# Reproducibility — set everywhere randomness could appear
np.random.seed(SEED)
torch.manual_seed(SEED)


class ProcessRewardModel:
    """
    Scores (query, paragraph) pairs and prunes those below a threshold.

    The cross-encoder was trained on (query, passage) relevance labels from
    MS MARCO, so it already understands "does this passage help answer this
    question?" — we do not need to fine-tune it for HotpotQA.

    If you *did* want to fine-tune: collect (question, gold_paragraph, label=1)
    and (question, distractor_paragraph, label=0) pairs from HotpotQA training
    split, then train with binary cross-entropy on the logit output.
    """

    def __init__(self, model_name: str = PRM_MODEL, threshold: float = THRESHOLD_LOW):
        logger.info(f"Loading PRM cross-encoder: {model_name}")
        # CrossEncoder wraps a BERT-style model for pairwise scoring
        self.model     = CrossEncoder(model_name, max_length=512)
        self.threshold = threshold
        logger.info(f"PRM threshold set to t={threshold}")

    # ── Core scoring ─────────────────────────────────────────────────────────

    def score(self, query: str, paragraphs: List[str]) -> List[float]:
        """
        Score each paragraph against the query.

        The cross-encoder returns raw logits; sigmoid maps them to [0, 1].
        Batch all pairs at once for GPU/CPU efficiency.

        Returns:
            List of floats, one per paragraph, in [0, 1].
        """
        if not paragraphs:
            return []

        # Build list of (query, paragraph) pairs for the cross-encoder
        pairs = [(query, para) for para in paragraphs]

        # Predict: raw logits, shape (n_paragraphs,)
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        # Sigmoid to normalise into [0, 1]
        scores = [float(1 / (1 + np.exp(-s))) for s in raw_scores]
        return scores

    # ── Pruning gate ─────────────────────────────────────────────────────────

    def prune(
        self,
        query: str,
        retrieved: List[Dict],
        threshold: float  = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Score all retrieved paragraphs and split into kept / pruned.

        Args:
            query:     the sub-question for this hop (not the original question)
            retrieved: list of dicts from Retriever.retrieve()
            threshold: override self.threshold if provided

        Returns:
            kept:   paragraphs with score >= threshold (used in the answer)
            pruned: paragraphs below threshold (discarded)
        """
        t = threshold if threshold is not None else self.threshold

        paragraphs = [r["text"] for r in retrieved]
        scores     = self.score(query, paragraphs)

        kept, pruned = [], []
        for result, score in zip(retrieved, scores):
            result["prm_score"] = score   # attach score for logging / analysis
            if score >= t:
                kept.append(result)
            else:
                pruned.append(result)

        logger.debug(
            f"PRM pruned {len(pruned)}/{len(retrieved)} paragraphs "
            f"(threshold t={t:.2f})"
        )
        return kept, pruned

    # ── Batch evaluation (for analysis notebook) ─────────────────────────────

    def score_gold_vs_distractor(
        self,
        question: str,
        gold_paragraphs: List[str],
        distractor_paragraphs: List[str],
    ) -> Dict:
        """
        Compare PRM scores on gold vs distractor paragraphs.
        Used in the analysis notebook to verify the PRM separates them.

        Returns a dict with mean scores and per-paragraph details.
        """
        all_paras  = gold_paragraphs + distractor_paragraphs
        all_scores = self.score(question, all_paras)

        n_gold = len(gold_paragraphs)
        gold_scores       = all_scores[:n_gold]
        distractor_scores = all_scores[n_gold:]

        return {
            "gold_mean":        np.mean(gold_scores),
            "distractor_mean":  np.mean(distractor_scores),
            "gold_scores":      gold_scores,
            "distractor_scores": distractor_scores,
            # separation = how much higher gold scores are on average
            "separation":       np.mean(gold_scores) - np.mean(distractor_scores),
        }
