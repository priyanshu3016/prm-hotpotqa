"""
retriever.py
------------
Builds a FAISS index over paragraph embeddings and supports iterative
multi-hop retrieval. Each "hop" is a separate nearest-neighbour search.

Why FAISS?
  - It stores dense vectors and finds the k nearest neighbours in milliseconds
    even over millions of entries — much faster than brute-force cosine search.
  - faiss-cpu is enough for HotpotQA (~5 M paragraphs in the full wiki dump,
    but we only index the 10 distractor paragraphs per question at eval time).

Why sentence-transformers?
  - all-MiniLM-L6-v2 turns a sentence into a 384-dimensional float vector that
    captures semantic meaning. "Who directed Inception?" and "Christopher Nolan
    directed Inception" are neighbours in that space.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, free
EMBED_DIM   = 384
TOP_K       = 5   # retrieve this many candidates per hop before PRM pruning


class Retriever:
    """
    Wraps a FAISS flat index for per-question retrieval.

    Usage pattern (called once per question):
        retriever = Retriever()
        retriever.build_index(paragraphs)          # index the 10 distractor paragraphs
        hop1_results = retriever.retrieve(query)   # semantic nearest-neighbour search
        hop2_results = retriever.retrieve(bridge)  # second hop with bridge entity
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        # Load embedding model once; reuse across all questions
        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        # FAISS index (rebuilt for each question's 10 paragraphs)
        self.index: faiss.IndexFlatIP  = None
        self.paragraphs: List[str] = []     # keep originals for reference
        self.paragraph_ids: List[str] = []  # e.g. Wikipedia titles

    # ── Index construction ───────────────────────────────────────────────────

    def build_index(self, paragraphs: List[str], ids: List[str] = None):
        """
        Embed all paragraphs and store them in a FAISS IndexFlatIP.

        IndexFlatIP = Inner Product (equivalent to cosine similarity when
        vectors are L2-normalised, which we do below).

        Args:
            paragraphs: list of paragraph text strings
            ids:        optional list of identifiers (e.g. Wikipedia titles)
        """
        self.paragraphs    = paragraphs
        self.paragraph_ids = ids or [str(i) for i in range(len(paragraphs))]

        # Encode: shape (n_paragraphs, 384)
        embeddings = self.encoder.encode(
            paragraphs,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → cosine sim = dot product
            show_progress_bar=False,
        ).astype("float32")

        # Build flat index (no compression, exact search — fine for ≤10 paras)
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.index.add(embeddings)
        logger.debug(f"Index built with {self.index.ntotal} vectors")

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Embed the query and return the top-k most similar paragraphs.

        Returns a list of dicts:
            {"text": str, "id": str, "score": float, "index": int}
        The score is cosine similarity in [0, 1].
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Call build_index() before retrieve()")

        # Encode query as a single normalised vector
        q_vec = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # FAISS search: returns (scores, indices) both shape (1, top_k)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:   # FAISS pads with -1 when k > ntotal
                continue
            results.append({
                "text":  self.paragraphs[idx],
                "id":    self.paragraph_ids[idx],
                "score": float(score),   # cosine similarity
                "index": int(idx),
            })
        return results

    # ── Sub-question / hop controller ────────────────────────────────────────

    def multi_hop_retrieve(
        self,
        sub_questions: List[str],
        original_question: str,
        top_k: int = TOP_K,
    ) -> List[Dict]:
        """
        Iterative 2-hop retrieval.

        Hop 1: search with sub_questions[0]  (the "bridge" sub-question)
        Hop 2: search with sub_questions[1]  (the "answer" sub-question)
                and also with the original question for coverage

        Returns deduplicated list of retrieved paragraphs across all hops,
        each tagged with which hop it came from.
        """
        all_results = []
        seen_ids    = set()

        for hop_idx, sub_q in enumerate(sub_questions[:2]):  # max 2 hops
            results = self.retrieve(sub_q, top_k=top_k)
            for r in results:
                if r["id"] not in seen_ids:
                    r["hop"] = hop_idx + 1
                    all_results.append(r)
                    seen_ids.add(r["id"])

        # Final pass: also search with the full question to improve recall
        for r in self.retrieve(original_question, top_k=top_k):
            if r["id"] not in seen_ids:
                r["hop"] = 0   # "coverage" hop
                all_results.append(r)
                seen_ids.add(r["id"])

        return all_results
