# Process Reward Model on Multi-Hop QA (HotpotQA)

AIMS DTU Research Internship 2026 — Assignment submission.

## Overview

This repository implements a **Process Reward Model (PRM)** that scores intermediate reasoning steps in a multi-hop retrieval-augmented QA pipeline evaluated on the HotpotQA distractor setting.

### How it works

1. **Decompose**: An LLM breaks each multi-hop question into 2 sequential sub-questions
2. **Retrieve**: FAISS nearest-neighbour search over the 10 distractor paragraphs
3. **PRM Gate**: A cross-encoder scores each retrieved paragraph; those below threshold `t` are discarded
4. **Synthesize**: An LLM generates the final answer from surviving context
5. **Evaluate**: RAGAS computes 5 metrics with 95% bootstrap CIs over 500 questions

### Architecture

| Component | Model | Why |
|-----------|-------|-----|
| Embedder (retrieval) | `all-MiniLM-L6-v2` | Fast 384-dim semantic embeddings |
| PRM (scoring) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Pairwise relevance, trained on MS MARCO |
| Answer generator | `gpt-3.5-turbo` or `flan-t5-base` | Instruction-following, deterministic (T=0) |
| Vector index | `faiss-cpu` IndexFlatIP | Exact cosine search, fast for ≤10 paragraphs |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── retriever.py       # FAISS index, iterative retrieval, hop controller
│   ├── prm.py             # Cross-encoder PRM, threshold pruning
│   └── pipeline.py        # End-to-end: decompose → retrieve → PRM gate → synthesize
├── eval/
│   └── ragas_eval.py      # RAGAS evaluation, bootstrap CIs, CSV/JSON output
├── results/
│   ├── results_t04.json   # t=0.4 scores
│   ├── results_t06.json   # t=0.6 scores
│   └── plots/             # Metric charts
└── notebooks/
    └── analysis.ipynb     # Failure analysis, hop-failure examples, distributions
```

---

## Setup

### Requirements

- Python 3.10+
- Ollama running locally with Mistral pulled (ollama pull mistral)
- No API key needed

### Installation

```bash
git clone https://github.com/<your-username>/prm-hotpotqa
cd prm-hotpotqa
pip install -r requirements.txt
```

### Environment

```bash
cp .env.example .env
# Edit .env and add:  OPENAI_API_KEY=sk-...
```

---

## Reproduction Commands

Run both ablations in sequence (requires ~2–4 hours on CPU, ~45 min with GPU):

```bash
# t=0.4 ablation
python eval/ragas_eval.py --threshold 0.4

# t=0.6 ablation
python eval/ragas_eval.py --threshold 0.6
```

Outputs appear in `results/`:
- `results_t04.json` / `results_t06.json` — metrics + CIs
- `results_t04.csv` / `results_t06.csv` — same in tabular form
- `raw_scores_t04.csv` / `raw_scores_t06.csv` — per-question scores

Generate plots and failure analysis:

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Hardware & Runtime

| Setting | Hardware | Approx. runtime (500 questions) |
|---------|----------|----------------------------------|
| CPU-only | 8-core, 16 GB RAM | ~3.5 hours |
| GPU (T4) | Google Colab Free | ~50 minutes |
| GPU (A100) | Colab Pro / local | ~20 minutes |

The embedding model and cross-encoder both run efficiently on CPU. The bottleneck is the LLM API calls (one decomposition + one synthesis per question).

---

## Fixed Random Seed

All randomness is seeded with `SEED = 42` (set in `src/prm.py`):
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- HotpotQA sampling uses `numpy.random.default_rng(42)`
- Bootstrap CI resampling uses `numpy.random.default_rng(42)`

---

## Results

*(Fill these after running the evaluation)*

| System | Faith. | Ans. Rel. | Ctx. Prec. | Ctx. Rec. | Ans. Corr. |
|--------|--------|-----------|------------|-----------|------------|
| PRM t=0.4 | X.XX [X.XX, X.XX] | | | | |
| PRM t=0.6 | X.XX [X.XX, X.XX] | | | | |

---

## Citation

```bibtex
@inproceedings{yang2018hotpotqa,
  title     = {HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author    = {Yang, Zhilin and others},
  booktitle = {EMNLP},
  year      = {2018}
}
```
