# APPLE SUPPORT AI

### Evidence-Grounded Customer Support Agent — technical assignment for Hiver

A machine-learning system that classifies AppleSupport customer issues,
retrieves historically similar support cases, generates responses **grounded
in that historical evidence**, and decides safely between auto-handling and
escalation — with quantitative evaluation of every component.

> **Current status: Phases 1–3 complete — data foundation + intent
> classification foundation + embedding classifier (measured).** The
> assignment mandates a phased build; Phases 4–9 are scoped but deliberately
> **not implemented** until each prior phase is working and tested.

---

## ⚠️ Historical-data disclaimer

The dataset contains **historical** customer-support interactions
(Customer Support on Twitter, 2017 era). Historical AppleSupport responses are
used strictly as *evidence* for retrieval, grounding and evaluation — they are
**not** a representation of current Apple policies, prices, warranty terms or
support practice.

---

## Problem statement

Given a customer support message, the system must (1) classify the issue,
(2) retrieve similar historical AppleSupport cases, (3) generate a reply
grounded in them, (4) auto-handle or escalate, and (5) measure all of the
above. Phase 1 delivers the data layer this depends on.

## Architecture (Phase 1 slice)

```
twcs.csv (516 MB, 2.8 M tweets, gitignored)
   │  scripts/find_author_ids.py ──► APPLE_SUPPORT_AUTHOR_ID (.env, never guessed)
   ▼
┌──────────────────────────────────────────────────────────────┐
│ scripts/prepare_data.py                                       │
│  1. scan + schema validation          (src/data/load.py)      │
│  2. AppleSupport thread closure       (memory-safe, chunked) │
│  3. conservative cleaning             (src/data/preprocess…) │
│  4. conversation reconstruction       (src/data/conversat…)  │
│  5. evidence-pair extraction          (customer ↔ reply)     │
│  6. 7-point validation gate           (exit ≠ 0 on failure)  │
└──────────────────────────────────────────────────────────────┘
   ▼ data/processed/
applesupport_tweets.csv · conversations.csv · conversation_pairs.csv
pipeline_run.json · (run_eda.py) eda_summary.json + charts/
   ▼
notebooks/01_data_exploration.ipynb (executed, cross-checked)
web dashboard (Next.js page + /api/phase1/status — this repo's root app)
```

Phases 4–9 (not yet built): FAISS retrieval
(Recall@K) → grounded generation → escalation policy (false-auto rate) →
unified agent → Streamlit app → final report.

## Dataset

Kaggle [`thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
— place `twcs.csv` in `data/raw/` (never committed). Full column semantics
and the AppleSupport account-identification workflow: see
[`data/README.md`](data/README.md).

## Setup

```bash
cd hiver-support-agent
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt                      # pandas, numpy, matplotlib, seaborn, python-dotenv, scikit-learn
cp .env.example .env

# 1) discover the Apple support account id from the data (do not guess it)
python scripts/find_author_ids.py
#    -> set APPLE_SUPPORT_AUTHOR_ID=AppleSupport in .env

# 2) put twcs.csv into data/raw/ (download options in data/README.md)
#    (no Kaggle access? smoke-test instead:)
#    python scripts/make_synthetic_sample.py   # clearly-labelled synthetic sample

# 3) build Phase 1 artifacts (full scale: ~212 s, <1 GB RAM)
python scripts/prepare_data.py

# 4) EDA summary + charts
python scripts/run_eda.py

# 5) Phase 2: golden set + baselines
python scripts/build_golden_set.py sample --n 200
python scripts/build_golden_set.py build      # validate labels, split, manifest
python scripts/train_baselines.py            # metrics -> evaluation/results/

# 6) Phase 3: embedding classifier (installs sentence-transformers/torch)
python scripts/train_embedding_classifier.py  # metrics + reusable model heads
```

Environment variables (`.env`) — full list in `.env.example`:
`APPLE_SUPPORT_AUTHOR_ID` (required), `RAW_DATA_PATH`, `PROCESSED_DIR`,
`RANDOM_SEED`, `CHUNK_SIZE`, `CLOSURE_MAX_PASSES`; `EMBEDDING_MODEL`
(consumed since Phase 3, default `sentence-transformers/all-MiniLM-L6-v2`);
Phase 4+ thresholds (`INTENT_CONFIDENCE_THRESHOLD=0.60` — already measured
as a preview in Phase 3, `RETRIEVAL_THRESHOLD=0.65`, `TOP_K=5`, `LLM_MODEL`,
`LLM_API_KEY`) are parsed but not consumed until the phase that needs them.
**Never commit `.env`.**

## Example output (real run)

`python scripts/prepare_data.py` on the full Kaggle dataset:

```
INFO  prepare_data: AppleSupport author id: 'AppleSupport' | input: .../twcs.csv (synthetic=False)
INFO  src.data.load: Scan complete: 2811774 rows (2811774 unique tweet ids, 0 duplicates),
                    1537843 inbound / 1273931 outbound, 794335 conversation starters
INFO  src.data.load: Closure pass 12/12: +41 rows recorded (238754 total)
INFO  src.data.preprocessing: Preprocessing: 238754 -> 237967 rows (dropped: 787 empty text)
INFO  src.data.conversations: Reconstructed 81156 conversations from 237967 tweets
INFO  src.data.conversations: Filtered to 80966 / 81156 conversations involving author 'AppleSupport'
INFO  src.data.conversations: Extracted 106137 customer↔support pairs
INFO  prepare_data: VALIDATION pair_ids_exist_in_tweets  PASS — 106137 pair endpoint ids resolved
... (7/7 PASS)
INFO  prepare_data: PHASE 1 PREPARATION COMPLETE in 211.8s
```

A reconstructed pair (from `conversation_pairs.csv`):

| conversation_id | customer_message | support_response |
|---|---|---|
| 14838 | "iOS 11 is draining the battery on my iPhone 7 twice as fast as iOS 10. Help!" | "We'd like to gather some more information for better troubleshooting. Can you DM us the country you are locate…" |

## Metrics (Phases 1–3 — actually measured)

| metric | value |
|---|---|
| tweets scanned (full corpus) | 2,811,774 |
| AppleSupport conversations | 80,966 |
| AppleSupport corpus tweets | 237,745 (130,612 inbound / 107,133 outbound) |
| customer↔support evidence pairs | 106,137 |
| avg / median / max turns per conversation | 2.94 / 2 / 282 |
| avg message length | 100.8 chars · 19.1 words |
| intent taxonomy classes (Phase 2) | 10 (EDA-anchored + unmatched-pool split) |
| golden set (Phase 2) | 200 manually labelled · split 142/29/29 · seed 42 |
| keyword-rules baseline (test) | accuracy 0.483 · macro-F1 **0.391** |
| majority baseline (test) | accuracy **0.655** · macro-F1 0.079 |
| TF-IDF + LogReg baseline (test) | accuracy 0.621 · macro-F1 0.078 |
| **MiniLM + LogReg head (Phase 3, test)** | accuracy 0.655 · macro-F1 **0.303** · weighted-F1 **0.659** |
| **MiniLM + centroids (Phase 3, test)** | accuracy 0.552 · macro-F1 **0.389** (zero hyperparams) |
| confidence (Phase 3, test) | 0.780 mean on correct vs 0.508 on wrong |
| structural validations | 7/7 PASS |

Recall@K / escalation metrics **do not exist yet** — they are Phase 4–6
deliverables and will be reported only once measured. Full Phase 3 detail:
`evaluation/results/embedding_results.json` and `reports/report.md` §7.

## Repository layout

```
hiver-support-agent/
├── data/                  # raw (gitignored) + processed outputs
├── notebooks/             # 01_data_exploration.ipynb (executed)
├── scripts/               # find_author_ids · prepare_data · run_eda · build_golden_set · train_baselines · train_embedding_classifier · make_synthetic_sample
├── src/
│   ├── config.py          # env-driven settings, no magic constants
│   ├── data/              # load · preprocessing · conversations
│   └── intents/           # taxonomy · baselines (Phase 2) · embeddings · embedding_classifier (Phase 3)
├── evaluation/            # Phase 2–3: golden set, labels, splits, results
├── models/                # Phase 3: trained heads + embedding cache (gitignored)
├── reports/               # report.md · decision_log.md
├── requirements.txt       # only packages actually used (Phases 1–3)
├── .env.example           # configuration template
└── .gitignore             # keeps the 516 MB corpus out of git
```

## Limitations & safety

* Historical evidence ≠ current policy (disclaimer above) — enforced in every
  user-facing surface.
* Provisional topics are transparent keyword heuristics feeding the Phase 2
  taxonomy; they are not a classifier.
* Ultra-deep threads are truncated by the configurable closure pass budget
  (0.017% of tweets; zero pair loss — all support replies are collected in
  pass 1).
* No model metrics are claimed anywhere in Phase 1; nothing is fabricated.

## Reproducibility

Deterministic seed (`RANDOM_SEED=42`), configuration-driven thresholds, run
metadata persisted per execution (`pipeline_run.json`), and a validation gate
that re-runs on every regeneration.
