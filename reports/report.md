# Apple Support AI — Technical Report

**Evidence-Grounded Customer Support Agent** · technical assignment for Hiver

> **Status: Phases 1–2 complete (data foundation + intent-classification
> foundation).** This report contains only metrics that were actually computed
> by executed code — every number below comes from `pipeline_run.json` /
> `eda_summary.json` (full-scale run on the real Kaggle `twcs.csv`, 2026-09-10,
> 211.8 s, 7/7 validations passed) and `evaluation/results/baseline_results.json`
> (Phase 2 baselines, seed 42, 2026-09-11). Later-phase metrics (embedding
> accuracy, Recall@K, escalation rates) do not exist yet and are not reported.

---

## 1. Executive Summary

Phase 1 delivers the data foundation for an evidence-grounded AppleSupport
agent: the Customer Support on Twitter corpus (2,811,774 tweets) is loaded and
validated, AppleSupport conversations are isolated via a *configurable*
account id (discovered from data, never hard-coded), conversations are
reconstructed from reply links, and 106,137 customer↔support evidence pairs
are extracted with conversation identity preserved for leakage-free splitting.
Exploratory analysis (notebook + dashboard) profiles the corpus and yields a
provisional topic distribution that will drive the Phase 2 intent taxonomy.
All outputs passed a 7-point structural validation gate. Phases 4–9
(retrieval, generation, escalation, agent, UI, final report) are scoped but
intentionally not implemented until each prior phase is verified — per the
assignment's implementation order.

**Phase 2 adds the classification foundation:** a 10-class intent taxonomy
derived from the EDA distribution (plus three classes grounded in a manual
reading of the unmatched pool), a 200-example **manually labelled golden set**
(uniform sample of customer-initiated openers, seed 42, conversation-level
stratified 70/15/15 split, isolation manifest for Phase 4), and three measured
baselines. The honest headline: with 200 labelled examples, **no learned
TF-IDF configuration beats the majority baseline on test accuracy**
(0.621 vs 0.655) while transparent keyword rules win macro-F1 (0.391 vs
0.079) — the measured motivation for the Phase 3 embedding classifier.

**Phase 3 delivers that classifier and confirms the diagnosis:** a frozen
`all-MiniLM-L6-v2` sentence encoder (384-d, trained on an external public
corpus — no golden-set leakage) with two heads evaluated under the *same*
protocol on the *same* test split. The learned LogReg head lifts test
macro-F1 **0.078 → 0.303** (≈3.9×) at accuracy 0.655 (ties majority) and
the best weighted-F1 of all five models (0.659); a zero-hyperparameter
centroid classifier reaches macro-F1 0.389 — matching the keyword rules'
0.391 at +17 points accuracy. Confidence is informative (0.780 mean on
correct vs 0.508 on wrong test predictions), which is the signal the Phase 6
escalation policy will threshold.

**Headline numbers (real run):** 80,966 AppleSupport conversations · 237,745
corpus tweets · 106,137 support pairs · avg 2.94 turns/conversation · avg
100.8 chars/message · provisional top topics: Software/Apps 29.0%,
Device/Hardware 14.1%, Connectivity 2.8% · multi-topic messages 9.9%.

## 2. Problem Definition

Build a support agent that, given a customer message, can
(1) classify the issue, (2) retrieve historically similar AppleSupport cases,
(3) generate a response *grounded* in those cases, (4) decide auto-handle vs
escalate, and (5) quantitatively evaluate every component. The assignment is
phased; Phase 1 covers the data layer: loading, AppleSupport filtering,
conversation reconstruction and EDA. Later phases depend on this foundation:
the pairs built here are the Phase 4 retrieval corpus; the `conversation_id`
column is the Phase 2 leakage-prevention key.

## 3. Dataset

**Source:** Kaggle `thoughtvector/customer-support-on-twitter` (`twcs.csv`,
516,508,641 bytes, not committed — see `data/README.md`).

**Schema (7 columns, validated on load):** `tweet_id`, `author_id`,
`inbound`, `created_at`, `text`, `response_tweet_id`,
`in_response_to_tweet_id`. All identifiers parsed as strings (18–19-digit ids
lose precision as numbers). `inbound=TRUE` marks customer→brand tweets;
outbound `author_id`s are account screen names.

**Full-corpus scan (single chunked pass):**

| measure | value |
|---|---|
| total tweets | 2,811,774 |
| unique tweet ids | 2,811,774 (0 duplicates) |
| inbound / outbound | 1,537,843 / 1,273,931 |
| conversation starters (tweets with no parent) | 794,335 |
| malformed inbound flags | 0 |
| top outbound authors | AmazonHelp 169,840 · **AppleSupport 106,860** · Uber_Support 56,270 · SpotifyCares 43,265 · Delta 42,253 |

**AppleSupport identification:** `APPLE_SUPPORT_AUTHOR_ID` is configuration;
`scripts/find_author_ids.py` derives candidates from the data itself (the
table above is its output). The configured value `AppleSupport` was verified
against this ranking — nothing was guessed.

**Historical-data disclaimer:** the corpus contains *historical* support
interactions. AppleSupport replies are evidence for retrieval/evaluation,
not a representation of current Apple policy. Observed timestamps in the
filtered corpus span 2013–2017 (min is an outlier; the bulk of threads are
late 2017).

## 4. Data Processing

**Collection (memory-safe):** full-scale runs use an AppleSupport-first
chunked closure (decision D8): pass 1 records all 106,860 AppleSupport tweets;
subsequent passes expand to parents/children until (near-)fixpoint. Canonical
run: 12 passes, 238,754 tweets collected, residual non-convergence +41 tweets
(0.017%, ultra-deep threads only; pairs unaffected). Peak memory stays in the
hundreds of MB on a 4 GB machine.

**Cleaning (conservative, decision D6):** over 238,754 collected rows —
HTML entities decoded 10,426 · URLs removed 99,269 · leading @mentions
stripped 187,230 · whitespace collapsed 19,765 · RT prefixes removed 2 ·
rows dropped 787 (empty text after cleaning; 0 missing ids; 0 duplicate ids).
Typos, casing and punctuation preserved by design.

**Conversation reconstruction:** reply trees built from parent links
(authoritative; 0 parent-link mismatches vs 14,005 dangling `response_tweet_id`
references, which are reported as noise). Missing-parent tweets become
*partial roots* (913) so subset runs never merge unrelated threads; 0 cyclic
tweets encountered in this run (guard exists and is tested by the synthetic
smoke sample). Result: 81,156 conversations, of which **80,966 involve
AppleSupport** (237,745 tweets).

**Evidence pairs:** every AppleSupport reply paired with the inbound customer
tweet it answers, keeping `conversation_id`, both tweet ids and timestamps →
**106,137 pairs**.

**Validation gate (executed, 7/7 PASS):** pair endpoints exist in the tweet
frame; pairs map to reconstructed conversations; pair texts non-empty; each
support reply used at most once; all responses authored by the configured
account; conversation ids unique; every tweet attached to a conversation.

## 5. Intent Taxonomy

**Status: FINAL (Phase 2).** The taxonomy is defined in
`src/intents/taxonomy.py` — the single source of truth consumed by the
golden-set builder, the baseline trainer, the dashboard and (unchanged) the
Phase 3 classifier. It is justified against the Phase 1 EDA distribution
(`eda_summary.json` provisional topics, word-boundary matching) and a manual
reading of randomly sampled openers:

| final class | EDA anchor | golden support | typically escalated |
|---|---|---|---|
| `software_bug` | Software / Apps 29.0% | 128 | no |
| `device_hardware` | Device / Hardware 14.1% | 11 | no |
| `account_icloud` | Account/Apple ID 2.2% + iCloud 1.7% (merged) | 9 | no |
| `connectivity` | Connectivity 2.8% | 9 | no |
| `billing_purchases` | Billing 2.4% + Refunds 1.5% + Orders 1.0% + Subscriptions 0.4% (merged) | 5 | **yes** |
| `repair_warranty` | Repair / Warranty 1.9% | 6 | **yes** |
| `security_privacy` | Security / Privacy 0.6% | 2 | **yes** |
| `general_complaint` | from the unmatched pool (reading) | 9 | no |
| `support_process` | from the unmatched pool (reading) | 6 | **yes** |
| `other` | unmatched remainder | 15 | **yes** |

Design rules (decision D13): every EDA topic ≥~2% share earns a class;
sub-2.5% topics sharing a routing profile are merged so the merged class can
reach usable golden support; the 53% unmatched pool is split by observed
behaviour — pure venting (`general_complaint`), meta-complaints about
reaching support (`support_process` — the natural escalation trigger), and
the remainder (`other`). Multi-topic messages remain 9.9% of the corpus —
empirical support for the assignment's `secondary_intents` / `multi_issue`
output design, deferred to the Phase 3 classifier head.

**Corpus period note:** 99.5% of the 80,244 customer-initiated openers fall
in Sep–Dec 2017 (the iOS 11 rollout window of the corpus). The golden set —
like the corpus — is dominated by iOS 11 era complaints; all Phase 2 metrics
measure this historical window, not a balanced topic mix.

## 6. Golden Set & Baselines — *measured (Phase 2)*

**Golden set** (`evaluation/golden_set.csv`): 200 examples drawn uniformly
(no rare-class boosting — the golden set must estimate performance on the
*natural* distribution; rare-class scarcity is a documented limitation, not a
sampling bug) from the 80,244 customer-initiated conversation openers, seed
42. Every message was read and labelled **manually** by meaning; the
labelling worksheet intentionally shows no keyword hints to avoid biasing
labels. An automated gate validates: every row labelled exactly once, labels
restricted to taxonomy classes, no unknown ids (build exits non-zero on
failure). Split: conversation-level, stratified by label, 70/15/15 →
**142 / 29 / 29**. A manifest (`golden_conversation_ids.json`) reserves all
200 conversation ids so Phase 4 retrieval never indexes golden evidence.

**Baselines (test split, 29 examples, seed 42):**

| baseline | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| keyword rules (Phase 1 EDA topics → labels) | 0.4828 | **0.3906** | 0.5298 |
| majority (`software_bug`) | **0.6552** | 0.0792 | 0.5187 |
| TF-IDF + LogReg | 0.6207 | 0.0783 | 0.5127 |

TF-IDF + LogReg details: word 1–2 grams, English stopwords, sublinear tf;
hyperparameters (C=10.0, class_weight=balanced, min_df=1) selected by 5-fold
CV **on the train split** with criterion mean(accuracy, macro-F1) — tuning
on the 29-example validation split proved too noisy (a val-selected config
scored 0.412 val macro-F1 but 0.104 test macro-F1; recorded in
`baseline_results.json`). Train accuracy 1.0 (the 142-example train split is
memorisable); CV-on-train macro-F1 0.16 ± 0.08.

**Failure analysis of the learned baseline (measured, not asserted):** the
confusion matrix shows 25 of 29 test messages predicted `software_bug` (18
correct). Every rare-class test example (device_hardware 2, account_icloud 1,
connectivity 1, billing 1, repair 1, general_complaint 1, support_process 1,
other 2) was misclassified — mostly into `software_bug`. Keyword rules, by
contrast, perfectly classified the rare classes present in test
(account_icloud, connectivity, billing: P=R=F1=1.0 each) but recall only 47%
of `software_bug`. **Conclusion:** at n=200, learning generalises badly on
short noisy tweets; semantics-aware embeddings (Phase 3) or a hybrid
keywords+learned router are the measured next steps. The val split is
retained as a held-out sanity check and test was untouched until final
evaluation.

Reproduce:

```bash
python scripts/build_golden_set.py sample --n 200   # deterministic worksheet
# (label evaluation/labeling_worksheet.jsonl → evaluation/golden_labels.json)
python scripts/build_golden_set.py build            # validate + split + manifest
python scripts/train_baselines.py                  # metrics → evaluation/results/
```

## 7. Main Classifier — *measured (Phase 3)*

Embedding-based intent classifier: frozen `all-MiniLM-L6-v2` sentence
encoder (384-d, 22.3M parameters, CPU float32, ~17 s to embed the entire
golden set) + two heads, both fit on the train split only, evaluated with
the **identical protocol as Phase 2** (hyperparameters by 5-fold CV on
train with criterion mean(accuracy, macro-F1); val = sanity check; test
untouched until final evaluation) so the comparison isolates the feature
representation, not the tuning protocol.

**Test-split results (29 examples, seed 42, same rows as Phase 2):**

| model | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| keyword rules | 0.4828 | 0.3906 | 0.5298 |
| majority | **0.6552** | 0.0792 | 0.5187 |
| TF-IDF + LogReg | 0.6207 | 0.0783 | 0.5127 |
| **MiniLM + LogReg head** (C=100, balanced) | **0.6552** | 0.3032 | **0.6586** |
| **MiniLM + centroids** (zero hyperparams) | 0.5517 | **0.3894** | 0.6029 |

Read honestly: embeddings **fix the class collapse, not the accuracy
ceiling**. The LogReg head predicts 7 distinct classes (vs 2 for TF-IDF:
25 of 29 predictions were `software_bug`) and lifts macro-F1 ≈3.9× while
tying majority accuracy; the centroid head has the best macro-F1 of all
learned models without a single hyperparameter. Train accuracy is 1.0
(142 examples remain memorisable); CV-on-train macro-F1 of the selected
configuration is 0.4288, selected over an 8-config grid. Rare-class over-sampling was *measured*, not
assumed: `class_weight="balanced"` (the linear-head equivalent of
duplicating rare examples) improved CV macro-F1 0.3495 → 0.4288 and was
selected by the criterion.

**Confidence surface (Phase 6 input, measured on test):** mean confidence
0.7801 on correct vs 0.5079 on wrong predictions; at the configured
threshold 0.60, 65.5% of test predictions would be auto-handled with
84.2% accuracy among them. Caveat: n=29, indicative only. The 10 test
errors (with per-example confidence and runner-up labels) are recorded in
`embedding_results.json` — including a sarcasm case misclassified *with
0.889 confidence*, an honest warning against confidence-only escalation.

Reproduce:

```bash
pip install -r requirements.txt              # adds sentence-transformers
python scripts/train_embedding_classifier.py # metrics + model artifacts
```

Artifacts: `evaluation/results/embedding_results.json` (full),
`phase3_summary.json` (dashboard), `models/embedding_logreg_head.joblib` +
`models/embedding_prototype.joblib` (reusable heads with confidence),
`models/embedding_cache/` (deterministic golden-set embeddings).

## 8. Retrieval System — *planned, Phase 4*

FAISS index over embedded support pairs (customer message + historical reply),
top-K retrieval with Recall@1/3/5 evaluation. The Phase 1 outputs
(`conversation_pairs.csv`) are exactly its corpus.

## 9. Response Generation — *planned, Phase 5*

LLM generation grounded in retrieved pairs, with the safety rules of
assignment §16 and the "based on similar historical support cases" framing.

## 10. Escalation Policy — *planned, Phase 6*

Rule-based, explainable policy consuming intent confidence, retrieval score,
security signals; false-auto rate as the primary safety metric.

## 11. Evaluation — *Phases 1–3 results + planned metrics*

**Executed in Phase 1:** dataset scan statistics (§3), cleaning report (§4),
reconstruction diagnostics (§4), 7/7 structural validations, EDA distributions
(§5, notebook `01_data_exploration.ipynb` executed end-to-end with 16/16
cross-check assertions against `eda_summary.json`).

**Executed in Phase 2:** golden-set validation gate (all 200 rows labelled
exactly once; labels restricted to taxonomy; splits stratified), baseline
metrics on val/test (accuracy, macro/weighted P/R/F1, per-class reports,
confusion matrix — full numbers in `evaluation/results/baseline_results.json`,
dashboard-ready summary in `phase2_summary.json`), hyperparameter-selection
grid (16 configs, CV-on-train), and the golden-set isolation manifest.

**Executed in Phase 3:** embedding classifier evaluation under the Phase 2
protocol — 8-config CV-on-train hyperparameter selection for the LogReg
head, both heads on val/test (accuracy, macro/weighted F1, per-class
reports, confusion matrices), measured rare-class over-sampling comparison,
confidence preview at the configured 0.60 threshold, and 10-example test
error analysis with confidences (full numbers in
`evaluation/results/embedding_results.json`, dashboard-ready summary in
`phase3_summary.json`).

**Planned:** Recall@K (Phase 4), reply scoring (Phase 5), escalation
precision/recall/F1/false-auto (Phase 6).

## 12. Results

See §1 (headline numbers), §3–§6 (detail), and the artifacts:
`data/processed/` — `applesupport_tweets.csv` (237,745 rows),
`conversations.csv` (80,966), `conversation_pairs.csv` (106,137),
`pipeline_run.json`, `eda_summary.json`, `charts/` (6 PNGs);
`evaluation/` — `golden_set.csv` (200 rows), `golden_labels.json`,
`labeling_worksheet.jsonl`, `split_assignments.csv`,
`golden_conversation_ids.json`, `sampling_frame.json`,
`results/baseline_results.json`, `results/phase2_summary.json`,
`results/embedding_results.json`, `results/phase3_summary.json`;
`models/` — `embedding_logreg_head.joblib` (+ `.json` metadata),
`embedding_prototype.joblib`, `embedding_cache/`.
Reproduce with:

```bash
python scripts/find_author_ids.py     # verify APPLE_SUPPORT_AUTHOR_ID
python scripts/prepare_data.py        # full-scale: ~212 s, <1 GB RAM
python scripts/run_eda.py             # EDA summary + charts
python scripts/build_golden_set.py sample --n 200 && \
  python scripts/build_golden_set.py build    # golden set + splits
python scripts/train_baselines.py     # baseline metrics
pip install -r requirements.txt && \
  python scripts/train_embedding_classifier.py   # Phase 3 classifier
```

## 13. Failure Analysis

Data-quality findings from the executed run (all handled + reported):

* **Dangling child links (14,005):** `response_tweet_id` values referencing
  tweets outside the filtered corpus. Handled: parent links are authoritative;
  dangling links are counted, never followed.
* **Partial roots (913):** threads whose starter tweet was not collected
  (deep-thread closure budget or dataset gaps). Handled: they root their own
  partially-observed conversation; boundaries stay stable.
* **Non-converging closure tail (+41 tweets at pass 12):** ultra-deep threads
  grow ~1 hop per pass. Pairs unaffected (all support replies collected in
  pass 1). Documented; pass budget is configurable.
* **Timestamp noise:** replies occasionally carry timestamps inconsistent with
  their structural position (twcs is known to be imperfect). Handled: thread
  order is structural, timestamps are display metadata only. The corpus
  date-range min (2013) is an outlier — bulk of threads are late 2017.
* **URL-heavy replies (99,269 URLs removed):** AppleSupport replies often link
  to support articles; the link targets are lost (dataset contains no
  resolved URLs). Consequence: some replies read as fragments ("Please take a
  look at this:"). Kept — they are still valid evidence of *how* support
  responded.
* **Keyword-matching pitfall (caught in QA):** substring matching made `app`
  match inside `apple`, inflating Software/Apps to 64.5%; fixed with
  word-boundary matching (29.0%). Recorded as decision D11.

## 14. Safety Considerations

* Historical evidence framing is enforced in all user-facing surfaces
  (dashboard, notebook, README, this report).
* No credentials, passwords or personal data are stored; the pipeline never
  requests anything from users (Phase 1 is offline data processing).
* Raw data stays local (gitignored); outputs contain only public tweets.
* The Phase 1 design already reserves the escalation inputs (confidence,
  retrieval score) so that unsafe auto-answers are a Phase 6-measurable
  failure mode (false-auto rate), not an accident.

## 15. Limitations

* **Golden set is small (200) and severely class-imbalanced** — `software_bug`
  holds 128/200 examples; rare classes have 1–2 test examples, so per-class
  test numbers are directional only. This is a property of the natural
  distribution being estimated, not a sampling bug; growing the golden set
  is the standing lever (Phase 3 measured the training-only reweighting
  alternative: balanced class weights helped CV macro-F1, but test
  per-class supports of 1–2 keep any per-class claim directional).
* **Corpus period concentration:** 99.5% of openers are Sep–Dec 2017 (iOS 11
  rollout) — metrics measure that window (§5 period note).
* Provisional Phase 1 topics remain keyword heuristics (now subsumed by the
  taxonomy and reused as the keyword baseline).
* Closure truncates ultra-deep threads beyond the pass budget (0.017% of
  tweets; no pair loss).
* The EDA "conversations" count for the whole dataset (794,335) counts
  conversation *starters* — threads whose starter is missing from the corpus
  are not included in that total.
* Reply link targets (shared URLs) are not resolvable in the dataset.
* English-only corpus; a few non-English messages are present and were
  labelled `other` where not confidently classifiable (documented in the
  labelling method).
* Single manual labeler (the author); inter-rater agreement is not measurable
  — mitigated by the documented taxonomy definitions, examples and the
  automated validation gate.

## 16. Future Improvements

Phases 4–9 per the assignment: FAISS retrieval + Recall@K (4); grounded
generation (5); escalation + false-auto evaluation (6); unified agent (7);
Streamlit app (8); final report (9). Engineering follow-ups: grow the
golden set with a second labelling pass (rare classes have 1–2 test
examples); fine-tune or domain-adapt the encoder on the unmatched pool;
hybrid keywords+embeddings router to combine the keyword baseline's
rare-class precision with the embedding head's context sensitivity;
parallelise chunk passes; persist the closure frontier to resume
interrupted runs; add dataset drift checks.

## 17. Conclusion

Phase 1 is complete and verified: the real twcs corpus loads, validates and
filters to 80,966 AppleSupport conversations; threads are reconstructed
faithfully (structural ordering, noise handled); 106,137 evidence pairs are
extracted with leakage-safe conversation identity; EDA is executed, charted
and cross-checked across three consumers. Phase 2 is complete and verified on
top of it: a 10-class EDA-grounded taxonomy, a validated manually-labelled
golden set with leakage-free conversation-level splits, and three measured
baselines whose honest failure analysis (majority wins accuracy, keywords win
macro-F1, the learned baseline collapses) precisely motivated the Phase 3
embedding classifier. Phase 3 is complete and verified on top of *that*: the
frozen-encoder classifier fixed the measured class collapse (macro-F1
0.078 → 0.303 learned / 0.389 centroid, ≈3.9×) while exposing an informative
confidence surface for the Phase 6 escalation policy — with every artifact,
including the reusable heads and per-example error analysis, reproducible
from configuration. The Phase 4 retrieval corpus already excludes golden
conversations via the isolation manifest.
