# Apple Support AI — Technical Report

**Evidence-Grounded Customer Support Agent** · technical assignment for Hiver

> **Status: Phase 1 complete (data foundation).** This report contains only
> metrics that were actually computed by executed code — every number below
> comes from `data/processed/pipeline_run.json` / `eda_summary.json`
> (full-scale run on the real Kaggle `twcs.csv`, 2026-09-10, 211.8 s, 7/7
> validations passed). Model-quality metrics (accuracy, F1, Recall@K, …) are
> Phase 2+ deliverables and are deliberately **not** reported here.

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
All outputs passed a 7-point structural validation gate. Phases 2–9
(classification, retrieval, generation, escalation, agent, UI, final report)
are scoped but intentionally not implemented until Phase 1 is verified —
per the assignment's implementation order.

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

**Status: provisional (Phase 1) → final in Phase 2.** EDA keyword tagging
(word-boundary matching) over the 106,137 customer messages gives the
distribution that the Phase 2 taxonomy must justify itself against:

| provisional topic | matched | share |
|---|---|---|
| Software / Apps | 30,740 | 29.0% |
| Device / Hardware | 15,015 | 14.1% |
| Connectivity | 2,975 | 2.8% |
| Billing / Payments | 2,582 | 2.4% |
| Account / Apple ID | 2,335 | 2.2% |
| Repair / Warranty | 1,974 | 1.9% |
| iCloud / Backup | 1,750 | 1.7% |
| Refunds / Purchases | 1,568 | 1.5% |
| Order / Delivery | 1,044 | 1.0% |
| Security / Privacy | 631 | 0.6% |
| Subscriptions | 470 | 0.4% |

Multi-topic messages: **9.9%** — empirical support for the assignment's
`secondary_intents` / `multi_issue` output design. 53% of messages match no
provisional keyword, confirming the need for an `Other` category and a
labelled golden set rather than keyword rules. The final taxonomy (target
8–12 categories) will be documented with per-class evidence in Phase 2.

## 6. Baselines — *planned, Phase 2*

Majority-class and TF-IDF + Logistic Regression baselines (accuracy, macro
P/R/F1, weighted F1, confusion matrix) are Phase 2 deliverables. Not run yet;
no numbers exist.

## 7. Main Classifier — *planned, Phase 3*

Embedding-based classifier (all-MiniLM-L6-v2 → simple head) returning intent,
confidence, secondary intents, multi-issue flag. Not implemented.

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

## 11. Evaluation — *Phase 1 results + planned metrics*

**Executed in Phase 1:** dataset scan statistics (§3), cleaning report (§4),
reconstruction diagnostics (§4), 7/7 structural validations, EDA distributions
(§5, notebook `01_data_exploration.ipynb` executed end-to-end with 16/16
cross-check assertions against `eda_summary.json`).

**Planned:** intent metrics (Phase 2/3), Recall@K (Phase 4), reply scoring
(Phase 5), escalation precision/recall/F1/false-auto (Phase 6).

## 12. Results

See §1 (headline numbers), §3–§5 (detail), and `data/processed/` artifacts:
`applesupport_tweets.csv` (237,745 rows), `conversations.csv` (80,966),
`conversation_pairs.csv` (106,137), `pipeline_run.json`, `eda_summary.json`,
`charts/` (6 PNGs). Reproduce with:

```bash
python scripts/find_author_ids.py     # verify APPLE_SUPPORT_AUTHOR_ID
python scripts/prepare_data.py        # full-scale: ~212 s, <1 GB RAM
python scripts/run_eda.py             # EDA summary + charts
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

* Provisional topics are keyword heuristics, not a classifier; final taxonomy
  lands in Phase 2.
* Closure truncates ultra-deep threads beyond the pass budget (0.017% of
  tweets; no pair loss).
* The EDA "conversations" count for the whole dataset (794,335) counts
  conversation *starters* — threads whose starter is missing from the corpus
  are not included in that total.
* Reply link targets (shared URLs) are not resolvable in the dataset.
* English-only corpus; no language detection performed.

## 16. Future Improvements

Phases 2–9 per the assignment: golden set + conversation-level splits +
baselines (2); embedding classifier + intent evaluation + failure analysis
(3); FAISS retrieval + Recall@K (4); grounded generation (5); escalation +
false-auto evaluation (6); unified agent (7); Streamlit app (8); final report
(9). Engineering follow-ups: parallelise chunk passes; persist the closure
frontier to resume interrupted runs; add dataset drift checks.

## 17. Conclusion

Phase 1 is complete and verified: the real twcs corpus loads, validates and
filters to 80,966 AppleSupport conversations; threads are reconstructed
faithfully (structural ordering, noise handled); 106,137 evidence pairs are
extracted with leakage-safe conversation identity; EDA is executed, charted
and cross-checked across three consumers. The foundation for every later
phase exists and is reproducible from configuration. Phase 2 may begin on
request.
