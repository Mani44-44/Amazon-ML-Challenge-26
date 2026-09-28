# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Biju99
**Team Members:** Yash Mani Singh, Tanisha Jaiwal, Akshit Das
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

The pipeline has four parts:

- **Multi-channel blocking.** Five key channels are unioned, and each keeps its own top-K per entity.
- **A three-stage LightGBM cascade.**
  - A cheap ranker over roughly 85 candidates per entity keeps the top 30.
  - A pair scorer then compares each candidate with the S1 record.
  - A collective scorer judges each candidate against the entity's *other* confident records.
- **A fine-tuned multilingual cross-encoder** (distilbert-base-multilingual-cased, Apache-2.0) re-scores the pairs that are still uncertain, blended with the bagged cascade score.
- **An exclusivity-aware expected-F0.5 decision rule.**

Four findings from error analysis drove the design:

- **Indic-script names are translated token by token** from a small vocabulary, so a dictionary learned from training pairs undoes the translation almost exactly.
- **Generic names on truncated addresses** are only distinctive as a *name × address* conjunction.
- **Most confident false positives were constructed "decoy" records.** Test has 2–3× more of them per entity than train. A decoy keeps the address (or bumps the house number up by a few) and changes the name meaningfully. Per-token features, the signed house-number change, and corroboration by the entity's other records separate decoys from ordinary typo noise.
- **True copies of a business share details that S1 may lack** (PIN or ZIP, unit number). A decoy's altered detail appears in no other record.

Results:

- Leaderboard: from 0.910 to **0.982**, with the final submission v9.
- Held-out validation: macro F0.5 of 0.988 (US) and 0.986 (India), against blocking ceilings of 0.995 and 0.992.
- No external data or hosted services are used. The only pretrained model is the open, Apache-2.0 cross-encoder, fine-tuned locally.

---

## 2. Methodology

### 2.1 Problem Analysis

Everything below was measured on the training data.

- **Hard constraints.**
  - Matches never cross `country`: 0 of 7.64M links do.
  - Assignment is exclusive: every S2/S3 record belongs to at most one S1 entity.
  - 85% of clusters contain both an S2 and an S3 record.
  - 5.6% of S1 entities are singletons.
- **Distribution shift.**
  - Test is 38% US, 47% India and 15% France. France has no training data.
  - Test has 5.7 S2/S3 records per S1 against 4.7 in train, so there are more non-matching records per entity.
  - We therefore avoid any learned prior on cluster size or candidate density. A learned decision layer that encoded train's cluster-size prior lost 0.019 on the leaderboard and was retired.
- **Noise catalogue.**
  - Name noise: typos and letter scrambles, accents, legal-suffix drift (Pvt/Private, Ltd/Limited), filler words (Services, Trading, Center, Ventures), domain-style names (`danitamedranotowing.com`), DBA names that are completely different, and full or partial Indic-script transliteration (about 22% of India S2/S3 names).
  - Address noise: component reordering, truncation to 2–4 tokens (`J-102, Ahmedabad, GJ`), empty addresses (about 5%), corrupted numbers (`45th`→`45ND`, `684`→`0684`), a literal `null`, state names in native script, and `St`→`SAINT`.
- **Decoys.**
  - 89% of the high-confidence false positives of our first model were S2/S3 records that belong to *no* S1 entity.
  - They copy the S1 address almost verbatim and change the name in a meaningful way:
    - a different first word: `Cindie Smith Investments` vs `Zuluaga Smith Investments`;
    - a different legal form: `ZR Finance LLP` vs `ZR Finance Private Limited`;
    - an added content word: `Martin Brothers Overseas Limited`.
- **Generic names.** In India the median missed entity had a name shared by 46 other S1 entities (for example "krishna consultants" or "hotel management"). The name identifies such an entity only together with the address: the *whole* name plus a single surviving address token ("good enterprises" + "25") is distinctive, where neither part is.

### 2.2 Solution Strategy

**Approach Type:** Blocking + three-stage gradient-boosted cascade (ranker → pair scorer → collective scorer), a fine-tuned multilingual cross-encoder on the uncertain band, and a constrained decision rule (hybrid)

**Core Innovations:**

1. An Indic-script → Latin dictionary learned from the training ground truth.
2. Name × address conjunction keys (single name token, and whole name), with a separate top-K kept per blocking channel.
3. Decoy-aware features, built from per-token alignment, legal-form conflict, and an unsupervised filler-vs-content word score.
4. **Collective evidence (stage C).** Each candidate is judged against the entity's *other* confident records, not only against S1. True copies of a business share details that S1 may lack (PIN or ZIP, unit number); a decoy's altered house number or qualifier word appears in no other record.
5. **Decoy signature features learned from train:** the *signed* house-number change (decoys add a small increment; true-match noise is symmetric) and supervised log-odds of added or missing name words.
6. Calibration discipline. Every downstream stage sees upstream scores produced the same way on train and on test: out-of-fold on train, and the mean of the same fold models on test.

---

## 3. Candidate Generation (Blocking)

### Normalization

- **Indic tokens.** Indic-script tokens are translated with a 1,347-entry dictionary. We learned it from aligned train pairs by position (every such pair has equal token counts); it is 96.6% consistent and covers 96.4% of test Indic tokens.
- **Everything else non-Latin** is romanized with `anyascii`.
- **Names.** Text is lowercased and punctuation folded. Dotted abbreviations are collapsed (`S.A.S.`→`sas`). Legal suffixes are canonicalized, including the French forms (SARL, SAS, SASU, EURL, SCI, SNC, EI and others), and a `core` name without legal words is kept.
- **Addresses.** Digits are split from letters and leading zeros removed. Street types are canonicalized, `null` tokens dropped, and Indic-script state names mapped to state codes.

### Blocking keys

Blocking runs within each country. Each channel keeps its own top-K per S1 entity, and the lists are unioned.

| Channel | Key | K per entity |
|---|---|---|
| Address pairs | pairs from each record's 6 rarest joint-vocabulary address tokens (S2/S3 frequency ≤ 100) | 35 |
| Name pairs | pairs from the 6 rarest joint-vocabulary name tokens (frequency ≤ 100) | 25 |
| Name × address | one of the first 4 name tokens × any address token (frequency ≤ 120) | 45 |
| Whole name × address | the entire core name (or its consonant skeleton) × any address token (frequency ≤ 60) | 20 |
| Exact | exact normalized core name, and exact consonant skeleton (frequency ≤ 30) | all |

Candidates within a channel are ranked by summed IDF of their shared keys.

- **Candidate pairs generated:** about 85 per entity (145M for test) before the stage A cut. `candidate_pairs.tsv` holds the **top 30 per entity after stage A**, 51.9M pairs for test. That is exactly the set the final model scores.
- **How we ensured true matches were not lost:**
  - Recall was measured by segment: empty address, short address, Indic name, domain name, exact name.
  - A channel was added only for each failure mode that was measured to exist.
  - Using a separate top-K per channel stops long-address neighbours from crowding out short-address matches.
  - Train pair recall went from 0.934 (v1) to 0.964 (v2) to 0.971 (v3) to **0.9795** (final, v4). Stage A's cut to the top 30 keeps 0.9794, so it loses almost nothing.

---

## 4. Matching Model

**Stage A (cheap ranker, all ~80 candidates):**

- Per-channel weights and ranks, total weight, margin to the entity's best, candidate count.
- **Rival count:** how many other S1 entities hold the same record. Exclusivity makes this informative.
- Token-set Jaccard for name, address and digits; length ratios.
- Name exact match, suffix agreement, address equality, empty address.

**Stage B (full scorer, top 30):**

- All stage A features, plus:
- **Per-token alignment (decoy features).**
  - For each S1 core token, the best Jaro-Winkler score against the candidate's tokens; reported as the minimum and the mean, in both directions.
  - The count of tokens with no counterpart above 0.8.
  - The similarity of the first token.
- **Filler vs content.** `extra_filler_min` is the log over-representation of each unmatched candidate token in S2/S3 names relative to S1 names. It is computed from each split's own unlabelled text, so it adapts to French without labels.
  - Generator filler words (Services, Trading) score high.
  - Real content words (Overseas, Exports) score near zero.
- **Missing-token rarity:** the IDF of the rarest S1 token that has no counterpart.
- **Legal form:** a conflict flag (both names have a legal form and the forms are disjoint) and the size of the symmetric difference.
- **Addresses:** first-number agreement, and the count of numbers in the candidate address that are absent from the S1 address.
- **Edit distances:** ratio and token-set ratio on the name, ratio on the consonant skeleton, and token-set and partial token-set ratios on the address.
- **Stage A score in context:** the score, its rank within the entity, the gap to the entity's best, the **best competing claim on the same record from another entity**, and whether this entity is the strongest claimant.

The highest-gain features are stage A's score in context, the numbers absent from the S1 address, the filler score, the strongest rival claim, unmatched-token counts, legal-form difference and conflict, and first-token similarity.

**Stage C (collective scorer, same 30 candidates):**

- All stage B features, plus stage B's score.
- A "support" set: the entity's other candidates with stage-B score ≥ 0.9, leave-one-out. At this threshold the support set has the same size on train and test (3.2–3.3 per entity), so it is not contaminated by test's denser decoys.
- Corroboration: numbers and name words the candidate has but S1 lacks, split into those found in some support record and those found in none ("orphans"); the share of all the candidate's numbers seen in S1 or in support; agreement of the first number with the support consensus; best address and name Jaccard with any support record.
  - On train, 69% of true matches with extra address numbers have them corroborated, against 4% of decoys.
- The signed change of the first house number, with small-increase and small-decrease flags. In US train, a first number that rises by 1–30 marks a non-match 86% of the time; one that falls by 1–30, only 4%.
- Log-odds that an added or missing name word marks a non-match (for example "holdings", "overseas", "midtown" vs "dba", "formerly"), learned from labelled train pairs cross-fold, so no row is described by statistics that saw its own label.
- **Vocabulary-free substitution evidence (v7):** an S1 word is missing *and* a real word (held by ≥ 20 S1 names of the same split) was added in its place. That is the decoy move in any language; typos produce rare strings, and filler has no missing partner. This lets the model read French decoys (Fetes → Ecole) without any French labels. Added words the learned log-odds table has never seen are left as missing values and counted, rather than filled with a sentinel.
- **Source balance (v7):** how many of the entity's other confident records come from the candidate's own source. Its distribution is the same on train and test (1.39–1.45 per entity).
- The support-set size is deliberately *not* a model input: it is a cluster-size proxy, and test's decoy density differs from train.

**Final combination:**

- **Bagging.** Three stage C models are averaged: seed 0 with feature fraction 0.8, seed 1 with 0.7, and seed 2 with 0.9. On train out-of-fold scores the bag reaches 0.98537, against 0.98527 for one model.
- **Cross-encoder.** `distilbert-base-multilingual-cased` (Apache-2.0, 134M parameters) is fine-tuned as a pair classifier on the *uncertain* train pairs only (stage C p between 0.02 and 0.98).
  - Its input is the S1 record's name and address, then the candidate's name and address plus up to two of the entity's other confident records, as raw text.
  - Two models are trained, one on each half of the train entities (A and B). Each train row is therefore scored by a model that never saw it.
  - One epoch, batch 32, maximum 128 tokens, learning rate 3e-5, fp16, on a 6 GB laptop GPU.
- **Blend.** On the uncertain band, a logistic regression combines the bagged stage C logit with the cross-encoder logit. It is fitted on train quarter B1 and evaluated on the held-out quarter B2.
  - Band AUC rose from 0.888 to 0.933.
  - Macro F0.5 rose from 0.98662 to 0.98819 (US) and from 0.98328 to 0.98636 (India).
  - On test the cross-encoder input is the mean of the two models' logits (correlation 0.935, same scale).
  - Outside the band, the bagged stage C score is used as is.

**Model type:** LightGBM binary classifiers (stage A: 127 leaves, 300 rounds; stages B and C: 255 leaves, 600 rounds), plus the fine-tuned transformer cross-encoder above.

**Training and calibration:**

- Two folds, split by S1 hash. Train is scored out-of-fold.
- Test is scored by the mean of the same two fold models.
- Stage B therefore learns from stage A scores produced the same way at train and test time. On test, the score distributions match OOF closely.

**Threshold selection method:** there is no global threshold.

1. **Exclusivity:** each S2/S3 record is kept only for its highest-scoring S1 entity.
2. **Choosing k:** for each entity, the number of matches k is the one that maximizes expected F0.5, with candidates sorted by probability. The empty prediction wins when the probability that none of the candidates matches exceeds the best expected F0.5.

We tried a learned decision layer. It gained offline but lost on the leaderboard because it encoded train's cluster-size prior, so it was removed.

---

## 5. Results & Error Analysis

**F_0.5 (macro), out-of-fold on train:**

| | US | India |
|---|---|---|
| v1 (0.910 leaderboard) | ~0.95 | ~0.90 |
| v2 | 0.9801 | 0.9616 |
| v3 | 0.9806 | 0.9675 |
| v4 | 0.9810 | 0.9768 |
| v5 (stage C, corroboration) | 0.9827 | 0.9794 |
| v6 (+ number delta, word log-odds) | 0.9858 | 0.9820 |
| v7 (+ substitution, unknown words, source balance, full-fold training) | 0.9865 | 0.9834 |
| final (v9: bag of 3 stage C models + cross-encoder blend), measured on held-out quarter B2 | **0.9882** | **0.9864** |

**Unseen-country proxy (for France):** both stages were refitted on US only and scored on all of India. The result is F0.5 **0.9234** (P 0.946, R 0.922), against 0.833 for v1 on the same proxy. This proxy is harsher than France, whose Latin script is closer to the US training data.

**Test-side diagnosis.** v4 scored 0.948 on the leaderboard against 0.972 predicted. Test has 2–3× more near-miss decoys per entity than train (mostly neighbour addresses with a small house-number increase), so candidates that look like 90% matches in train are much less likely matches in test. The share of uncertain candidates (p 0.1–0.9) per test entity was 0.81 under v4 (train: 0.33). Under v6 it is 0.27, and v6's self-expected F0.5 on test is 0.978 (v4: 0.959).

Leaderboard: v1 0.910; v4 0.948; v6 0.978; v7 0.980; v8 0.982. v9 (final: v8 with the second cross-encoder averaged in) is the submitted file.

The self-expected F0.5 on test tracked the leaderboard closely: 0.9784 for v6 (0.978), 0.9794 for v7 (0.980), and 0.9820 for v8 (0.982). So it served as the test-side proxy for every decision. v9's self-expected value is 0.9822: US 0.9838, India 0.9849, France 0.9700.

- **Common false positives:**
  - Residual decoys whose substituted token is itself a plausible typo (`Devavi`→`Devavy`).
  - Records with an empty address and a generic name shared by several S1 entities.
- **Common false negatives:**
  - Empty-address records with generic names. They are unresolvable without further signal, and they are about a third of the remaining misses.
  - Heavily truncated addresses combined with a typo'd name.
  - DBA names with a changed address.

---

## 6. Conclusion

Measured error analysis made the gains, not model capacity:

- Each blocking channel and feature targets a failure mode we counted in the data.
- The largest single lessons were:
  - blocking recall, not the classifier, capped the first system at 0.974;
  - the dominant false positives were constructed decoys, which needed features describing the *kind* of difference rather than its size;
  - learned decision layers must not encode distribution-specific priors such as cluster size, because test density differs from train.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/` has one entry point: `python run_all.py`. See `README.md` for the data layout and environment variables.

| Order | Scripts | Role |
|---|---|---|
| 1 | `prepare_data.py` | TSV → Parquet |
| 2 | `indic_dict.py` | Learns `indic_dict.json` from train ground truth |
| 3 | `normalize_all.py` (uses `normalize.py`) | Normalization |
| 4 | `blocking.py` | Candidate generation |
| 5 | `stage_a.py` | Cheap ranker and cut |
| 6 | `stage_b.py` (uses `decoy_features.py`) | Full pair scorer; writes `candidate_pairs.tsv` |
| 7 | `stage_c.py` (run three times: seeds 0, 1, 2) | Collective scorer |
| 8 | `xenc.py` | Cross-encoder: builds text pairs, fine-tunes models A and B, scores the uncertain band (GPU) |
| 9 | `final_blend.py` (uses `predict.decide` for the decision rule) | Bag + cross-encoder blend, decision rule, writes `matching_results.tsv` |

Supporting modules are `common.py`, `evaluate.py` and `features.py`.

### B. Compliance

- **Data.** Only the provided train and test files are used. Blocking vocabularies and the filler and IDF statistics come from each split's own unlabelled text, including test; no labels are involved. The Indic dictionary is learned from the training ground truth only.
- **No external services.** There is no geocoding, no registry lookup, and no internet data.
- **Models.**
  - LightGBM (MIT), trained from scratch.
  - One pretrained open model: `distilbert-base-multilingual-cased` (Apache-2.0, 134M parameters, well under the 8B limit). It is fine-tuned locally on the provided training pairs.
  - No hosted or proprietary model or API is used at any point.
- **Libraries.** polars (MIT), pyarrow (Apache-2.0), numpy (BSD), scipy (BSD), scikit-learn (BSD), lightgbm (MIT), rapidfuzz (MIT), anyascii (ISC), torch (BSD), transformers / tokenizers / safetensors / huggingface-hub (Apache-2.0).
- **Compute.** A single local laptop: 16 threads, 16 GB RAM, NVIDIA RTX 4050 Laptop GPU (6 GB). No cloud compute was used.
