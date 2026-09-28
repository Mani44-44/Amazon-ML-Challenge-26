# Business Entity Resolution — reproduction guide

This pipeline regenerates both submission files from the challenge data:

- `output/matching_results.tsv` — the final matches
- `output/candidate_pairs.tsv` — the candidate set the final model scores

Only the provided train and test files are used. The pipeline makes no network calls and uses no external data or hosted models.

## 1. Environment

- Python 3.11.
- All dependencies are permissively licensed (MIT / BSD / Apache-2.0 / ISC).
- One pretrained model is used: `distilbert-base-multilingual-cased` (Apache-2.0, 134M parameters). `xenc.py` downloads it from the Hugging Face hub on first use and fine-tunes it on the training pairs. No other external resource is accessed.

```bash
pip install -r requirements.txt
```

Hardware used:

- 16 logical cores and 16 GB RAM, Windows 11.
- NVIDIA RTX 4050 Laptop GPU (6 GB), used only by the cross-encoder steps.
- Peak memory is about 12 GB.
- Full wall-clock time is about 6 hours (about 1.5 hours of it on the GPU).
- Free disk needed is about 40 GB for `work/`.

## 2. Data layout

By default the code expects this layout, relative to the directory that contains `code/`:

```
<root>/
├── code/business_entity_resolution/src/     (this code)
├── 6ab10eb3b23ba_student_resource/student_resource/dataset/
│   ├── train/train_source{1,2,3}.tsv, train_ground_truth.tsv
│   └── test/test_source{1,2,3}.tsv
├── work/            (created: intermediate Parquet and models)
└── output/          (created: the two submission files)
```

If the dataset lives elsewhere, set these environment variables:

- `ER_DATA_DIR`: the folder that contains `train/` and `test/`.
- `ER_ROOT`: where `work/` and `output/` should be created. Optional.

```bash
export ER_DATA_DIR=/path/to/dataset
```

## 3. Run end to end

```bash
cd code/business_entity_resolution/src
python run_all.py
```

GPU training is not bit-for-bit deterministic, so a rerun reproduces the result closely but not byte-for-byte. Given the saved cross-encoder scores, `final_blend.py` does reproduce the submitted file byte-for-byte.

Always reproduce into a fresh `ER_ROOT`, or an empty `work/`. `stage_b.py features` skips shards that already exist, so rerunning into an old `work/` would silently reuse stale features.

`run_all.py` runs the steps below in order. Each step caches its output under `work/`, and each can be re-run on its own.

| # | Command | What it does |
|---|---|---|
| 1 | `python prepare_data.py` | Converts the TSVs to Parquet, with integer ids. Explodes the ground truth into long form. |
| 2 | `python indic_dict.py` | Learns the Indic-script → Latin token dictionary from aligned train ground-truth pairs. Writes `src/indic_dict.json`. |
| 3 | `python normalize_all.py` | Normalizes names and addresses: transliteration, legal-suffix canonicalization, digit isolation, state folding. |
| 4 | `python blocking.py train` then `python blocking.py test` | Multi-channel candidate generation (§4). Train candidates are needed for fitting. |
| 5 | `python stage_a.py features train` / `features test` / `fit` / `score train` / `score test` | Stage A: a cheap LightGBM ranker over the full union. Keeps the top 30 per S1 entity. This cut set is the final candidate set. |
| 6 | `python stage_b.py features train` / `features test` / `fit` / `score train` / `score test` | Stage B: the full LightGBM pair scorer. `score test` writes `candidate_pairs.tsv` (the cut set) and a provisional `matching_results.tsv`. |
| 7 | `python stage_c.py features train` / `features test` / `fit` / `score train` / `score test` | Stage C: collective evidence (each candidate judged against the entity's other confident records), signed house-number change, learned word log-odds, and vocabulary-free substitution features. `score train` prints the out-of-fold report. |
| 8 | stage C twice more, with `ER_MODEL_TAG=_s1 ER_SEED=1 ER_FF=0.7` and `ER_MODEL_TAG=_s2 ER_SEED=2 ER_FF=0.9` (see `run_all.py`) | Two more stage C models for the bag. |
| 9 | `python xenc.py prep`, then `train` / `predict` with `XE_HALF=A`, then `train` with `XE_HALF=B` and `predict` with `XE_HALF=B XE_TEST_ONLY=1` | Cross-encoder (GPU): two models fine-tuned on the two halves of train's uncertain pairs. |
| 10 | `python final_blend.py` | Bags the three stage C models, blends in the cross-encoder on the uncertain band, and writes the final `matching_results.tsv`. |

Validate the result with the official checker:

```bash
cd <root>/6ab10eb3b23ba_student_resource/student_resource
python utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test
```

## 4. Method in brief

- **Normalization.**
  - Indic-script tokens are translated through a dictionary learned from the training ground truth. The generator transliterates token by token from a vocabulary of about 1.3k words.
  - Tokens the dictionary lacks, and other non-Latin text, are romanized with `anyascii`.
  - Legal suffixes are canonicalized, including the French forms SARL, SAS, EURL and similar.
  - Digit runs are split from letters and stripped of leading zeros.
- **Blocking.** Five channels are built within each country. Each channel keeps its own top-K per S1 entity, and the lists are unioned:
  1. address token-pair keys;
  2. name token-pair keys;
  3. name-token × address-token keys, for generic names on truncated addresses;
  4. whole-name × address-token keys, for names shared by many entities;
  5. exact normalized-name and consonant-skeleton keys.
- **Stage A.** LightGBM on vectorized features: per-channel weights and ranks, set overlaps, and rival counts. Keeps the top 30 per entity. Train pair recall is 0.9795 before the cut and 0.9794 after it.
- **Stage B.** LightGBM on:
  - the stage A features;
  - per-token alignment, both directions;
  - legal-form conflict;
  - a filler-vs-content score for added words, learned without labels from each split's own S1 vs S2/S3 token frequencies;
  - edit distances;
  - stage A's score in context, including the strongest competing claim on the same record.
- **Stage C.** It adds evidence from the entity's *other* confident records (stage B score ≥ 0.9, leave-one-out):
  - whether numbers and name words that the candidate has but S1 lacks also appear in those records (true copies of a business share them; decoys do not);
  - the signed change of the first house number (decoys add a small increment; true-match noise is symmetric);
  - log-odds of added and missing name words, learned from train labels cross-fold.
- **Bagging and cross-encoder.** The three stage C models are averaged. On pairs that are still uncertain, a fine-tuned multilingual cross-encoder reads the S1 record, the candidate and the entity's other confident records as text. A logistic blend of the two scores is fitted on held-out train data.
- **Decision.** Exclusivity is enforced: each S2/S3 record goes to its strongest S1 claimant only. The number of matches per entity is then chosen to maximize expected F0.5 given the scores.
- **Calibration.**
  - Every model is trained as 2 folds split by S1 hash. Train is scored out-of-fold.
  - Test is scored by the mean of the same fold models, so the probabilities the downstream stage sees on test match what it saw in training.

## 5. Files

| File | Role |
|---|---|
| `common.py` | Paths, id packing, and the submission writer |
| `prepare_data.py`, `normalize.py`, `normalize_all.py`, `indic_dict.py` | Data preparation and normalization |
| `blocking.py` | Candidate generation |
| `stage_a.py`, `stage_b.py`, `stage_c.py`, `decoy_features.py` | Scoring models and features |
| `xenc.py`, `final_blend.py` | Cross-encoder and the final blend |
| `predict.py` | The expected-F0.5 decision rule (`decide`) |
| `evaluate.py` | Macro F0.5 and recall utilities |
| `run_all.py` | The end-to-end driver |
