# Amazon ML Challenge 2026 — Business Entity Resolution: Analysis & Plan

**Deadline:** 27 Sep 2026, 23:59 IST (~30 hours from now). 5 submissions/day.
**Hardware:** Ryzen 7 8845HS (8c/16t), 15.3 GB RAM, RTX 4050 6GB **but torch is CPU-only
build** (2.11.0+cpu), 51 GB free disk. → **CPU-only plan. Do not detour into CUDA.**

---

## 1. The task, precisely

Source 1 is a deduplicated reference. For **every** S1 record, emit the list of S2/S3
records referring to the same real business. Scored by **macro-averaged F_0.5** over all
S1 entities (per-entity F, then mean).

### Scale

| File | Rows | Size |
|---|---|---|
| train_source1 | 2,206,821 | 201 MB |
| train_source2 | 5,034,616 | 467 MB |
| train_source3 | 5,285,603 | 481 MB |
| test_source1 | **1,732,544** | 167 MB |
| test_source2 | 4,887,273 | 486 MB |
| test_source3 | 5,082,316 | 483 MB |

---

## 2. Facts established from the data (verified, not assumed)

### 2.1 Three structural constraints — these shape everything

1. **Matches never cross `country`.** Verified over all 7,638,365 ground-truth links:
   7,638,365 same-country, 0 cross-country. → Country is a **hard partition**. Blocking
   runs independently on US / India / France. Free ~3× reduction and a free precision filter.

2. **Assignment is exclusive.** 7,638,365 total matched IDs, **7,638,365 unique** — zero
   duplicates. Every S2/S3 record belongs to **at most one** S1 entity. This is the single
   biggest precision lever in the problem: it turns the decision stage from independent
   per-pair thresholding into a **competitive assignment problem**.

3. **Clusters are usually cross-source.** 1,776,047 clusters (85.4%) contain both an S2 and
   an S3 member; 143,029 S2-only; 164,498 S3-only. Useful as a per-entity prior.

### 2.2 Cluster-size distribution

| #matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| count | 123,247 | 119,157 | 375,212 | 530,841 | 484,115 | 321,957 | 164,868 | 63,968 | 18,680 | 4,205 | 534 | 37 |

- **Singletons are only 5.6%** (123,247 / 2,206,821).
- Mean cluster size among non-singletons ≈ **3.66**.
- ~37% of S2/S3 records are distractors belonging to no S1 entity.

### 2.3 Distribution shift — the biggest validation trap

| | US | India | France |
|---|---|---|---|
| **train** S1 | 1,323,633 (60%) | 883,188 (40%) | **0** |
| **test** S1 | 663,106 (38%) | **809,986 (47%)** | **259,452 (15%)** |

**15% of the test set is France with zero training data**, and the US/India ratio inverts.
A random holdout from train will look excellent and tell you nothing about 15% of the score.

### 2.4 Noise catalogue (from worked ground-truth clusters)

| Pattern | Example |
|---|---|
| Typos / letter scrambling | `Payne Enterprises` → `PAYNE-ENRTPRMISES`, `Payne Etrepndiels` |
| Accents | `Payne Énterprises` |
| **Full script transliteration** | `Ss Food Private Limited` ↔ `एसएस फूड प्राइवेट लिमिटेड`; `Raj Investments LLP` ↔ `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` |
| Partial transliteration | `Raj Investments எல்எல்பி` |
| Domain-style names | `maurewilliamscolombier.com`, `wilfordhancock.com` |
| **Completely different trade name (DBA)** | `Maure Williams Colombier Inc` → `Dréxkor` — *recoverable only via address* |
| Legal-suffix drift | `Inc` / `LLC` / `Pvt` / `Private` / `Ltd` / `Limited` added, dropped, swapped |
| **Empty address** | `S2-681193310` has no address — *recoverable only via name* |
| Address token reorder | `630 45th Terrace, Kansas City, MO` → `KANSAS CITY, MO, 630 45ND TERRACE, null` |
| State abbrev ↔ full | `IL`↔`Illinois`, `MO`↔`Missouri`, `TN`↔`தமிழ்நாடு` |
| Street-type abbrev + bad expansion | `St` → `Street`, and `ST` → **`SAINT`** |
| Number corruption (prefix survives) | `45th`→`45ND`, `Af-684`→`AF-0684` |
| Literal `null` tokens | `..., null, KANSAS CITY, MO` |

**Consequence:** name-only blocking permanently loses the DBA cases; address-only blocking
permanently loses the empty-address cases. Blocking **must** be a multi-channel union.

---

## 3. The metric — work the numbers before designing

F_0.5 = 1.25·P·R / (0.25·P + R), macro-averaged. For a true cluster of size 4:

| prediction | F_0.5 |
|---|---|
| all 4 correct | **1.000** |
| 3 of 4 (one FN) | **0.938** |
| all 4 + 1 FP | **0.833** |
| top-1 only, correct | **0.625** |

**FP costs ≈ 2.7× an FN — but recall still matters enormously.** "Precision-heavy" does
*not* mean "emit one match." With mean cluster size 3.66, a top-1-only strategy **caps out
near 0.62**. Target ~3–4 emissions per entity.

**Singletons are a minor, late lever.** At 5.6%, always predicting non-empty costs ≈0.056
total. A high-precision singleton gate recovering half is worth ≈ +0.03 — real, but do it
last. Predicting ∅ for a non-singleton forfeits the full 1.0, so **the default must be
non-empty**.

---

## 4. Approach

**Normalize → multi-channel block → GBDT pair scorer → exclusivity-aware assignment →
per-entity k selection.** Classical ER. Correct call given CPU-only + 30h + the
8B/MIT-Apache model rule, which makes LLM embeddings a distraction.

### Phase 0 — Setup (~45 min)

```
pip install polars pyarrow rapidfuzz lightgbm unidecode
```
None are currently installed (only numpy, scipy, scikit-learn, networkx, torch-cpu, tqdm).

- Convert all 6 TSVs → **Parquet**, partitioned by `country`.
- **Strip `S1-`/`S2-`/`S3-` prefixes; carry IDs as int32.** With 15 GB RAM and ~12 M test
  records, Python string ID columns will exhaust memory.
- Persist every intermediate to Parquet on disk (51 GB available). Never recompute blocking.

### Phase 1 — Floor submission, TODAY (~1 h) ⚠️ do this before anything clever

Crudest viable pipeline (normalized-name exact/near match, top-1 per entity, empty
otherwise) → run `utils/validate_submission.py` → **upload**. This proves format acceptance
and sets a floor. Discovering a format rejection at 23:00 on the 27th is the worst available
outcome, and the validator checks format only, never score.

### Phase 2 — Normalization (~2 h) — highest value-per-hour in the whole plan

One shared `normalize()` for name and address:

- **Romanize non-Latin scripts with `unidecode`** before anything else. India is 47% of
  test and names appear *entirely* in Devanagari/Tamil. Non-negotiable.
  *(This is a static character-map library, not an entity lookup — permitted, but state it
  explicitly in the methodology doc so a reviewer need not guess.)*
- Strip accents, lowercase, collapse punctuation, drop literal `null`.
- Canonicalize legal suffixes: `pvt`→`private`, `ltd`→`limited`, `corp`→`corporation`,
  `inc`, `llp`, `llc`, `co`; **keep a suffix-stripped variant as a separate field.**
- `&` ↔ `and`. Strip `.com`/`.in` and split domain-style names into tokens.
- Addresses: expand street types both directions (`rd`↔`road`, `st`↔`street`, and handle the
  `st`→`saint` corruption), US state abbrev ↔ full name, Indian state abbrev ↔ full.
- **Split digit runs out of tokens** (`45th`→`45`, `Af-684`→`684`) — number corruption is
  common but the leading digits survive.
- Extract structured keys: PIN/ZIP, numeric street number, city token, state.

Validate normalization by measuring **blocking recall against train ground truth** at each
step, not by eyeballing.

### Phase 2/3 — MEASURED RESULTS (26 Sep, after Phases 0–1)

Channel recall measured against all 7,638,365 train ground-truth links. **This reorders
the blocking design: address is the primary channel, name is secondary.**

| Channel | India | US |
|---|---|---|
| exact `name_nospace` key | 47.2% | 55.5% |
| exact `name_skel` (consonant skeleton) | 48.7% | 57.4% |
| union of the two | 48.8% | 57.4% |
| **shares ≥1 address digit-run** | **83.9%** | **81.5%** |
| **shares ≥2 address tokens** | **95.8%** | **95.2%** |
| either address channel | 95.9% | 95.2% |

**The consonant-skeleton idea underperformed.** It was designed for transliterated Indic
names, but on that exact subset it lifts recall only 17.1% → 19.7%:

| segment | links | nospace | skeleton |
|---|---|---|---|
| India, transliterated | 719,443 | 17.1% | 19.7% |
| India, Latin | 2,340,400 | 56.4% | 57.7% |
| US, accented Latin | 341,338 | 75.9% | 76.7% |
| US, plain | 4,237,184 | 53.9% | 55.8% |

Romanization is *phonetic, not orthographic* (`Ss Food Private Limited` →
`eses phuudd praaivett limittedd`), and the residual distortion is too variable for an
exact key. Keep the skeleton as one cheap channel among several, **but do not rely on it**.

**Transliterated Indic names (~719k links, 9.4% of all links) are the hardest segment.**
They are reachable mainly through the address channel — digits survive transliteration
unchanged — plus fuzzy char-n-gram cosine over romanized text.

Also measured: **12.2% of S2/S3 records have no digits in their address at all** (empty or
digit-free). These are exactly the records the name channel must carry, confirming that
neither channel alone suffices.

### Phase 2/3 — MEASURED RESULTS, second round (26 Sep, advisor review)

**The transliterated-India segment IS reachable through the address channel — confirmed.**
PLAN.md previously asserted this without measuring it on that segment. Measured on GT links:

| segment | links | shares >=1 digit-run | shares >=2 addr tokens | empty S2/S3 address |
|---|---|---|---|---|
| India, transliterated | 719,443 | 79.7% | **98.4%** | 1.3% |
| India, Latin | 2,340,400 | 78.2% | 95.0% | 4.7% |
| US, transliterated/accented | 341,338 | 77.7% | 94.7% | 5.2% |
| US, plain | 4,237,184 | 78.4% | 95.2% | 4.7% |

The address channel is **not weaker** on transliterated names — it is slightly *stronger*
(98.4% vs 95.0%), because those records carry fuller, digit-bearing addresses. This is the
opposite of the name channels (17.1% there). The hardest segment for names is the easiest
for address, so the multi-channel union is confirmed as the right design.

**Legal-suffix agreement carries real precision signal.** Splitting the floor submission's
2.43 M predicted pairs by whether the full `name_norm` matched or only the suffix-stripped
`name_core`:

| pair type | pairs | precision |
|---|---|---|
| `name_norm` identical | 1,190,403 | **0.977** |
| core-only (legal suffixes differ) | 1,241,435 | **0.780** |

So suffix stripping buys recall at a real precision cost. Keep both variants, pass
`suffix_agrees` to the model as a feature, and **never emit on a core-only name match
without address corroboration**.

### Phase 3 — Blocking (~5 h) — determines the recall ceiling AND the wall-clock cost

**Do not run a global char-3-gram cosine.** 810k India S1 × 2.31M India S2 with trigrams
will blow up the sparse product — trigrams are far too common. **Nest every similarity
inside a coarse pre-block** (city token / PIN / numeric street number) so each cosine runs
over thousands of rows, not millions. Use **char 4-grams** with `min_df` to raise IDF.
This one decision is the difference between blocking taking 1 hour and 12.

Channels (union, then cap per entity at ~25–40):

- **(a) Name channel** — char-4-gram TF-IDF cosine top-k within country, nested inside a
  coarse key (first rare name token / phonetic code). Catches typos, transliteration,
  domain names.
- **(b) Address channel** — key on `(numeric street number, city token)` and on `PIN`.
  **This is the only channel that recovers the DBA cases** (`Dréxkor`).
- **(c) Rare-token inverted index** — for records where both above are weak; IDF-weighted,
  skip tokens above a document-frequency cap.

**Acceptance metric is recall@K, not unbounded recall.** The 95% figures in the table
above were measured on *true pairs only* — they say "when two records match, they usually
share 2 address tokens". They are **not** achievable blocking output: `road`, the city name
and `ny` are each shared by millions of records, so "shares >=2 address tokens" as a literal
blocking rule returns an unusable candidate set. The channel must be **IDF-weighted with a
document-frequency cap, ranked, and truncated to the top K per S1**. Report recall at
K = 20 / 30 / 40 alongside actual pairs-per-entity. **Target: >=95% at K ~= 30.** A number
quoted without its K is meaningless.

Also record the reduction ratio. Blocking recall is a hard ceiling on everything downstream. Persist candidates to Parquet *as generated* —
`candidate_pairs.tsv` must be a verifiable superset of final matches (1.73 M × ~30 ≈ 50 M
pairs ≈ 1–2 GB).

### Phase 4 — Pair scorer (~5 h)

**Block ALL 2.2 M train S1 entities, then subsample rows only when fitting LightGBM.**
Do *not* sample S1 entities before blocking. The strongest precision features —
"how many other S1 entities also claim this S2/S3 record" and the Phase 5 competition
softmax — are only correct if every competitor for a record is present. Blocking a
150–300k sample would make those counts several times smaller than at inference (where
all 1.73 M test S1 are blocked): a guaranteed train/test skew on the best feature in the
model. Full train blocking costs ~1.3x the test run, which has to be built anyway.
If time forces a cut, sample **whole partitions** (entire PIN/ZIP prefixes) so that
every competitor inside a partition survives.

Negatives = blocked-but-not-matched pairs (hard negatives by construction).

Features — **all country-agnostic**:
- Name: char-4-gram cosine, token Jaccard, `rapidfuzz` ratio / token_set_ratio /
  partial_ratio, suffix-stripped variants of each, length ratio, initials match.
  Also run `rapidfuzz` over `name_skel` as a *continuous* feature — the skeleton failed as
  an exact key but the residual distortion is bounded, so a fuzzy ratio on it should carry
  more signal than the binary key did.
- Address: token Jaccard, numeric-token exact overlap (street number, PIN), city match,
  state match, cosine, **missing-address indicator**.
- Cross: does name similarity survive suffix removal; is this an S2 or S3 record.
- Contextual: candidate's rank and score *within* this S1 entity's candidate list;
  margin to the entity's best candidate; **how many other S1 entities also list this S2/S3
  record** (a direct exploit of the exclusivity constraint).

**Legal-suffix agreement is a feature, not noise** (measured, see below): exact-name pairs
where `name_norm` agreed in full scored 0.977 precision against 0.780 when only `name_core`
agreed. Emit `suffix_agrees`, plus the raw and suffix-stripped similarities separately.

**Do NOT feed `country` as a categorical** — the model would meet an unseen level (France)
at inference.

**Audit every feature for "is this structurally different in France?"** `addr_state` is the
known offender: `US_STATES`/`IN_STATES` contain no French regions, so a state-match feature
is 100% missing for 15% of the test set — a condition training never exhibits, since no
training country behaves that way. Either drop state from the model, or randomly blank it
on a sizeable share of training pairs so the model learns to work without it. Apply the
same test to PIN/ZIP shape, digit-run counts and token-count priors.

**Validation = leave-one-country-out.** Train on US → validate on India, and the reverse.
That is the only honest estimate of France transfer. A random split is actively misleading
here. If US and India need different thresholds, route France to whichever transfers better
under LOCO.

### Phase 5 — Decision layer (~4 h) — where the remaining points are

1. **Exploit exclusivity.** Each S2/S3 record may go to at most one S1 entity. Normalize
   each candidate's score by the competition for that record (softmax across the S1 entities
   claiming it), then resolve conflicts by greedy / mutual-best assignment. Independent
   per-pair thresholding leaves real precision on the table.
2. **Per-entity k, not a global threshold.** With calibrated probabilities `p_1 ≥ p_2 ≥ …`,
   choose k maximizing expected F_0.5: `P̂ = Σ_{i≤k} p_i / k`, `R̂ = Σ_{i≤k} p_i / Σ_all p_i`,
   comparing against the k=0 option scored as `Π(1 − p_i)`. Per-entity k selection reliably
   beats a single global cut on a macro-averaged metric.
3. **Singleton gate, last.** Only where all candidate probabilities are very low. Worth
   ≈ +0.03 at most; a false ∅ forfeits a full 1.0.
4. Optional: use the 85% "both S2 and S3" prior to nudge k.

### Phase 6 — Package (~2 h)

```
<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md
```
Sections 2–5 of the template map onto Phases 2–5 above; most of this document is reusable
verbatim. Follow the **README's** "no page limit, prioritise technical depth" over the PDF's
"1–2 pages."

---

## 5. Compliance

- **No external data lookup of any kind** — no geocoding APIs, no business registries, no
  internet augmentation. Instant disqualification. `unidecode` is a static character map
  shipped as a library, not a lookup service; declare it anyway.
- Any final model must be **MIT/Apache-2.0 and ≤ 8B parameters**. LightGBM (MIT) is clear.
- **`unidecode` is GPL-licensed.** The licence rule as written targets the *model*, and a
  GBDT pipeline ships no model weights, so this does not breach it — but to remove the
  question entirely, `anyascii` 0.3.3 (**ISC**, permissive) is installed as a drop-in
  romanizer. Swapping it costs one function in `normalize.py` plus a ~9 min re-run of
  `normalize_all.py`; do it during Phase 6 packaging, and re-measure, since the character
  tables differ slightly. Declare whichever ships, and state that it is a static character
  map performing no external lookup.
- Max 5 submissions/day; keep a version history of every upload and its score.

---

## 6. Risk register

| Risk | Mitigation |
|---|---|
| **France (15%) transfers badly** | LOCO validation from the start; strictly country-agnostic features |
| **Blocking explodes in runtime** | Nest cosine inside coarse pre-blocks; 4-grams; cap candidates; checkpoint to Parquet |
| **15 GB RAM exhausted** | int32 IDs, country-partitioned processing, Parquet streaming, never hold two full sources in memory |
| **Format rejection late** | Floor submission uploaded today (Phase 1) |
| **Over-tuning for precision** | Keep the metric table in §3 visible; target ~3–4 emissions/entity, not 1 |
| **Clock** | Phases 0–3 must be done by end of 26 Sep; Phase 4–5 on the 27th; hard stop 20:00 IST for packaging |

---

## 7. Progress log

**Phase 0 — DONE.** `polars`, `pyarrow`, `rapidfuzz`, `lightgbm`, `unidecode` installed.
All 6 TSVs → Parquet with int32 IDs (2.4 GB → 924 MB, ~10 s). Verified: every row has
exactly 4 fields, all IDs well-formed, max id 999,999,995 fits int32. `quote_char=None`
is required — a few hundred rows contain literal `"` in the name. Ground truth exploded
to a long `(s1_id, match_id, match_src)` table. Cross-source IDs are packed into a single
int64 `mid = src*1e9 + id`, because `S2-123` and `S3-123` can both exist.

**Phase 2 (first pass) — DONE.** All 6 sources normalized and cached to
`*_norm.parquet` (~9 min total).

**Phase 1 — DONE.** Floor submission built and validated.

- Exact `name_nospace` key within country, with an exclusivity guard (a key claimed by
  >1 S1 entity emits nothing) and a max-group cap.
- **Train F_0.5 = 0.3918** — precision 0.876, recall 0.279, mean 1.10 predictions/entity
  vs 3.46 true. Confirms the §3 analysis: under-emitting is the dominant error mode.
- `output/matching_results.tsv` + `output/candidate_pairs.tsv` written, 1,732,544 rows
  each. **`validate_submission.py --check-ids` → PASS.** Format risk retired.

### Phase 3/4 — MEASURED RESULTS (26 Sep)

**Blocking (`blocking.py`), full train, 173 s:** 64,427,759 candidates.

| | recall | cands/entity |
|---|---|---|
| recall@10 | 0.8881 | 10.0 |
| recall@20 | 0.9216 | 19.7 |
| **recall@30** | **0.9338** | 29.2 |

Design that got there: two-token composite keys over the k=6 rarest **joint-vocabulary**
tokens (df cap 100), separately for address and name, unioned with exact `name_nospace`
and `name_skel`. Single-token keys were measured first and rejected — 71% recall needed a
df cap of 1000, implying ~450M join rows for India alone. Measured pair-key curve (India,
address): k=4/cap=500 → 0.834 @ 147M rows; **k=6/cap=100 → 0.857 @ 173M**; k=8/cap=500 →
0.903 @ 1.30B rows (9x over budget). k=6/cap=100 dominates k=4/cap=500 on both axes.

**Two bugs found and fixed here, both of which were silently costing recall:**
1. `normalize_address` never isolated digit runs, despite §Phase 2 claiming it did.
   `45th` / `45ND` and `0684` / `684` stayed distinct tokens. After the fix, digit-sharing
   recall rose 79.7% → **85.4%** (India transliterated) and 78.2% → 83.4% (India Latin).
2. The first `blocking.py` accumulated all channels' pairs before truncating — ~260M rows
   for US, 7.4 GB resident with 0.15 GB free, i.e. swapping. Restructured to truncate to
   TOP_K inside each S1 partition: byte-identical output, US now 93 s with 4.9 GB free.

**Pair scorer (`model.py`), LightGBM, 14.6M labelled rows (11.0% positive):**

| train → validate | Average Precision | P @ top-1/entity |
|---|---|---|
| US → India | 0.9601 | 0.9020 |
| India → US | 0.9826 | 0.9357 |

Cross-country transfer is strong, which is the best available evidence that France will
not collapse. Top features by gain: `rank`, **`skel_ratio`**, `addr_len_ratio`,
`addr_digit_jac`, `addr_ratio`, **`n_rivals`**. Two things worth noting: the consonant
skeleton, which failed as an exact blocking key, is the **2nd** strongest feature as a
*continuous* similarity; and `n_rivals` — the exclusivity feature — ranks 6th, confirming
that competition for a record is real signal.

### Phase 5 — MEASURED RESULTS (26 Sep)

**Honest leave-one-country-out end-to-end score.** All 883,188 India entities scored with
a booster fit on US only — the closest available proxy for France, which is 15% of test
and absent from training:

| decision layer | F_0.5 | P | R | pred/entity |
|---|---|---|---|---|
| **exclusivity ON** | **0.8328** | 0.9265 | 0.7557 | 2.83 (true 3.46) |
| exclusivity OFF | 0.8282 | 0.9182 | 0.7569 | 2.86 |

Exclusivity wins on precision (+0.008) at no recall cost, as the constraint predicts, so
it ships ON. vs the Phase 1 floor of 0.3918.

**A bug in the decision layer, caught before it shipped.** `MIN_P` filtered candidates
*before* `sum_p` and `p_empty` were computed. A typical entity carries ~25 candidates at
p = 0.01-0.03; dropping them lifts their product prod(1-p) from ~0.5 to near 1, so the
empty prediction would have won far too often — and a false empty forfeits the whole
entity, on the 94% of entities that are not singletons. The pre-filter is removed
entirely: the k search is vectorized, so it bought nothing.

Also: scored probabilities are now persisted to `{split}_scored.parquet`. Scoring is
~7 min per pass, so every further decision-layer variant now costs seconds instead.

### Submission #2 — READY (27 Sep 01:15)

`output/matching_results.tsv` rewritten from the full pipeline. **Validator PASS with
`--check-ids`.** 1,732,544 rows; 1,630,156 non-empty; 5,247,435 emitted pairs.

**France gate — PASSED.** The failure mode that mattered was one visible only on France
(15% of test, zero training data). It is not present; France tracks US almost exactly and
is *better* than India on every statistic:

| country | entities | empty rate | mean emitted | mean top-1 p |
|---|---|---|---|---|
| France | 259,452 | 0.0534 | 3.19 | 0.9505 |
| India | 809,986 | 0.0674 | 2.87 | 0.9273 |
| US | 663,106 | 0.0512 | 3.16 | 0.9502 |

Overall 3.03 emitted/entity against a true train average of 3.46 — in the 3-4 band that
§3 identified as the target, and far from the top-1-only trap that caps out near 0.62.

Expected score ~0.83 (from the US-only -> India LOCO measurement). Floor was 0.3918.

### Submission #3 — stage-2 decision model (27 Sep)

**Leaderboard feedback on submission #2: 0.910.** The two proxies predict this almost
exactly — 0.85 x 0.9258 (OOF) + 0.15 x 0.8328 (LOCO) = **0.9119 vs 0.910 actual**. The
measurement framework is trustworthy, so changes can be judged offline before spending a
submission.

**Error budget, measured both ways** (`error_budget.py`):

| loss source | France proxy (LOCO) | in-distribution (OOF) |
|---|---|---|
| **k-decision** | 0.0937 | **0.0400** |
| blocking | 0.0488 | 0.0261 |
| model ranking | 0.0256 | 0.0081 |

The decision layer is the largest lever on both. A flat-threshold sweep confirmed this is
*not* merely miscalibration: the best flat threshold scores 0.8288 against the analytic
rule's 0.8328, so the gap to the oracle survives.

**Stage-2 decision model (`decide2.py`).** Rather than deriving the emit set from an
analytic expected-F rule — which assumes calibrated, independent p, both false — a second
LightGBM learns "emit this candidate?" from each candidate's position in its entity's
score distribution: rank, ratio to best, gaps to neighbours, total mass, rank within its
own source, rival count.

| | analytic rule | stage 2 | delta |
|---|---|---|---|
| in-distribution (OOF) | 0.9258 | **0.9342** | +0.0084 |
| France proxy (LOCO) | 0.8328 | **0.8436** | +0.0108 |

It gains on BOTH, which is the acceptance test — the LOCO gate is deliberately harsh, as
stage 2 there meets an unseen country *and* an unseen calibration, a bigger shift than
France will actually present. Predicted leaderboard **~0.921**.

Emitted 3.39/entity (true average 3.46), up from 3.03. Validator PASS with `--check-ids`.
Per-country: France 3.40 emitted / 6.1% empty, India 3.26 / 7.9%, US 3.53 / 5.7%.

**Infrastructure:** pair features for all 64M train candidates are cached to 16 shards
(1.3 GB). Feature building was ~80% of a scoring pass, so OOF scoring fell from ~13 min
to 4, and every further experiment is now cheap.

## 8. Immediate next actions

**Sequencing rule (decided 26 Sep):** get a *thin* end-to-end pipeline scored before
improving any single stage. ~30 h remain and nothing above the 0.39 floor is on the board.
Order: blocking v1 (timeboxed) → LightGBM v1 → simple per-entity threshold → **submit
tonight**. Only then iterate on blocking recall. Perfecting blocking and arriving at the
deadline with no scorer is the worst reachable outcome; a mediocre full pipeline is
recoverable, a missing one is not.

1. **Upload the floor submission** to lock in a scored baseline (1 of 5 for the day).
2. Build the address blocking channel: `(country, rare address digit-run)` and
   `(country, state, city token)`, IDF-weighted to control group sizes.
3. Add the fuzzy name channel on its **own** name-derived pre-block (rare name token
   within country, or a skeleton prefix) — **not** nested inside the address pre-blocks.
   Nesting it under address would make the 12.2% of digit-free / empty-address records
   unreachable, and those are precisely the records the name channel exists to carry.
   Still never global: the pre-block is what keeps the cosine tractable. (Matches §4(a).)
4. Measure union **recall@K for K = 20 / 30 / 40** plus reduction ratio and actual
   pairs-per-entity; target ≥95% at K ≈ 30. Never quote a recall figure without its K.
5. Then Phase 4 (LightGBM, LOCO validation) and Phase 5 (assignment + per-entity *k*).

---

## 9. Architecture review after submission #3 scored 0.891 (27 Sep, ~03:00 IST)

### Findings (measured)
- **No leakage.** Row-order and ID correlation between S1 and matched S2 are both ~0.001. The top teams' 0.99 comes from modelling.
- **Stage 2 caused the drop from 0.910 to 0.891.** It added 660k pairs that the analytic rule did not emit, with median first-stage p = 0.44 (10th percentile 0.19), mostly at ranks 3–6. In effect it learned train's cluster-size prior ("fill up to about 3.5"). That prior fails on test, which has **5.7 S2/S3 records per S1 against 4.7 in train**. Stage 2 is retired and the analytic rule is the baseline again.
- **Blocking caps us.** A perfect scorer on the current candidates reaches only 0.974 OOF.
  - Recall@K by country:

    | K | US | India |
    |---|---|---|
    | 30 | 0.9725 | 0.8914 |
    | 50 | 0.9777 | 0.9040 |
    | 100 | 0.9804 | 0.9150 |

  - Going deeper does not fix India. Its missing links are never generated at all.
  - Missed links by bucket:

    | bucket | US miss rate | India miss rate |
    |---|---|---|
    | empty S2/S3 address | 34% | 48% |
    | short address (≤4 tokens) | 19% | 50% |
    | exact normalized name equal, still missed | 24k links | 41k links |

  - **22% of India S2 names are in Indic script.** unidecode turns them into tokens like "praaivett limittedd", which share no token with "Private Limited", so token and pair keys cannot link them.
- **89% of confident false positives are constructed decoys**, not records owned by another S1. A decoy copies the address nearly verbatim and changes the name in a meaningful way:
  - it substitutes the distinctive token (Cindie→Zuluaga, Hatchell→Gundlach, YT→RT);
  - it changes the legal form (LLP ↔ Pvt Ltd, Private Limited → Limited);
  - or it adds a real word (Overseas, Motors, Exports).

  True-match noise looks different: typos, suffix abbreviation, and filler words (Services, Trading, Center, Ventures). Our aggregate-similarity features cannot tell the two apart.

### Plan
0. Make the 0.910 analytic file the safe baseline. Stop spending submissions on decision-rule variants.
1. **Blocking v2.** The target is ≥0.99 US and ≥0.97 India pair recall at a similar or lower row count. The channels form a union:
   - **Phonetic token skeleton.** Transliterate with anyascii, collapse repeated letters, fold consonant classes (g/k/q, b/p, d/t, v/w, j/z), and drop inner vowels. Build name pair keys and an exact skeleton-sequence key from it. This fixes Indic names.
   - **Short names.** Single-token core-name keys for short names, and an exact-name key with no df cap that is ranked by address agreement instead of being dropped.
   - **Short addresses.** A key from house/unit number plus city, with state-code folding (GJ/KA/DL/MH…, including Indic-script state names).
   - **Typo-tolerant name channel.** Char-3gram TF-IDF top-N within each country, using sparse matmul (sparse_dot_topn, MIT).
   - **2-hop S2↔S3 expansion** from confident links.
2. **Decoy-aware scorer features.**
   - Per-token alignment in both directions: min and IDF-weighted Jaro-Winkler, and the count of tokens below 0.6.
   - Legal-form class, with a conflict flag kept separate from an absent flag.
   - Added and removed token class, as filler vs content log-odds learned from train. This needs a France check.
   - Address exact-equality and house-number delta.
   - Phonetic-skeleton token features.
   - Cluster coherence: the candidate's similarity to the entity's other confident candidates in the other source.
   - Train on all entities, not the 250k sample.
3. **Decision.** Use the analytic expected-F rule plus exclusivity. The decision layer gets no learned density or cluster-size priors.
4. **Optional, on AWS EC2 GPU:** a multilingual cross-encoder (mDeBERTa-v3-base / XLM-R-base, MIT; ≤8B) fine-tuned on (S1, candidate) pairs to re-rank the top-20. Its score feeds LightGBM as a feature. Declare it in the documentation. Build it only if phases 1–2 land early.
5. Package by about 21:00 IST.

### Follow-up measurements (03:30 IST)
- **Safe baseline restored.** `output/matching_results.tsv` is the analytic rule again: 5,247,435 pairs, the 0.910 set, validator PASS. The 0.891 stage-2 file is kept in the job's tmp directory.
- **Indic-script names**
  - They make up 551k India links (18%).
  - Their miss rate is 22%, which is **123k missed links, 35% of all India misses**.
  - Every one of them has the same token count as its S1 name, so the generator transliterates token for token.
- **Learned dictionary from train GT, aligned by position**
  - **1,347 distinct Indic tokens**, 96.6% purity, covering **96.4% of test India Indic tokens**.
  - Priority 1: substitute these tokens before normalization, with the phonetic skeleton as the fallback for tokens the dictionary hasn't seen.
- **Non-Indic India misses:** 230k links (9.2% miss rate). These are mostly short or truncated addresses, so the short-address key is priority 2.
- **France legal forms seen in test S1–S3 last tokens:** sarl 291k, sas 206k, eurl 83k, sa 67k, sasu 61k, sci 53k, ei 15k, snc 11k, plus s.a.s / sàrl variants. Fillers include groupe, fils, cie, holding, (france). These must go into the legal-form and filler tables.

### v2 pipeline results (27 Sep, 06:51 IST)
- **Normalization.** Added the Indic token dictionary (1,347 tokens), swapped in anyascii (ISC licence, replacing GPL unidecode), added French legal forms, and collapsed dotted abbreviations.
- **Blocking v2.** Added a name-token × address-token cross channel (cap 50). Each channel now keeps its own top-K per entity (addr 25, name 15, cross 25, exact ≤30), and the lists are unioned.
  - Train pair recall rose from **0.9338 to 0.9642**. India alone went from 0.891 to 0.941.
  - This is 51 candidates per entity: 112M train rows, 85M test rows.
  - TF-IDF channels were tried and dropped: they added only 0.26 points for 19 more candidates per entity.
- **Stage A.** A cheap LightGBM with 2 folds; test is scored by the mean of the fold models. It cuts to 25 per entity, and recall after the cut is still **0.9642**, so the cut is lossless. Stage A alone scores OOF US 0.9504 / India 0.9287.
- **Stage B.** Adds the decoy features plus stage A's score in context. Out-of-fold results:

  | | F0.5 | P | R | ceiling |
  |---|---|---|---|---|
  | US | **0.9801** | 0.9955 | 0.9511 | 0.9938 |
  | India | **0.9616** | 0.9943 | 0.9154 | 0.9753 |

  - Top gain: p_a, p_a_owner, p_a_gap, num_extra, extra_filler_min, p_a_rival, n_bad_21, legal_diff, legal_conflict, first_sim.
- **Test output.** 5,815,740 pairs, validator PASS with `--check-ids`.

  | | empty | emitted per entity |
  |---|---|---|
  | France | 0.053 | 3.29 |
  | India | 0.054 | 3.32 |
  | US | 0.047 | 3.42 |

### v3 (27 Sep 10:08 IST): wider blocking
- **Blocking changes.** CAP_CROSS 120; per-channel K: addr 35, name 25, cross 45. Stage A cut raised to 30.
- **Train recall 0.9710** at 81 candidates per entity (v2: 0.9642). After the cut@30 it is 0.9708.
- **OOF F0.5:** US **0.9806**, India **0.9675**. Ceilings are 0.9947 and 0.9815.
- **Test output.** 5,826,724 pairs, validator PASS with `--check-ids`.

  | | empty | emitted per entity |
  |---|---|---|
  | France | 0.050 | 3.32 |
  | India | 0.053 | 3.33 |
  | US | 0.047 | 3.43 |

- Backups are in `work/v2_backup` and `work/v3_backup`.
- **Step 5 (cross-encoder) skipped** by decision: no AWS credentials, CPU-only torch, a 6 GB laptop GPU, and 50M test pairs, for at most about 0.012 on the leaderboard.

### v4 (27 Sep 13:33 IST): whole-name × address channel. FINAL CANDIDATE
- **New blocking channel.** Whole name (core name, or its consonant skeleton) × address token, with cap 60 and K=20.
  - It recovers generic-name links on truncated addresses, such as "good enterprises" + "25".
  - India recall went from 0.9530 to 0.9733 in the isolated test.
- **Train recall 0.9795** (v3: 0.9710); after the cut@30 it is 0.9794.
- **OOF F0.5:** US **0.9810**, India **0.9768**. Ceilings are 0.9950 and 0.9919.
- **Test output.** 5,903,697 pairs, validator PASS with `--check-ids`.

  | | empty | emitted per entity |
  |---|---|---|
  | France | 0.046 | 3.38 |
  | India | 0.048 | 3.39 |
  | US | 0.048 | 3.44 |

- **Packaged** in `team_submission.zip`. The v3 zip is kept at `work/v3_backup/team_submission_v3.zip`.
- **Predicted leaderboard:** 0.38 × 0.981 + 0.47 × 0.977 + 0.15 × (France ≈ 0.94) ≈ **0.973**.
- **France proxy (v4, `loco_v4.py`).** Both stages were fitted US-only and scored on India: F0.5 **0.9234** (P 0.9463, R 0.9219). v1 scored 0.8328 on the same proxy. Predicted leaderboard is 0.970–0.974.

## 10. After the 0.948 leaderboard score (27 Sep, afternoon)
- **Diagnosis.** Test has 2–3× more near-miss decoys per entity than train in every country, mostly neighbour addresses (house number +1..30) with an optional qualifier word.
  - Uncertain candidates (p 0.1–0.9) per entity: test 0.81 against train 0.33. Confident matches per entity are the same.
  - v4's self-expected F on test was 0.959. It scored 0.948 on the leaderboard.
- **Rejected ideas.**
  - Global label-shift EM: its assumptions are violated.
  - Decoy sibling clusters: only 3% of decoys have a sibling record sharing their deviation.
  - Stage D, a second collective pass using stage C's scores: slightly worse, at 0.9852 / 0.9813.
- **v5 (stage C: collective corroboration).** On train, 69% of true matches with extra address numbers have them corroborated by the entity's other records, against 4% of decoys. The support set is p_b ≥ 0.9, leave-one-out; n_support is excluded as a density proxy.
  - OOF: 0.9827 / 0.9794.
  - Test: 109k corroborated pairs moved from 0.56 to 0.96, and 962k orphan-number pairs from 0.41 to 0.27.
- **v6 (+ signed house-number delta, + supervised word log-odds, cross-fold).**
  - In US train, a +1..30 delta is a non-match 86% of the time; −30..−1 only 4%.
  - OOF **US 0.9858 / India 0.9820**.
  - On test, uncertain candidates per entity fell from 0.81 to **0.27**. Self-expected F rose from 0.959 to **0.978**.
  - 5,792,005 pairs, validator PASS with `--check-ids`. This is the version in `output/` and in `team_submission.zip`.
- **Backups.** `work/v4_backup`, `work/v5_backup`, `work/v6_backup`, and the older zips.
- **v6 leaderboard 0.978**, which matches its self-expected F (0.9784). Self-expected F is now the test-side proxy.
- **v7 (17:38).** Added vocabulary-free substitution features, null-filled unknown words plus unknown-word counts, source balance, and full-opposite-fold training (easy negatives subsampled 1 in 4 with weight 4).
  - OOF US 0.9865 / India 0.9834.
  - Test self-expected 0.9794 (France 0.9692).
  - French substitution pairs moved from 0.50 to 0.31; coined-name pairs are unchanged.
  - Validator PASS with the ID check; 5,782,676 pairs. v7 is the version in `output/` and the zip.
- **Checks with no gain.** Decision-rule logit offsets, the v6+D and v6+v7 ensembles, and stage D.
- **Loss that remains.** On test about 0.021: roughly 0.006 blocking, mostly empty-address records whose name several companies share; roughly 0.013 uncertain decisions, about 40% of it the same ambiguity. Above 0.985 is not reachable with the resolvable remainder.
- **v7 leaderboard 0.980**, which matches its self-expected F (0.9794).
- **v8 (22:00).** Bagged stage C (v7 + seed 1 at ff 0.7 + seed 2 at ff 0.9), then a cross-encoder blend on the uncertain band (stage-C p 0.02–0.98).
  - The cross-encoder is distilbert-base-multilingual-cased (Apache-2.0), fine-tuned locally on the RTX 4050 on half A of train's uncertain pairs.
  - The blend is logistic, fitted on B1 and gated on B2.
  - On B2 the band AUC rose from 0.888 to 0.933; F0.5 gained +0.0016 in the US and +0.0031 in India.
  - Test self-expected **0.9820**: US 0.9836, India 0.9846, France 0.9695.
  - Validator PASS with the ID check. It is now `output/matching_results.tsv`; backup in `work/v8_backup`.
