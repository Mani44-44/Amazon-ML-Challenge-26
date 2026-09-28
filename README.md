# Amazon ML Challenge 2026: Business Entity Resolution

**Team Biju99:** Yash Mani Singh, Tanisha Jaiwal, Akshit Das

For each Source 1 business record, the task is to find its matching records in Sources 2 and 3 (US, India, and France, which has no training data). Scoring is macro F0.5 per Source 1 entity.

**Final leaderboard score: 0.982**, up from 0.910 for the first full pipeline.

## Approach

1. **Normalization.** Indic-script names are translated through a token dictionary learned from the training ground truth. Legal suffixes are canonicalized, including French forms, and digit runs are isolated.
2. **Multi-channel blocking.** Address pairs, name pairs, name × address, whole name × address and exact keys. Each channel keeps its own top-K and the lists are unioned. Pair recall on train is 0.980.
3. **Three-stage LightGBM cascade.**
   - A cheap ranker cuts the list to the top 30 candidates.
   - A pair scorer with decoy-aware features compares each candidate with the S1 record.
   - A collective scorer judges each candidate against the entity's *other* confident records.
4. **Cross-encoder.** A fine-tuned `distilbert-base-multilingual-cased` (Apache-2.0) re-scores the uncertain band, blended with the bagged cascade score.
5. **Decision.** Exclusivity is enforced, then the number of matches per entity maximizes expected F0.5.

## Repository layout

| Path | Contents |
|---|---|
| `code/business_entity_resolution/` | The pipeline (`src/`), reproduction guide (`README.md`) and pinned `requirements.txt` |
| `Documentation_template.md` | Full methodology write-up |
| `PLAN.md` | Working log: analyses, measurements and decisions, version by version |
| `output/matching_results.tsv` | Final submitted matches (v9) |

The challenge dataset is not included. See `code/business_entity_resolution/README.md` for the expected data layout and how to run the pipeline end to end.
