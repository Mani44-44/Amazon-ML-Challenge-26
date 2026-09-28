"""Phase 5: score candidates, resolve exclusivity, and choose how many to emit.

Three decisions, in order:

1. **Exclusivity.** Every S2/S3 record belongs to at most one S1 entity (verified: all
   7,638,365 matched ids are unique). So when several entities hold the same record, at
   most one can be right. Keeping only the strongest claim cannot lose more than one true
   link per record and removes a whole class of false merges.

2. **Per-entity k.** A single global probability threshold is wrong for a macro-averaged
   metric: each entity is worth the same 1.0 regardless of its cluster size, so the
   decision is per entity. With calibrated p_1 >= p_2 >= ... the expected size of the true
   cluster is sum(p_i), which gives, for each k,

       P_hat = sum_{i<=k} p_i / k          R_hat = sum_{i<=k} p_i / sum_all p_i

   and k is chosen to maximize F_0.5(P_hat, R_hat).

3. **The empty option.** Predicting nothing scores 1.0 on a true singleton and 0.0
   otherwise, so its expected value is prod(1 - p_i) - the probability that no candidate
   is a real match. It competes directly against the best k >= 1, which is the principled
   version of a "singleton gate".

Inference is chunked by S1 hash partition: the test candidate set is ~50M rows and the
feature builder materializes several string columns per row.

Usage:
    python predict.py train    # score the decision layer against ground truth
    python predict.py test     # write output/matching_results.tsv
"""

from __future__ import annotations

import gc
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from common import OUTPUT, PARQUET, WORK, write_submission
from features import FEATURES, add_context, build

CHUNKS = 16
EXCLUSIVE = True   # keep only the strongest claim on each S2/S3 record


def score_split(split: str, model: str = "model_lgb.txt",
                dst: str | None = None, country: str | None = None) -> pl.DataFrame:
    """Score a split's candidates and persist (s1_id, mid, p) to Parquet.

    Scoring is the expensive step (feature building dominates), so the probabilities are
    written to disk: every later variant of the decision layer then costs seconds rather
    than another full pass.
    """
    booster = lgb.Booster(model_file=str(WORK / model))
    cands = pl.read_parquet(WORK / "parquet" / f"{split}_candidates.parquet")

    # Context FIRST, then subset. `n_rivals` counts competing claims on a record, so it
    # must be computed over the complete candidate set. Subsetting first happened to be
    # harmless for a country filter - rivals never cross country - but for any other
    # subset (e.g. an entity-hash fold) it would silently halve the rival counts.
    cands = add_context(cands)

    if country is not None:
        s1c = pl.read_parquet(
            PARQUET / f"{split}_source1_norm.parquet", columns=["id", "country"]
        ).filter(pl.col("country") == country).select(
            pl.col("id").cast(pl.Int64).alias("s1_id")
        )
        cands = cands.join(s1c, on="s1_id", how="semi")

    out = []
    for c in range(CHUNKS):
        part = cands.filter(pl.col("s1_id").hash(seed=5) % CHUNKS == c)
        if not part.height:
            continue
        feats = build(part, split)
        p = booster.predict(feats.select(FEATURES).to_numpy())
        out.append(feats.select("s1_id", "mid").with_columns(
            pl.Series("p", p, dtype=pl.Float32)
        ))
        del part, feats
        gc.collect()
        print(f"    scored chunk {c + 1}/{CHUNKS}", flush=True)

    scored = pl.concat(out)
    path = WORK / "parquet" / (dst or f"{split}_scored.parquet")
    scored.write_parquet(path, compression="zstd")
    print(f"    saved {path} ({scored.height:,} rows)", flush=True)
    return scored


def decide(scored: pl.DataFrame, exclusive: bool = EXCLUSIVE) -> pl.DataFrame:
    """Apply exclusivity, then pick k per entity by expected F_0.5.

    No low-probability pre-filter is applied. `sum_p` (the expected true cluster size)
    and `p_empty` (the probability that nothing matches) are sums and products over
    EVERY candidate, so dropping the ~25 near-zero candidates an entity typically has
    would inflate `p_empty` - their product alone runs 0.5-0.7 - and shrink `sum_p`.
    Both errors push the empty prediction to win, and a false empty forfeits the entire
    entity. The k search is vectorized, so filtering would save nothing anyway.
    """
    df = scored

    if exclusive:
        # strongest claim on each record wins; ties broken deterministically by s1_id
        df = (
            df.sort(["mid", "p", "s1_id"], descending=[False, True, False])
            .group_by("mid", maintain_order=True)
            .head(1)
        )

    df = df.sort(["s1_id", "p"], descending=[False, True]).with_columns(
        pl.col("p").cum_sum().over("s1_id").alias("cum_p"),
        pl.col("p").sum().over("s1_id").alias("sum_p"),
        pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
        # log-space so the product stays stable for long candidate lists
        (1.0 - pl.col("p")).log().sum().over("s1_id").exp().alias("p_empty"),
    )

    precision = pl.col("cum_p") / pl.col("k")
    recall = pl.col("cum_p") / pl.max_horizontal(pl.col("sum_p"), pl.col("cum_p"))
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)

    df = df.with_columns(f05.alias("f05"))
    best = (
        df.sort(["s1_id", "f05"], descending=[False, True])
        .group_by("s1_id", maintain_order=True)
        .head(1)
        .select("s1_id", pl.col("k").alias("k_star"), pl.col("f05").alias("f_best"),
                "p_empty")
    )
    # the empty prediction is just another option, scored on the same scale
    best = best.with_columns(
        pl.when(pl.col("p_empty") > pl.col("f_best"))
        .then(0)
        .otherwise(pl.col("k_star"))
        .alias("k_star")
    )
    return (
        df.join(best.select("s1_id", "k_star"), on="s1_id", how="inner")
        .filter(pl.col("k") <= pl.col("k_star"))
        .select("s1_id", "mid")
    )


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    t0 = time.time()

    cached = WORK / "parquet" / f"{split}_scored.parquet"
    if cached.exists():
        scored = pl.read_parquet(cached)
        print(f"  reusing {cached.name} ({scored.height:,} rows)", flush=True)
    else:
        scored = score_split(split)
    print(f"  scored {scored.height:,} candidates ({time.time() - t0:.0f}s)", flush=True)
    pairs = decide(scored)
    print(f"  emitting {pairs.height:,} pairs over "
          f"{pairs['s1_id'].n_unique():,} entities", flush=True)

    s1_ids = pl.read_parquet(
        PARQUET / f"{split}_source1_norm.parquet", columns=["id"]
    )["id"]

    if split == "train":
        from evaluate import f_beta_half

        gt = pl.read_parquet(PARQUET / "train_gt_long.parquet").select(
            "s1_id",
            (
                pl.col("match_src").cast(pl.Int64) * 1_000_000_000
                + pl.col("match_id").cast(pl.Int64)
            ).alias("mid"),
        )
        for k, v in f_beta_half(pairs, gt, s1_ids.cast(pl.Int64)).items():
            print(f"  {k:<18} {v:.4f}" if isinstance(v, float) else f"  {k:<18} {v:,}")
    else:
        write_submission(pairs, s1_ids, OUTPUT / "matching_results.tsv",
                         "matched_entity_ids")
        print(f"  wrote {OUTPUT / 'matching_results.tsv'}")
    print(f"  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
