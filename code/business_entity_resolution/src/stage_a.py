"""Stage A: a cheap ranker over the full blocking union, cut to the top N per entity.

Blocking v2 keeps each channel's own top-K and unions them, which is what recovers the
short-address and generic-name links, but it leaves ~60 candidates per entity. The
expensive features (edit distances, per-token alignment) cost roughly a microsecond per
token pair in Python, so they cannot run over 130M rows. Stage A scores every candidate
with vectorized features only (channel weights, set overlaps, blocking context), and
keeps the top N_KEEP per entity. That cut set is the final candidate set:
`candidate_pairs.tsv` is written from it, and stage B scores exactly it.

Calibration discipline: stage B uses stage A's probability as a feature. Train rows
get out-of-fold probabilities from 2 fold models. Test rows get the MEAN of those same
two models, not a third model refit on everything. A refit model would be calibrated
differently from the OOF scores stage B learned from, and that kind of mismatch is what
cost submission #3 0.019.

Usage:
    python stage_a.py features train|test   # shard cheap features to disk
    python stage_a.py fit                    # 2-fold OOF on train, save fold models
    python stage_a.py score train|test       # score, cut to top N_KEEP, save
"""

from __future__ import annotations

import gc
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from common import PARQUET, WORK

SHARDS = 16
N_KEEP = int(os.environ.get("N_KEEP", 30))
TRAIN_FRACTION = 4
DIR = WORK / "parquet"

A_FEATURES = [
    "w", "w_addr", "w_name", "w_cross", "w_fullx", "w_exact",
    "rank", "rank_addr", "rank_name", "rank_cross", "rank_fullx", "margin", "n_cand", "n_rivals",
    "is_s3", "name_exact", "suffix_agrees", "name_jac", "name_len_ratio",
    "addr_jac", "addr_digit_jac", "addr_digit_n", "addr_empty", "addr_len_ratio",
    "addr_equal", "name_len1", "addr_len1", "addr_len2",
]

PARAMS = {
    "objective": "binary", "learning_rate": 0.08, "num_leaves": 127,
    "min_data_in_leaf": 200, "feature_fraction": 0.9, "bagging_fraction": 0.8,
    "bagging_freq": 1, "num_threads": 14, "verbose": -1, "seed": 0,
}
NUM_ROUNDS = 300


def shard_of(col: str = "s1_id") -> pl.Expr:
    return pl.col(col).hash(seed=5) % SHARDS


def labels() -> pl.DataFrame:
    return pl.read_parquet(PARQUET / "train_gt_long.parquet").select(
        pl.col("s1_id").cast(pl.Int64),
        (pl.col("match_src").cast(pl.Int64) * 1_000_000_000
         + pl.col("match_id").cast(pl.Int64)).alias("mid"),
        pl.lit(1, dtype=pl.Int8).alias("y"),
    )


def context(c: pl.DataFrame) -> pl.DataFrame:
    """Candidate-list context. Must run on the COMPLETE union (rival counts)."""
    return c.with_columns(
        pl.col("s1_id").cast(pl.Int64),
        pl.col("w").rank("ordinal", descending=True).over("s1_id").cast(pl.Int16).alias("rank"),
        *[pl.when(pl.col(f"w_{l}") > 0)
          .then(pl.col(f"w_{l}").rank("ordinal", descending=True).over("s1_id"))
          .otherwise(99).cast(pl.Int16).alias(f"rank_{l}")
          for l in ("addr", "name", "cross", "fullx")],
        (pl.col("w") - pl.col("w").max().over("s1_id")).cast(pl.Float32).alias("margin"),
        pl.len().over("s1_id").cast(pl.Int16).alias("n_cand"),
        (pl.len().over("mid") - 1).cast(pl.Int16).alias("n_rivals"),
        (pl.col("mid") // 1_000_000_000 == 3).cast(pl.Int8).alias("is_s3"),
    )


def _strings(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    cols = ["name_norm", "name_core", "name_nospace", "name_skel", "addr_norm"]
    s1 = pl.read_parquet(PARQUET / f"{split}_source1_norm.parquet").select(
        pl.col("id").cast(pl.Int64).alias("s1_id"), *[pl.col(c).alias(f"{c}_1") for c in cols])
    s23 = pl.concat([
        pl.read_parquet(PARQUET / f"{split}_source{s}_norm.parquet")
        .select("mid", *[pl.col(c).alias(f"{c}_2") for c in cols]) for s in (2, 3)])
    return s1, s23


def join_strings(df: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    return df.join(s1, on="s1_id", how="left").join(s23, on="mid", how="left")


def cheap(df: pl.DataFrame) -> pl.DataFrame:
    """Vectorized string features over a frame that carries the _1/_2 string columns."""
    def jac(a, b):
        inter = a.list.set_intersection(b).list.len()
        union = a.list.set_union(b).list.len()
        return (inter / pl.when(union == 0).then(1).otherwise(union)).cast(pl.Float32)

    def lr(c1, c2):
        a = pl.col(c1).str.len_chars().cast(pl.Float32)
        b = pl.col(c2).str.len_chars().cast(pl.Float32)
        lo, hi = pl.min_horizontal(a, b), pl.max_horizontal(a, b)
        return (lo / pl.when(hi == 0).then(1.0).otherwise(hi)).cast(pl.Float32)

    dig1 = pl.col("addr_norm_1").str.extract_all(r"\b\d+\b")
    dig2 = pl.col("addr_norm_2").str.extract_all(r"\b\d+\b")
    return df.with_columns(
        (pl.col("name_nospace_1") == pl.col("name_nospace_2")).cast(pl.Int8).alias("name_exact"),
        (pl.col("name_norm_1") == pl.col("name_norm_2")).cast(pl.Int8).alias("suffix_agrees"),
        jac(pl.col("name_core_1").str.split(" "), pl.col("name_core_2").str.split(" ")).alias("name_jac"),
        jac(pl.col("addr_norm_1").str.split(" "), pl.col("addr_norm_2").str.split(" ")).alias("addr_jac"),
        jac(dig1, dig2).alias("addr_digit_jac"),
        dig1.list.set_intersection(dig2).list.len().cast(pl.Int16).alias("addr_digit_n"),
        (pl.col("addr_norm_2").fill_null("").str.len_chars() == 0).cast(pl.Int8).alias("addr_empty"),
        lr("name_core_1", "name_core_2").alias("name_len_ratio"),
        lr("addr_norm_1", "addr_norm_2").alias("addr_len_ratio"),
        ((pl.col("addr_norm_1") == pl.col("addr_norm_2"))
         & (pl.col("addr_norm_1").str.len_chars() > 0)).cast(pl.Int8).alias("addr_equal"),
        pl.col("name_core_1").str.split(" ").list.len().cast(pl.Int16).alias("name_len1"),
        pl.col("addr_norm_1").str.split(" ").list.len().cast(pl.Int16).alias("addr_len1"),
        pl.col("addr_norm_2").fill_null("").str.split(" ").list.len().cast(pl.Int16).alias("addr_len2"),
    )


def features(split: str) -> None:
    t0 = time.time()
    out = DIR / f"{split}_a_features"
    out.mkdir(parents=True, exist_ok=True)
    # Everything in `context` is per entity, and so local to an S1 shard, except the
    # rival count, which needs every entity's claim on a record. That count is taken
    # once over the full frame (one small column), and each shard is then processed
    # alone, so the 112M-row frame is never held together with the string tables.
    src = DIR / f"{split}_candidates.parquet"
    rivals = (pl.scan_parquet(src).group_by("mid").agg((pl.len() - 1).cast(pl.Int16)
              .alias("n_rivals_full")).collect())
    print(f"  rival counts over {rivals.height:,} records ({time.time() - t0:.0f}s)", flush=True)
    s1, s23 = _strings(split)
    for c in range(SHARDS):
        part = (pl.scan_parquet(src).with_columns(pl.col("s1_id").cast(pl.Int64))
                .filter(shard_of() == c).collect())
        part = (context(part).drop("n_rivals")
                .join(rivals, on="mid", how="left")
                .rename({"n_rivals_full": "n_rivals"}))
        # list set-operations allocate per row; 1M-row slices keep them bounded
        f = pl.concat([
            cheap(join_strings(part.slice(i, 1_000_000), s1, s23))
            .select("s1_id", "mid", *A_FEATURES)
            for i in range(0, part.height, 1_000_000)
        ])
        f.write_parquet(out / f"shard_{c:02d}.parquet", compression="zstd")
        del part, f
        gc.collect()
        print(f"    shard {c:02d} ({time.time() - t0:.0f}s)", flush=True)


def _load(split: str, shards: list[int], sample: bool = False) -> pl.DataFrame:
    frames = []
    for c in shards:
        df = pl.read_parquet(DIR / f"{split}_a_features" / f"shard_{c:02d}.parquet")
        if sample:
            df = df.filter(pl.col("s1_id") % TRAIN_FRACTION == 0)
        frames.append(df)
    return pl.concat(frames)


def fit() -> None:
    t0 = time.time()
    gt = labels()
    for fold in (0, 1):
        tr = (_load("train", [c for c in range(SHARDS) if c % 2 != fold], sample=True)
              .join(gt, on=["s1_id", "mid"], how="left").with_columns(pl.col("y").fill_null(0)))
        booster = lgb.train(PARAMS, lgb.Dataset(tr.select(A_FEATURES).to_numpy(),
                                                label=tr["y"].to_numpy()), NUM_ROUNDS)
        booster.save_model(str(WORK / f"stage_a_fold{fold}.txt"))
        print(f"  fold {fold}: fitted on {tr.height:,} rows ({time.time() - t0:.0f}s)", flush=True)
        del tr
        gc.collect()


def score(split: str) -> None:
    t0 = time.time()
    models = [lgb.Booster(model_file=str(WORK / f"stage_a_fold{f}.txt")) for f in (0, 1)]
    out = []
    for c in range(SHARDS):
        part = _load(split, [c])
        x = part.select(A_FEATURES).to_numpy()
        if split == "train":           # out-of-fold: the model that did NOT see shard c
            p = models[c % 2].predict(x)
        else:                           # test: mean of the same two models
            p = (models[0].predict(x) + models[1].predict(x)) / 2
        part = part.select("s1_id", "mid").with_columns(pl.Series("p_a", p, dtype=pl.Float32))
        out.append(
            part.sort(["s1_id", "p_a"], descending=[False, True])
            .group_by("s1_id", maintain_order=True).head(N_KEEP)
        )
        print(f"    shard {c:02d} ({time.time() - t0:.0f}s)", flush=True)
    cut = pl.concat(out)
    cut.write_parquet(DIR / f"{split}_cut.parquet", compression="zstd")
    print(f"  wrote {split}_cut.parquet: {cut.height:,} rows "
          f"({cut.height / cut['s1_id'].n_unique():.1f}/entity)", flush=True)
    if split == "train":
        gt = labels()
        full = pl.concat([_load("train", [c]).select("s1_id", "mid") for c in range(SHARDS)])
        r_all = full.join(gt, on=["s1_id", "mid"], how="semi").height / gt.height
        r_cut = cut.join(gt, on=["s1_id", "mid"], how="semi").height / gt.height
        print(f"  pair recall: union {r_all:.4f}  after cut@{N_KEEP} {r_cut:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        features(sys.argv[2])
    elif cmd == "fit":
        fit()
    elif cmd == "score":
        score(sys.argv[2])
