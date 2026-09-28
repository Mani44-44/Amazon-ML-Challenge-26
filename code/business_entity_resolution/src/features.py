"""Phase 4: pair features for the candidate scorer.

Every feature here must be meaningful for a country the model has never seen, because
15% of the test set is France and none of it is in training. Two rules follow:

  * **No `country` column, and no feature derived from it.** The model would meet an
    unseen level at inference.
  * **No `addr_state`.** `US_STATES` / `IN_STATES` contain no French regions, so a state
    feature would be missing for 100% of France - a condition that no training country
    exhibits, so the model could never learn to cope with it. Dropped outright rather
    than fed as a mostly-null column.

Features split into two groups by cost. Set-overlap, equality and blocking-context
features are computed natively in Polars over the whole frame. The three edit-distance
features cross into Python once per pair, so they are kept deliberately few: at ~52M test
candidates, each additional `rapidfuzz` call adds minutes to inference.

The blocking-context features are where the exclusivity constraint enters the model:
`n_rivals` is how many *other* S1 entities also hold this S2/S3 record as a candidate.
Since every S2/S3 record belongs to at most one S1 entity, a record claimed by many
entities is evidence against any single one of them. This feature is only correct if
every rival was actually blocked, which is why blocking runs over all of train rather
than a sample.
"""

from __future__ import annotations

import polars as pl
from rapidfuzz import fuzz

FEATURES = [
    # name
    "name_exact", "suffix_agrees", "name_jac", "name_len_ratio",
    "name_ratio", "name_tsr", "skel_ratio",
    # address
    "addr_jac", "addr_digit_jac", "addr_digit_n", "addr_ratio",
    "addr_empty", "addr_len_ratio",
    # blocking context / exclusivity
    "w", "rank", "margin", "n_cand", "n_rivals", "is_s3",
]


def _side(split: str, side: str) -> pl.DataFrame:
    from common import PARQUET

    cols = ["name_norm", "name_core", "name_nospace", "name_skel", "addr_norm"]
    if side == "s1":
        return (
            pl.read_parquet(PARQUET / f"{split}_source1_norm.parquet")
            .select(pl.col("id").cast(pl.Int64).alias("s1_id"),
                    *[pl.col(c).alias(f"{c}_1") for c in cols])
        )
    return pl.concat(
        [
            pl.read_parquet(PARQUET / f"{split}_source{s}_norm.parquet")
            .select("mid", *[pl.col(c).alias(f"{c}_2") for c in cols])
            for s in (2, 3)
        ]
    )


def _jaccard(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    inter = a.list.set_intersection(b).list.len()
    union = a.list.set_union(b).list.len()
    return (inter / pl.when(union == 0).then(1).otherwise(union)).cast(pl.Float32)


def add_context(cands: pl.DataFrame) -> pl.DataFrame:
    """Blocking-context features. MUST be computed on the complete candidate frame.

    `n_rivals` counts the other S1 entities competing for the same S2/S3 record, and
    `rank` / `margin` / `n_cand` describe a candidate's standing inside its entity's
    list. All four are properties of the whole candidate set, so computing them after
    subsampling entities would shrink them far below their inference-time values and
    teach the model the wrong scale. They are cheap - no string joins - so the full
    frame is always processed here, and only the expensive string features below are
    ever restricted to a sample.
    """
    return cands.with_columns(
        pl.col("w").rank("ordinal", descending=True).over("s1_id").alias("rank"),
        (pl.col("w") - pl.col("w").max().over("s1_id")).alias("margin"),
        pl.len().over("s1_id").alias("n_cand"),
        (pl.len().over("mid") - 1).alias("n_rivals"),
        (pl.col("mid") // 1_000_000_000 == 3).cast(pl.Int8).alias("is_s3"),
    )


def build(cands: pl.DataFrame, split: str) -> pl.DataFrame:
    """Attach the string-similarity features to a context-annotated candidate frame."""
    missing = {"rank", "margin", "n_cand", "n_rivals", "is_s3"} - set(cands.columns)
    if missing:
        raise ValueError(f"call add_context() on the full candidate set first: {missing}")

    df = cands.join(_side(split, "s1"), on="s1_id", how="left").join(
        _side(split, "s23"), on="mid", how="left"
    )

    # ---- vectorized string features ---------------------------------------- #
    ntok1 = pl.col("name_core_1").str.split(" ")
    ntok2 = pl.col("name_core_2").str.split(" ")
    atok1 = pl.col("addr_norm_1").str.split(" ")
    atok2 = pl.col("addr_norm_2").str.split(" ")
    dig1 = pl.col("addr_norm_1").str.extract_all(r"\b\d+\b")
    dig2 = pl.col("addr_norm_2").str.extract_all(r"\b\d+\b")

    df = df.with_columns(
        (pl.col("name_nospace_1") == pl.col("name_nospace_2")).cast(pl.Int8).alias("name_exact"),
        (pl.col("name_norm_1") == pl.col("name_norm_2")).cast(pl.Int8).alias("suffix_agrees"),
        _jaccard(ntok1, ntok2).alias("name_jac"),
        _jaccard(atok1, atok2).alias("addr_jac"),
        _jaccard(dig1, dig2).alias("addr_digit_jac"),
        dig1.list.set_intersection(dig2).list.len().cast(pl.Int16).alias("addr_digit_n"),
        (pl.col("addr_norm_2").str.len_chars() == 0).cast(pl.Int8).alias("addr_empty"),
        _len_ratio("name_core_1", "name_core_2").alias("name_len_ratio"),
        _len_ratio("addr_norm_1", "addr_norm_2").alias("addr_len_ratio"),
    )

    # ---- edit-distance features (the only Python-side loop) ---------------- #
    n1 = df["name_core_1"].fill_null("").to_list()
    n2 = df["name_core_2"].fill_null("").to_list()
    k1 = df["name_skel_1"].fill_null("").to_list()
    k2 = df["name_skel_2"].fill_null("").to_list()
    a1 = df["addr_norm_1"].fill_null("").to_list()
    a2 = df["addr_norm_2"].fill_null("").to_list()

    df = df.with_columns(
        pl.Series("name_ratio",
                  [fuzz.ratio(x, y) for x, y in zip(n1, n2)], dtype=pl.Float32),
        pl.Series("name_tsr",
                  [fuzz.token_set_ratio(x, y) for x, y in zip(n1, n2)], dtype=pl.Float32),
        # the skeleton failed as an exact key but its residual distortion is bounded,
        # so as a continuous similarity it still carries signal
        pl.Series("skel_ratio",
                  [fuzz.ratio(x, y) for x, y in zip(k1, k2)], dtype=pl.Float32),
        pl.Series("addr_ratio",
                  [fuzz.token_set_ratio(x, y) for x, y in zip(a1, a2)], dtype=pl.Float32),
    )
    return df.select("s1_id", "mid", *FEATURES)


def build_chunked(cands: pl.DataFrame, split: str, chunks: int = 8) -> pl.DataFrame:
    """`build` over S1 hash partitions, to bound peak memory.

    The edit-distance features pull whole string columns into Python lists. At ~20M rows
    that is several GB of interned strings held at once, which on a 15 GB machine means
    swapping. Only the compact numeric output of each partition is retained.
    """
    import gc

    out = []
    for c in range(chunks):
        part = cands.filter(pl.col("s1_id").hash(seed=5) % chunks == c)
        if part.height:
            out.append(build(part, split))
        del part
        gc.collect()
    return pl.concat(out) if out else build(cands.head(0), split)


def _len_ratio(c1: str, c2: str) -> pl.Expr:
    a = pl.col(c1).str.len_chars().cast(pl.Float32)
    b = pl.col(c2).str.len_chars().cast(pl.Float32)
    lo, hi = pl.min_horizontal(a, b), pl.max_horizontal(a, b)
    return (lo / pl.when(hi == 0).then(1.0).otherwise(hi)).cast(pl.Float32)
