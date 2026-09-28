"""Macro-averaged F_0.5 scoring, matching the challenge definition exactly.

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

computed per Source-1 entity and then averaged over ALL Source-1 entities in the
evaluation set, singletons included. Edge cases, straight from the README:

  * truth empty, prediction empty  -> 1.0  (correctly identified singleton)
  * truth empty, prediction non-empty -> 0.0
  * truth non-empty, prediction empty -> 0.0
  * no overlap at all (tp == 0)    -> 0.0  (P = R = 0, formula undefined)
"""

from __future__ import annotations

import polars as pl


def f_beta_half(
    pred: pl.DataFrame,
    truth: pl.DataFrame,
    s1_ids: pl.Series,
) -> dict[str, float]:
    """Score long-form (s1_id, mid) predictions against long-form truth.

    `s1_ids` is the full evaluation universe; entities missing from `pred` or `truth`
    are treated as empty predictions / true singletons respectively.
    """
    universe = pl.DataFrame({"s1_id": s1_ids})

    pred = pred.unique(subset=["s1_id", "mid"])
    truth = truth.unique(subset=["s1_id", "mid"])

    n_pred = pred.group_by("s1_id").len().rename({"len": "n_pred"})
    n_true = truth.group_by("s1_id").len().rename({"len": "n_true"})
    n_tp = (
        pred.join(truth, on=["s1_id", "mid"], how="inner")
        .group_by("s1_id")
        .len()
        .rename({"len": "tp"})
    )

    df = (
        universe.join(n_pred, on="s1_id", how="left")
        .join(n_true, on="s1_id", how="left")
        .join(n_tp, on="s1_id", how="left")
        .with_columns(
            pl.col("n_pred").fill_null(0),
            pl.col("n_true").fill_null(0),
            pl.col("tp").fill_null(0),
        )
    )

    precision = pl.col("tp") / pl.col("n_pred")
    recall = pl.col("tp") / pl.col("n_true")
    f = (1.25 * precision * recall) / (0.25 * precision + recall)

    df = df.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0))
        .then(1.0)
        .when(pl.col("tp") == 0)  # covers both "empty prediction" and "no overlap"
        .then(0.0)
        .otherwise(f)
        .alias("f05")
    )

    tp_total = int(df["tp"].sum())
    return {
        "f05_macro": float(df["f05"].mean()),
        "micro_precision": tp_total / max(int(df["n_pred"].sum()), 1),
        "micro_recall": tp_total / max(int(df["n_true"].sum()), 1),
        "mean_pred_size": float(df["n_pred"].mean()),
        "mean_true_size": float(df["n_true"].mean()),
        "n_entities": df.height,
    }


def candidate_recall(
    candidates: pl.DataFrame,
    truth: pl.DataFrame,
) -> dict[str, float]:
    """Blocking diagnostics: what fraction of true links survive candidate generation.

    This is a hard ceiling on final recall, so it is worth tracking on its own.
    """
    truth = truth.unique(subset=["s1_id", "mid"])
    hit = truth.join(
        candidates.unique(subset=["s1_id", "mid"]),
        on=["s1_id", "mid"],
        how="semi",
    )
    n_true = truth.height
    return {
        "pair_recall": hit.height / max(n_true, 1),
        "n_true_links": n_true,
        "n_candidate_pairs": candidates.height,
        "cands_per_entity": candidates.height / max(candidates["s1_id"].n_unique(), 1),
    }
