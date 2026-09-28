"""Stage B: the full pair scorer over stage A's cut set, then the decision rule.

Features are stage A's cheap features, plus the expensive per-pair features from
`decoy_features` (per-token alignment, legal-form conflict, filler vs content words,
edit distances), plus stage A's probability placed in context:

  p_a, p_a_rank          stage A's score and its rank within the entity
  p_a_gap                distance to the entity's best stage-A score
  p_a_rival              best stage-A score this record gets from any OTHER entity.
                         Assignment is exclusive, so a record that another entity
                         claims more strongly is evidence against this one.
  p_a_owner              1 if this entity is the record's strongest claimant

The same calibration rule as stage A applies: train rows are scored out-of-fold, and
test rows by the mean of the two fold models. The decision is the analytic
expected-F_0.5 rule with exclusivity (`predict.decide`). No learned decision layer, since
that is what learned train's cluster-size prior and failed on test.

Usage:
    python stage_b.py features train|test
    python stage_b.py fit
    python stage_b.py score train|test      # train: OOF + F0.5 report; test: writes output
"""

from __future__ import annotations

import gc
import sys
import time

import lightgbm as lgb
import polars as pl

import decoy_features as dfeat
from common import OUTPUT, PARQUET, WORK, write_submission
from stage_a import A_FEATURES, DIR, SHARDS, _strings, join_strings, labels, shard_of

TRAIN_FRACTION = 2
B_FEATURES = A_FEATURES + [f for f in dfeat.DECOY_FEATURES if f not in A_FEATURES] + [
    "p_a", "p_a_rank", "p_a_gap", "p_a_rival", "p_a_owner",
]
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 255,
    "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": 14, "verbose": -1, "seed": 0,
}
NUM_ROUNDS = 600


def _cut_context(cut: pl.DataFrame) -> pl.DataFrame:
    """Stage-A score in context. Runs on the COMPLETE cut set (rivals)."""
    top2 = cut.group_by("mid").agg(
        pl.col("p_a").max().alias("_p1"),
        pl.col("p_a").top_k(2).min().alias("_p2"),
        pl.len().alias("_n"),
    )
    return (
        cut.join(top2, on="mid", how="left")
        .with_columns(
            pl.col("p_a").rank("ordinal", descending=True).over("s1_id")
            .cast(pl.Int16).alias("p_a_rank"),
            (pl.col("p_a").max().over("s1_id") - pl.col("p_a")).alias("p_a_gap"),
            # best OTHER claim: the runner-up if this entity holds the best, else the best
            pl.when(pl.col("_n") == 1).then(0.0)
            .when(pl.col("p_a") >= pl.col("_p1")).then(pl.col("_p2"))
            .otherwise(pl.col("_p1")).cast(pl.Float32).alias("p_a_rival"),
            (pl.col("p_a") >= pl.col("_p1")).cast(pl.Int8).alias("p_a_owner"),
        )
        .drop("_p1", "_p2", "_n")
    )


def features(split: str) -> None:
    t0 = time.time()
    out = DIR / f"{split}_b_features"
    out.mkdir(parents=True, exist_ok=True)
    cut = _cut_context(pl.read_parquet(DIR / f"{split}_cut.parquet"))
    s1, s23 = _strings(split)
    filler, idf = dfeat.token_stats(split)
    print(f"  cut {cut.height:,} rows, token stats ready ({time.time() - t0:.0f}s)", flush=True)
    with dfeat.make_pool(filler, idf) as pool:
        for c in range(SHARDS):
            dst = out / f"shard_{c:02d}.parquet"
            if dst.exists():
                continue
            part = cut.filter(shard_of() == c)
            a = pl.read_parquet(DIR / f"{split}_a_features" / f"shard_{c:02d}.parquet")
            part = part.join(a, on=["s1_id", "mid"], how="left")
            strs = join_strings(part.select("s1_id", "mid"), s1, s23)
            # join does not guarantee row order, so features are keyed, not pasted
            dec = pl.concat([strs.select("s1_id", "mid"),
                             dfeat.compute(strs, filler, idf, pool).drop("addr_equal")],
                            how="horizontal")
            part = part.join(dec, on=["s1_id", "mid"], how="left")
            part.select("s1_id", "mid", *B_FEATURES).write_parquet(dst, compression="zstd")
            print(f"    shard {c:02d}: {part.height:,} rows ({time.time() - t0:.0f}s)", flush=True)
            del part, a, strs, dec
            gc.collect()


def _load(split: str, shards: list[int], sample: bool = False) -> pl.DataFrame:
    frames = []
    for c in shards:
        df = pl.read_parquet(DIR / f"{split}_b_features" / f"shard_{c:02d}.parquet")
        if sample:
            df = df.filter(pl.col("s1_id") % TRAIN_FRACTION == 0)
        frames.append(df)
    return pl.concat(frames)


def fit(countries: tuple[str, ...] | None = None, tag: str = "") -> None:
    """2-fold OOF models. `countries` restricts training rows (LOCO: US only)."""
    t0 = time.time()
    gt = labels()
    keep = None
    if countries:
        keep = (pl.read_parquet(PARQUET / "train_source1_norm.parquet", columns=["id", "country"])
                .filter(pl.col("country").is_in(list(countries)))
                .select(pl.col("id").cast(pl.Int64).alias("s1_id")))
    for fold in (0, 1):
        tr = _load("train", [c for c in range(SHARDS) if c % 2 != fold], sample=True)
        if keep is not None:
            tr = tr.join(keep, on="s1_id", how="semi")
        tr = tr.join(gt, on=["s1_id", "mid"], how="left").with_columns(pl.col("y").fill_null(0))
        booster = lgb.train(PARAMS, lgb.Dataset(tr.select(B_FEATURES).to_numpy(),
                                                label=tr["y"].to_numpy()), NUM_ROUNDS)
        booster.save_model(str(WORK / f"stage_b{tag}_fold{fold}.txt"))
        print(f"  fold {fold}: fitted on {tr.height:,} rows ({time.time() - t0:.0f}s)", flush=True)
        if fold == 0:
            imp = sorted(zip(B_FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])
            print("  top gain:", ", ".join(f"{n}" for n, _ in imp[:15]))
        del tr
        gc.collect()


def score(split: str, tag: str = "") -> pl.DataFrame:
    t0 = time.time()
    models = [lgb.Booster(model_file=str(WORK / f"stage_b{tag}_fold{f}.txt")) for f in (0, 1)]
    out = []
    for c in range(SHARDS):
        part = _load(split, [c])
        x = part.select(B_FEATURES).to_numpy()
        if split == "train" and not tag:
            p = models[c % 2].predict(x)
        else:
            p = (models[0].predict(x) + models[1].predict(x)) / 2
        out.append(part.select("s1_id", "mid").with_columns(pl.Series("p", p, dtype=pl.Float32)))
    scored = pl.concat(out)
    scored.write_parquet(DIR / f"{split}_b{tag}_scored.parquet", compression="zstd")
    print(f"  scored {scored.height:,} ({time.time() - t0:.0f}s)", flush=True)
    return scored


def report(scored: pl.DataFrame, countries=("US", "India")) -> None:
    from evaluate import f_beta_half
    from predict import decide

    gt = labels().drop("y")
    s1 = (pl.read_parquet(PARQUET / "train_source1_norm.parquet", columns=["id", "country"])
          .select(pl.col("id").cast(pl.Int64).alias("s1_id"), "country"))
    pairs = decide(scored).select("s1_id", "mid")
    for c in countries:
        uni = s1.filter(pl.col("country") == c)
        g = gt.join(uni, on="s1_id", how="semi")
        m = f_beta_half(pairs.join(uni, on="s1_id", how="semi"), g, uni["s1_id"])
        ceil = f_beta_half(scored.select("s1_id", "mid").join(g, on=["s1_id", "mid"], how="semi"),
                           g, uni["s1_id"])
        print(f"  {c:<6} F0.5={m['f05_macro']:.4f}  P={m['micro_precision']:.4f} "
              f"R={m['micro_recall']:.4f}  pred/ent={m['mean_pred_size']:.2f}  "
              f"ceiling={ceil['f05_macro']:.4f}", flush=True)


def write_test(scored: pl.DataFrame) -> None:
    from predict import decide

    pairs = decide(scored).select("s1_id", "mid")
    s1_ids = pl.read_parquet(PARQUET / "test_source1_norm.parquet", columns=["id"])["id"]
    write_submission(pairs, s1_ids, OUTPUT / "matching_results.tsv", "matched_entity_ids")
    cut = pl.read_parquet(DIR / "test_cut.parquet").select("s1_id", "mid")
    write_submission(cut, s1_ids, OUTPUT / "candidate_pairs.tsv", "candidate_entity_ids")
    print(f"  wrote matching_results.tsv ({pairs.height:,} pairs) and candidate_pairs.tsv "
          f"({cut.height:,} rows)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        features(sys.argv[2])
    elif cmd == "fit":
        fit()
    elif cmd == "fit_loco":
        fit(("US",), tag="_us")
    elif cmd == "score":
        s = score(sys.argv[2])
        report(s) if sys.argv[2] == "train" else write_test(s)
    elif cmd == "loco":
        s = score("train", tag="_us")
        report(s, countries=("India",))
