"""Stage C: collective evidence. Each candidate is judged against its entity's OTHER records.

Stage B compares a candidate with the S1 record alone. But an entity's true S2/S3 records
are all noisy copies of the same full business record, and that record often carries
information S1 lacks: a PIN or ZIP, a unit number, a fuller street. A decoy is a single
record derived from a different business. Its altered house number or substituted name
token appears in none of the entity's other records.

Measured on train: when a candidate's address has numbers absent from the S1 address,
69% of true matches have those numbers corroborated by another candidate of the same
entity, against 4% of decoys. A corroborated pair is 98% true, yet stage B scored such
pairs at 0.91. On test, where decoys are about 2.5x denser, 55% of the uncertain pairs
are corroborated and stage B scores them at only 0.59.

"Support" is the entity's other candidates with stage-B score >= SUPPORT_P (leave one
out: a candidate never supports itself). Features, all relative to that support set:

  n_support            size of the support set (excluding this candidate)
  num_dev              numbers in the candidate's address absent from the S1 address
  num_dev_corrob       ... of those, how many appear in some support record
  num_dev_orphan       ... how many appear in NO support record (the decoy signature)
  num_all_corrob_frac  share of all the candidate's numbers seen in S1 or in support
  first_num_support    the candidate's first number equals the most common first number
                       among support records (-1 when undefined)
  tok_dev              candidate name tokens absent from the S1 name
  tok_dev_corrob       ... that appear in some support record's name
  tok_dev_orphan       ... that appear in none
  addr_jac_support     best address-token Jaccard with any support record
  name_jac_support     best name-token Jaccard with any support record

Calibration discipline as before: train uses out-of-fold stage-B scores to define
support, and test uses the mean-of-folds stage-B scores. Stage C is trained 2-fold and
test is scored by the mean of the fold models.

Usage:
    python stage_c.py features train|test
    python stage_c.py fit
    python stage_c.py score train|test
"""

from __future__ import annotations

import gc
import os
import sys
import time

import lightgbm as lgb
import polars as pl

from common import OUTPUT, PARQUET, WORK, write_submission
from stage_a import DIR, SHARDS, labels, shard_of
from stage_b import B_FEATURES

# The stage whose scores define support, and the tag this stage writes under. The
# defaults run stage C on stage B's scores. A second pass (IN=c, OUT=d) repeats it using
# stage C's own, cleaner scores to choose the support records.
IN = os.environ.get("ER_STAGE_IN", "b")
OUT = os.environ.get("ER_STAGE_OUT", "c")
# Variant switches for bagging: a model tag (so variants never overwrite each other),
# the random seed and feature fraction, and optional pseudo-labelled test rows.
MTAG = os.environ.get("ER_MODEL_TAG", "")
SEED = int(os.environ.get("ER_SEED", 0))
FEATURE_FRACTION = float(os.environ.get("ER_FF", 0.8))
PSEUDO = os.environ.get("ER_PSEUDO", "")   # path to test scores used as pseudo-labels
SUPPORT_P = 0.9   # at 0.9 the support set is the same size on train and test (3.2-3.3); at 0.5 test is inflated by decoys
TRAIN_FRACTION = 1   # each fold model fits on ALL entities of the opposite fold
EASY_P = 0.001
REAL_WORD_DF = 20    # a token held by >= this many S1 names of the split is a real word
COLLECTIVE = [
    "p_b", "n_support", "num_dev", "num_dev_corrob", "num_dev_orphan",
    "num_all_corrob_frac", "first_num_support", "tok_dev", "tok_dev_corrob",
    "tok_dev_orphan", "addr_jac_support", "name_jac_support", "sup_same_src",
]
C_EXTRA = [
    "p_b", "n_support", "num_dev", "num_dev_corrob", "num_dev_orphan",
    "num_all_corrob_frac", "first_num_support", "tok_dev", "tok_dev_corrob",
    "tok_dev_orphan", "addr_jac_support", "name_jac_support",
    # house-number change and supervised word log-odds (see `pair_extras`)
    "num_delta", "num_delta_up", "num_delta_down",
    "extra_lo_max", "extra_lo_sum", "missing_lo_max", "missing_lo_sum",
    # v7: vocabulary-free substitution evidence, unknown-word counts, source balance
    "n_extra_unknown", "n_missing_unknown", "extra_real_n", "extra_df_max",
    "missing_df_max", "subst_real", "sup_same_src",
]
# n_support is left out of the model: it counts the other strong candidates, which is
# a cluster-size proxy, and test has 2-3x the decoy density of train, so it shifts.
C_FEATURES = B_FEATURES + [c for c in C_EXTRA if c != "n_support"]
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 255,
    "min_data_in_leaf": 100, "feature_fraction": FEATURE_FRACTION, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": 14, "verbose": -1, "seed": SEED,
}
NUM_ROUNDS = 600


def _texts(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    s1 = pl.read_parquet(PARQUET / f"{split}_source1_norm.parquet",
                         columns=["id", "name_core", "addr_norm"]).select(
        pl.col("id").cast(pl.Int64).alias("s1_id"),
        pl.col("name_core").fill_null("").alias("n1"), pl.col("addr_norm").fill_null("").alias("a1"))
    s23 = pl.concat([
        pl.read_parquet(PARQUET / f"{split}_source{k}_norm.parquet",
                        columns=["mid", "name_core", "addr_norm"]) for k in (2, 3)
    ]).select("mid", pl.col("name_core").fill_null("").alias("n2"),
              pl.col("addr_norm").fill_null("").alias("a2"))
    return s1, s23


def collective(sc: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    """Leave-one-out support features for every (s1_id, mid) in `sc` (needs column p_b)."""
    x = sc.select("s1_id", "mid", "p_b").join(s1, on="s1_id").join(s23, on="mid")
    x = x.with_columns(
        pl.col("a1").str.extract_all(r"\b\d+\b").list.unique().alias("num1"),
        pl.col("a2").str.extract_all(r"\b\d+\b").alias("num2_seq"),
        pl.col("n1").str.split(" ").list.unique().alias("tok1"),
        pl.col("n2").str.split(" ").list.unique().alias("tok2"),
        pl.col("a2").str.split(" ").list.unique().alias("atok2"),
    ).with_columns(pl.col("num2_seq").list.unique().alias("num2"),
                   pl.col("num2_seq").list.first().alias("first2"))

    sup = x.filter(pl.col("p_b") >= SUPPORT_P).select(
        "s1_id", pl.col("mid").alias("smid"), (pl.col("mid") // 1_000_000_000).alias("ssrc"), pl.col("num2").alias("snum"),
        pl.col("tok2").alias("stok"), pl.col("atok2").alias("satok"), pl.col("first2").alias("sfirst"))

    # pairs (candidate, other support record of the same entity)
    pr = (x.select("s1_id", "mid", "num2", "tok2", "atok2", (pl.col("mid") // 1_000_000_000).alias("src"))
          .join(sup, on="s1_id", how="inner").filter(pl.col("mid") != pl.col("smid")))

    def jac(a, b):
        i = a.list.set_intersection(b).list.len()
        u = a.list.set_union(b).list.len()
        return (i / pl.when(u == 0).then(1).otherwise(u)).cast(pl.Float32)

    agg = pr.group_by("s1_id", "mid").agg(
        pl.len().alias("n_support"),
        (pl.col("ssrc") == pl.col("src")).sum().alias("sup_same_src"),
        pl.col("snum").flatten().unique().alias("sup_nums"),
        pl.col("stok").flatten().unique().alias("sup_toks"),
        jac(pl.col("atok2"), pl.col("satok")).max().alias("addr_jac_support"),
        jac(pl.col("tok2"), pl.col("stok")).max().alias("name_jac_support"),
        pl.col("sfirst").drop_nulls().mode().first().alias("sup_first"),
    )
    x = x.join(agg, on=["s1_id", "mid"], how="left").with_columns(
        pl.col("n_support").fill_null(0),
        pl.col("sup_same_src").fill_null(0),
        pl.col("sup_nums").fill_null(pl.lit([], dtype=pl.List(pl.String))),
        pl.col("sup_toks").fill_null(pl.lit([], dtype=pl.List(pl.String))),
    )
    dev_num = pl.col("num2").list.set_difference(pl.col("num1"))
    dev_tok = pl.col("tok2").list.set_difference(pl.col("tok1"))
    x = x.with_columns(
        dev_num.list.len().alias("num_dev"),
        dev_num.list.set_intersection(pl.col("sup_nums")).list.len().alias("num_dev_corrob"),
        dev_tok.list.len().alias("tok_dev"),
        dev_tok.list.set_intersection(pl.col("sup_toks")).list.len().alias("tok_dev_corrob"),
        (pl.col("num2").list.set_intersection(pl.col("num1").list.concat(pl.col("sup_nums"))).list.len()
         / pl.when(pl.col("num2").list.len() == 0).then(1).otherwise(pl.col("num2").list.len()))
        .alias("num_all_corrob_frac"),
        pl.when(pl.col("first2").is_null() | pl.col("sup_first").is_null()).then(-1)
        .when(pl.col("first2") == pl.col("sup_first")).then(1).otherwise(0).alias("first_num_support"),
    ).with_columns(
        (pl.col("num_dev") - pl.col("num_dev_corrob")).alias("num_dev_orphan"),
        (pl.col("tok_dev") - pl.col("tok_dev_corrob")).alias("tok_dev_orphan"),
        pl.col("addr_jac_support").fill_null(-1.0),
        pl.col("name_jac_support").fill_null(-1.0),
    )
    return x.select("s1_id", "mid", *[pl.col(c).cast(pl.Float32) for c in COLLECTIVE])


LO_SMOOTH = 5.0
LO_MIN_COUNT = 20


def word_logodds(rows: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Log-odds that a word ADDED to (or MISSING from) a candidate name marks a non-match.

    Decoys add real qualifier words ('Uptown', 'West', 'Holdings'), whereas the true-match
    noise adds generator filler. The label-free filler score in stage B separates these
    only partly; here the odds are learned from labelled train pairs. `rows` carries
    s1_id, mid and y and is restricted to candidates stage B found plausible.
    """
    x = rows.join(s1, on="s1_id").join(s23, on="mid").with_columns(
        pl.col("n1").str.split(" ").list.unique().alias("t1"),
        pl.col("n2").str.split(" ").list.unique().alias("t2"))
    npos = max(int(rows["y"].sum()), 1)
    nneg = max(rows.height - npos, 1)
    out = []
    for name, expr in (("extra", pl.col("t2").list.set_difference(pl.col("t1"))),
                       ("missing", pl.col("t1").list.set_difference(pl.col("t2")))):
        t = (x.select("y", expr.alias("tok")).explode("tok")
             .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
             .group_by("tok").agg((pl.col("y") == 0).sum().alias("neg"), (pl.col("y") == 1).sum().alias("pos"))
             .filter(pl.col("neg") + pl.col("pos") >= LO_MIN_COUNT)
             .select("tok", (((pl.col("neg") + LO_SMOOTH) / (pl.col("pos") + LO_SMOOTH)).log()
                             - float(__import__("math").log(nneg / npos))).cast(pl.Float32).alias("lo")))
        out.append(t)
    return out[0], out[1]


def pair_extras(sc: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame,
                lo_extra: pl.DataFrame, lo_missing: pl.DataFrame,
                s1_df: pl.DataFrame) -> pl.DataFrame:
    """Signed house-number change, and supervised log-odds of added / missing name words.

    Train shows decoys are made by ADDING a small increment to the house number: a first
    number that rises by 1..30 is a non-match 86% of the time in US train, one that falls
    by 1..30 only 4%. True-match noise is symmetric, so the sign is the signal.
    """
    x = sc.select("s1_id", "mid").join(s1, on="s1_id").join(s23, on="mid").with_columns(
        pl.col("a1").str.extract(r"\b(\d+)\b").cast(pl.Int64, strict=False).alias("f1"),
        pl.col("a2").str.extract(r"\b(\d+)\b").cast(pl.Int64, strict=False).alias("f2"),
        pl.col("n1").str.split(" ").list.unique().alias("t1"),
        pl.col("n2").str.split(" ").list.unique().alias("t2"),
    ).with_columns((pl.col("f2") - pl.col("f1")).clip(-1000, 1000).alias("d"))
    x = x.with_columns(
        pl.col("d").fill_null(0).alias("num_delta"),
        pl.col("d").is_between(1, 30).fill_null(False).cast(pl.Int8).alias("num_delta_up"),
        pl.col("d").is_between(-30, -1).fill_null(False).cast(pl.Int8).alias("num_delta_down"),
    )
    res = x.select("s1_id", "mid", "num_delta", "num_delta_up", "num_delta_down")
    for name, expr, lo in (("extra", pl.col("t2").list.set_difference(pl.col("t1")), lo_extra),
                           ("missing", pl.col("t1").list.set_difference(pl.col("t2")), lo_missing)):
        ex = (x.select("s1_id", "mid", expr.alias("tok")).explode("tok")
              .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
              .join(lo, on="tok", how="left").join(s1_df, on="tok", how="left")
              .with_columns(pl.col("df1").fill_null(0)))
        e = ex.group_by("s1_id", "mid").agg(
            # words the learned table has never seen stay NULL, not a sentinel: in train an
            # unseen extra word is mostly a typo, and a French decoy word must not read as one
            pl.col("lo").max().alias(f"{name}_lo_max"),
            pl.col("lo").sum().alias(f"{name}_lo_sum"),
            pl.col("lo").is_null().sum().alias(f"n_{name}_unknown"),
            (pl.col("df1") >= REAL_WORD_DF).sum().alias(f"{name}_real_n"),
            (pl.col("df1") + 1).log().max().alias(f"{name}_df_max"),
        )
        res = res.join(e, on=["s1_id", "mid"], how="left")
    return res.with_columns(
        pl.col("extra_lo_sum", "missing_lo_sum").fill_null(0.0),
        pl.col("n_extra_unknown", "n_missing_unknown", "extra_real_n", "missing_real_n").fill_null(0),
    ).with_columns(
        # substitution: an S1 word is gone AND a real word (not a typo, since typos are
        # rare strings) took its place - the decoy move, in any language
        ((pl.col("missing_real_n") >= 1) & (pl.col("extra_real_n") >= 1)).cast(pl.Int8).alias("subst_real"),
    ).drop("missing_real_n")


def _logodds_tables(split: str, s1: pl.DataFrame, s23: pl.DataFrame) -> dict:
    """Word log-odds tables. Train: one per fold, each learned from the OTHER fold's
    shards, so no row is described by statistics that saw its own label. Test: learned
    from all train."""
    s1t, s23t = _texts("train")
    tr = (pl.read_parquet(DIR / "train_b_scored.parquet").filter(pl.col("p") > 0.02)
          .join(labels(), on=["s1_id", "mid"], how="left").with_columns(pl.col("y").fill_null(0))
          .with_columns(shard_of().alias("_sh")))
    if split == "train":
        return {f: word_logodds(tr.filter(pl.col("_sh") % 2 != f), s1t, s23t) for f in (0, 1)}
    both = word_logodds(tr, s1t, s23t)
    return {0: both, 1: both}


def features(split: str) -> None:
    t0 = time.time()
    out = DIR / f"{split}_{OUT}_features"
    out.mkdir(parents=True, exist_ok=True)
    scored = pl.read_parquet(DIR / f"{split}_{IN}_scored.parquet").rename({"p": "p_b"})
    s1, s23 = _texts(split)
    lo = _logodds_tables(split, s1, s23)
    s1_df = (s1.select(pl.col("n1").str.split(" ").list.unique().alias("tok")).explode("tok")
             .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
             .group_by("tok").len().rename({"len": "df1"}))
    print(f"  word log-odds ready ({time.time() - t0:.0f}s)", flush=True)
    for c in range(SHARDS):
        dst = out / f"shard_{c:02d}.parquet"
        if dst.exists():
            continue
        sc = scored.filter(shard_of() == c)
        ext = pl.concat([pair_extras(sc.filter(pl.col("s1_id").hash(seed=11) % 4 == k), s1, s23,
                                     *lo[c % 2], s1_df) for k in range(4)])
        # list operations allocate per row; sub-chunks by entity keep them bounded
        col = pl.concat([collective(sc.filter(pl.col("s1_id").hash(seed=11) % 12 == k), s1, s23)
                         for k in range(12)])
        b = pl.read_parquet(DIR / f"{split}_b_features" / f"shard_{c:02d}.parquet")
        (b.join(col, on=["s1_id", "mid"], how="inner").join(ext, on=["s1_id", "mid"], how="left")
         .write_parquet(dst, compression="zstd"))
        print(f"    shard {c:02d}: {col.height:,} rows ({time.time() - t0:.0f}s)", flush=True)
        del sc, col, b, ext
        gc.collect()


def _load(split: str, shards: list[int], sample: bool = False) -> pl.DataFrame:
    frames = []
    for c in shards:
        df = pl.read_parquet(DIR / f"{split}_{OUT}_features" / f"shard_{c:02d}.parquet")
        if sample:
            df = df.filter(pl.col("s1_id") % TRAIN_FRACTION == 0)
        frames.append(df)
    return pl.concat(frames)


def fit() -> None:
    t0 = time.time()
    gt = labels()
    for fold in (0, 1):
        # every entity of the opposite fold; easy negatives (stage-B score <= 0.001) are
        # subsampled 1 in 4 and weighted 4, which keeps the full information at a
        # quarter of the memory
        tr = (_load("train", [c for c in range(SHARDS) if c % 2 != fold])
              .filter((pl.col("p_b") > EASY_P) | (pl.col("mid").hash(seed=3) % 4 == 0))
              .join(gt, on=["s1_id", "mid"], how="left").with_columns(pl.col("y").fill_null(0)))
        w = pl.when(pl.col("p_b") > EASY_P).then(1.0).otherwise(4.0)
        if PSEUDO:
            # Self-training: test pairs the previous model is sure about (p > 0.99 or
            # < 0.01) join the training set at half weight. This shows the model test's
            # own decoy mix and French vocabulary, with no labels and no external data.
            ps = (pl.read_parquet(PSEUDO).filter((pl.col("p") > 0.99) | (pl.col("p") < 0.01))
                  .select("s1_id", "mid", (pl.col("p") > 0.5).cast(pl.Int8).alias("y")))
            te = (_load("test", [c for c in range(SHARDS) if c % 2 != fold])
                  .filter((pl.col("p_b") > EASY_P) | (pl.col("mid").hash(seed=3) % 4 == 0))
                  .join(ps, on=["s1_id", "mid"], how="inner"))
            wt = pl.when(pl.col("p_b") > EASY_P).then(0.5).otherwise(2.0)
            tr = pl.concat([tr.with_columns(w.alias("_w")),
                            te.with_columns(wt.alias("_w"))], how="diagonal_relaxed")
            w = pl.col("_w")
            print(f"  fold {fold}: +{te.height:,} pseudo-labelled test rows", flush=True)
        booster = lgb.train(PARAMS, lgb.Dataset(tr.select(C_FEATURES).to_numpy().astype("float32"),
                                                label=tr["y"].to_numpy(),
                                                weight=tr.select(w)[:, 0].to_numpy()), NUM_ROUNDS)
        booster.save_model(str(WORK / f"stage_{OUT}{MTAG}_fold{fold}.txt"))
        print(f"  fold {fold}: fitted on {tr.height:,} rows ({time.time() - t0:.0f}s)", flush=True)
        if fold == 0:
            imp = sorted(zip(C_FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])
            print("  top gain:", ", ".join(n for n, _ in imp[:15]))
        del tr
        gc.collect()


def score(split: str) -> pl.DataFrame:
    t0 = time.time()
    models = [lgb.Booster(model_file=str(WORK / f"stage_{OUT}{MTAG}_fold{f}.txt")) for f in (0, 1)]
    out = []
    for c in range(SHARDS):
        part = _load(split, [c])
        x = part.select(C_FEATURES).to_numpy()
        p = models[c % 2].predict(x) if split == "train" else \
            (models[0].predict(x) + models[1].predict(x)) / 2
        out.append(part.select("s1_id", "mid").with_columns(pl.Series("p", p, dtype=pl.Float32)))
    scored = pl.concat(out)
    scored.write_parquet(DIR / f"{split}_{OUT}{MTAG}_scored.parquet", compression="zstd")
    print(f"  scored {scored.height:,} ({time.time() - t0:.0f}s)", flush=True)
    return scored


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        features(sys.argv[2])
    elif cmd == "fit":
        fit()
    elif cmd == "score":
        from stage_b import report
        s = score(sys.argv[2])
        if sys.argv[2] == "train":
            report(s)
        else:
            from predict import decide
            pairs = decide(s).select("s1_id", "mid")
            s1_ids = pl.read_parquet(PARQUET / "test_source1_norm.parquet", columns=["id"])["id"]
            write_submission(pairs, s1_ids, OUTPUT / "matching_results.tsv", "matched_entity_ids")
            print(f"  wrote matching_results.tsv ({pairs.height:,} pairs)")
