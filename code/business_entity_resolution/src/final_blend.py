"""Final step: bag the stage-C models, then blend in the cross-encoder on the uncertain band.

  1. Bagging: the mean of three stage-C models (seed 0 at feature fraction 0.8, seed 1 at
     0.7, seed 2 at 0.9). On train out-of-fold scores the bag beats the single model
     (0.98537 vs 0.98527).
  2. Cross-encoder blend: on the uncertain band (bagged p between 0.02 and 0.98), a
     two-input logistic regression combines the bagged logit with the cross-encoder
     logit. It is fitted on train quarter B1, using cross-encoder A's scores. Model A
     never saw half B. On the held-out quarter B2 it raised macro F0.5 from 0.98662 to
     0.98819 (US) and from 0.98328 to 0.98636 (India).
  3. Test: the cross-encoder input is the MEAN of models A and B's test logits (two
     independently fine-tuned models; test correlation 0.935, same scale).

Writes output/matching_results.tsv.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from common import OUTPUT, PARQUET, WORK, write_submission
from evaluate import f_beta_half
from predict import decide
from stage_a import labels

D = WORK / "parquet"
X = WORK / "xenc"
BAG = ["", "_s1", "_s2"]


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bag(split: str) -> pl.DataFrame:
    fr = None
    for t in BAG:
        d = pl.read_parquet(D / f"{split}_c{t}_scored.parquet").rename({"p": "p" + (t or "_v7")})
        fr = d if fr is None else fr.join(d, on=["s1_id", "mid"])
    return fr.select("s1_id", "mid", pl.mean_horizontal("p_v7", "p_s1", "p_s2").alias("p"))


def apply(lr, sc: pl.DataFrame, xl: pl.DataFrame) -> pl.DataFrame:
    j = sc.join(xl, on=["s1_id", "mid"], how="left")
    has = j["xlogit"].is_not_null().to_numpy()
    p = j["p"].to_numpy().copy()
    p[has] = lr.predict_proba(np.c_[lg(p[has]), j["xlogit"].to_numpy()[has]])[:, 1]
    return j.select("s1_id", "mid").with_columns(pl.Series("p", p.astype(np.float32)))


def main() -> int:
    """Test-only variant: v8's blend weights (fitted on B1, gated on B2), applied to the
    MEAN of the two cross-encoders' test logits."""
    gt = labels()
    tr = bag("train")
    xa = pl.read_parquet(X / "train_B_x.parquet")
    rows = tr.join(xa, on=["s1_id", "mid"]).join(gt, on=["s1_id", "mid"], how="left").with_columns(
        pl.col("y").fill_null(0))
    fit = rows.filter(pl.col("s1_id") % 4 == 1)
    lr = LogisticRegression().fit(np.c_[lg(fit["p"].to_numpy()), fit["xlogit"].to_numpy()], fit["y"])
    print(f"  blend coefs {lr.coef_.round(3)} (same fit as v8)", flush=True)
    ta = pl.read_parquet(X / "test_x.parquet").rename({"xlogit": "a"})
    tb = pl.read_parquet(X / "test_x_mB.parquet").rename({"xlogit": "b"})
    j = ta.join(tb, on=["s1_id", "mid"])
    print(f"  test logit correlation A vs B: {np.corrcoef(j['a'], j['b'])[0, 1]:.3f}  "
          f"std A {j['a'].std():.3f}  std B {j['b'].std():.3f}", flush=True)
    tx = j.select("s1_id", "mid", ((pl.col("a") + pl.col("b")) / 2).alias("xlogit"))
    te = apply(lr, bag("test"), tx)
    te.write_parquet(D / "test_final9_scored.parquet")
    pairs = decide(te).select("s1_id", "mid")
    s1_ids = pl.read_parquet(PARQUET / "test_source1_norm.parquet", columns=["id"])["id"]
    write_submission(pairs, s1_ids, OUTPUT / "matching_results.tsv", "matched_entity_ids")
    print(f"  wrote matching_results.tsv ({pairs.height:,} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
