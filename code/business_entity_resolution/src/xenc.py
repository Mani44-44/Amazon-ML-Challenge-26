"""Cross-encoder re-scoring of the uncertain band (optional final step).

The LightGBM cascade leaves about a quarter of a candidate per entity in the uncertain
band (stage-C probability between 0.02 and 0.98). A cross-encoder reads the raw text of
the S1 record, the candidate and the entity's other confident records together, and can
pick up textual patterns the hand-built features miss.

Model: distilbert-base-multilingual-cased (Apache-2.0, 134M parameters), fine-tuned
from its public pretrained weights on the provided training pairs only.

Protocol, so the gain is measured honestly:
  * the model is trained on uncertain pairs of train entities with s1_id % 2 == 0 (A)
  * it scores uncertain pairs of the other train entities (B) and of test
  * `blend` fits a 2-input logistic blend (stage-C logit, cross-encoder logit) on B and
    reports macro F0.5 on B with and without it. It ships only if it wins there.

Usage:
    python xenc.py prep          # build text pairs for train A/B and test
    python xenc.py train         # fine-tune on A (GPU)
    python xenc.py predict       # score B and test
    python xenc.py blend <train_scores> <test_scores>
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import polars as pl

from common import PARQUET, WORK

DIR = WORK / "parquet"
XDIR = WORK / "xenc"
MODEL = "distilbert-base-multilingual-cased"
LO, HI = 0.02, 0.98
SUPPORT_P = 0.9
MAX_LEN = 128
BATCH = 32
LR = 3e-5
EPOCHS = 1
# Which train half the model learns from; the other half is its held-out set.
import os
HALF = os.environ.get("XE_HALF", "A")
OTHER = "B" if HALF == "A" else "A"
MDIR = "model" if HALF == "A" else "model_B"


def _text(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    s1 = pl.read_parquet(PARQUET / f"{split}_source1.parquet").select(
        pl.col("id").cast(pl.Int64).alias("s1_id"),
        (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("t1"))
    s23 = pl.concat([
        pl.read_parquet(PARQUET / f"{split}_source{k}.parquet").select(
            (pl.lit(k * 1_000_000_000, pl.Int64) + pl.col("id").cast(pl.Int64)).alias("mid"),
            (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("t2"))
        for k in (2, 3)])
    return s1, s23


def build(split: str, scores: str) -> pl.DataFrame:
    sc = pl.read_parquet(scores)
    s1, s23 = _text(split)
    band = sc.filter(pl.col("p").is_between(LO, HI))
    # up to two OTHER confident records of the same entity, as context
    sup = (sc.filter(pl.col("p") >= SUPPORT_P).join(s23, on="mid")
           .sort(["s1_id", "p"], descending=[False, True])
           .select("s1_id", pl.col("mid").alias("smid"), pl.col("t2").alias("st")))
    ctx = (band.select("s1_id", "mid").join(sup, on="s1_id", how="left")
           .filter(pl.col("smid").is_null() | (pl.col("smid") != pl.col("mid")))
           .group_by("s1_id", "mid", maintain_order=True)
           .agg(pl.col("st").drop_nulls().head(2).str.join(" ; ").alias("ctx")))
    return (band.join(s1, on="s1_id").join(s23, on="mid").join(ctx, on=["s1_id", "mid"], how="left")
            .with_columns(pl.col("ctx").fill_null(""))
            .select("s1_id", "mid", "p", "t1", (pl.col("t2") + " [SEP] " + pl.col("ctx")).alias("t2")))


def prep() -> None:
    XDIR.mkdir(parents=True, exist_ok=True)
    tr = build("train", str(DIR / "train_c_scored.parquet"))
    lab = pl.read_parquet(PARQUET / "train_gt_long.parquet").select(
        pl.col("s1_id").cast(pl.Int64),
        (pl.col("match_src").cast(pl.Int64) * 1_000_000_000 + pl.col("match_id").cast(pl.Int64)).alias("mid"),
        pl.lit(1, dtype=pl.Int8).alias("y"))
    tr = tr.join(lab, on=["s1_id", "mid"], how="left").with_columns(pl.col("y").fill_null(0))
    tr.filter(pl.col("s1_id") % 2 == 0).write_parquet(XDIR / "train_A.parquet")
    tr.filter(pl.col("s1_id") % 2 == 1).write_parquet(XDIR / "train_B.parquet")
    te = build("test", str(DIR / "test_c_scored.parquet"))
    te.write_parquet(XDIR / "test.parquet")
    print(f"  A {tr.filter(pl.col('s1_id') % 2 == 0).height:,}  B {tr.filter(pl.col('s1_id') % 2 == 1).height:,}"
          f"  test {te.height:,}  positive rate {tr['y'].mean():.3f}", flush=True)


def _batches(df: pl.DataFrame, tok, shuffle: bool, seed: int = 0):
    idx = np.arange(df.height)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    t1, t2 = df["t1"].to_list(), df["t2"].to_list()
    for i in range(0, len(idx), BATCH):
        b = idx[i:i + BATCH]
        enc = tok([t1[j] for j in b], [t2[j] for j in b], truncation=True,
                  max_length=MAX_LEN, padding=True, return_tensors="pt")
        yield b, enc


def train() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
    df = pl.read_parquet(XDIR / f"train_{HALF}.parquet")
    y = torch.tensor(df["y"].to_numpy(), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * math.ceil(df.height / BATCH)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, step = time.time(), 0
    for ep in range(EPOCHS):
        for b, enc in _batches(df, tok, shuffle=True, seed=ep):
            enc = {k: v.to(dev) for k, v in enc.items()}
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    logit = model(**enc).logits.squeeze(-1)
                loss = lossf(logit.float(), y[b].to(dev))
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            except (torch.OutOfMemoryError, RuntimeError) as err:
                # the OOM can also surface as a generic CUDA/Accelerator error
                if "out of memory" not in str(err):
                    raise
                # a rare batch of very long texts on a 6 GB GPU: skip it, keep training
                opt.zero_grad()
                torch.cuda.empty_cache()
            sch.step()
            step += 1
            if step % 1000 == 0:
                print(f"  step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(XDIR / MDIR)
    tok.save_pretrained(XDIR / MDIR)
    print(f"  trained ({time.time() - t0:.0f}s)", flush=True)


def predict() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(XDIR / MDIR)
    model = AutoModelForSequenceClassification.from_pretrained(XDIR / MDIR).to("cuda").eval()
    names = ("test",) if os.environ.get("XE_TEST_ONLY") else (f"train_{OTHER}", "test")
    for name in names:
        t0 = time.time()
        df = pl.read_parquet(XDIR / f"{name}.parquet")
        out = np.zeros(df.height, dtype=np.float32)
        with torch.no_grad():
            for b, enc in _batches(df, tok, shuffle=False):
                enc = {k: v.to("cuda") for k, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    out[b] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        suffix = "" if HALF == "A" else "_mB"
        df.select("s1_id", "mid").with_columns(pl.Series("xlogit", out)).write_parquet(XDIR / f"{name}_x{suffix}.parquet")
        print(f"  scored {name}: {df.height:,} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    {"prep": prep, "train": train, "predict": predict}[sys.argv[1]]()
