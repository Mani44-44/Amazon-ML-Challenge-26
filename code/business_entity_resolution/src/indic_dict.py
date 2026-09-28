"""Learn an Indic-script -> Latin token dictionary from the training ground truth.

22% of India S2/S3 names are written in an Indic script, and the generator makes them by
transliterating the S1 name one token at a time: every one of the 551k such training
links has the same token count as its S1 name. Phonetic romanization does not undo this
('Private Limited' -> 'praaivett limittedd'), so those links share no token with their
S1 record and blocking loses 22% of them.

Because the generator's vocabulary is small (about 1.3k distinct Indic tokens), the
inverse mapping can be read straight off the aligned training pairs: pair tokens by
position and keep the majority Latin token for each Indic token. Measured 96.6% purity
and 96.4% coverage of test India Indic name tokens. Tokens the dictionary has not seen
fall back to character-level romanization in `normalize.romanize`.

Built only from the provided training files. Writes `indic_dict.json` next to this file.

Usage:  python indic_dict.py
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from common import PARQUET

INDIC = (r"[\p{Devanagari}\p{Bengali}\p{Telugu}\p{Tamil}\p{Kannada}\p{Gujarati}"
         r"\p{Gurmukhi}\p{Malayalam}\p{Oriya}]")
DST = Path(__file__).with_name("indic_dict.json")
MIN_COUNT = 2


def main() -> int:
    gt = pl.read_parquet(PARQUET / "train_gt_long.parquet").select(
        pl.col("s1_id").cast(pl.Int64),
        (pl.col("match_src").cast(pl.Int64) * 1_000_000_000
         + pl.col("match_id").cast(pl.Int64)).alias("mid"),
    )
    other = pl.concat([
        pl.read_parquet(PARQUET / f"train_source{k}.parquet",
                        columns=["id", "business_name"])
        .select((pl.lit(k * 1_000_000_000, pl.Int64) + pl.col("id").cast(pl.Int64))
                .alias("mid"), pl.col("business_name").alias("n2"))
        .filter(pl.col("n2").str.contains(INDIC))
        for k in (2, 3)
    ])
    s1 = pl.read_parquet(PARQUET / "train_source1.parquet",
                         columns=["id", "business_name"]).select(
        pl.col("id").cast(pl.Int64).alias("s1_id"), pl.col("business_name").alias("n1"))

    pairs = (
        gt.join(other, on="mid").join(s1, on="s1_id")
        .with_columns(pl.col("n1").str.split(" ").alias("t1"),
                      pl.col("n2").str.split(" ").alias("t2"))
        .filter(pl.col("t1").list.len() == pl.col("t2").list.len())
        .select("t1", "t2").explode(["t1", "t2"])
        .filter(pl.col("t2").str.contains(INDIC))
    )
    counts = pairs.group_by("t2", "t1").len()
    best = (
        counts.sort("len", descending=True)
        .group_by("t2", maintain_order=True).head(1)
        .join(counts.group_by("t2").agg(pl.col("len").sum().alias("tot")), on="t2")
        .filter(pl.col("tot") >= MIN_COUNT)
    )
    mapping = dict(zip(best["t2"].to_list(), best["t1"].to_list()))
    DST.write_text(json.dumps(mapping, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    purity = best["len"].sum() / best["tot"].sum()
    print(f"  {len(mapping):,} tokens, purity {purity:.4f} -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
