"""Phase 0: convert the raw challenge TSVs into Parquet with compact int32 IDs.

Why this step exists:
  * The raw TSVs total ~2.4 GB. Carrying the "S1-"/"S2-"/"S3-" string IDs through the
    pipeline would blow the 15 GB RAM budget, so IDs are stripped to their numeric part
    and stored as int32 (max observed id is 999,999,995 < 2^31-1).
  * The source prefix is redundant with the file a record came from, so it is dropped
    here and re-attached only when writing the submission.
  * quote_char=None is REQUIRED: a few hundred rows contain a literal '"' inside the
    business name. These files are tab-delimited with no quoting, so letting the CSV
    reader treat '"' as a quote character silently mis-parses those rows.

Ground truth is additionally exploded from the packed "S2-1,S3-2,..." string into a long
(s1_id, match_id, match_src) table, which is what every later stage actually wants.
"""

from __future__ import annotations

import sys
import time
import os
from pathlib import Path

import polars as pl

# Paths can be overridden with environment variables; the defaults match the layout
# of the submission zip (code/business_entity_resolution/src three levels below ROOT).
ROOT = Path(os.environ.get("ER_ROOT", Path(__file__).resolve().parents[3]))
DATA = Path(os.environ.get(
    "ER_DATA_DIR", ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"))
OUT = ROOT / "work" / "parquet"

SCHEMA = {
    "entity_id": pl.String,
    "business_name": pl.String,
    "business_address": pl.String,
    "country": pl.String,
}


def read_source_tsv(path: Path) -> pl.DataFrame:
    """Read one *_sourceN.tsv into a frame with an int32 `id` column."""
    return (
        pl.read_csv(
            path,
            separator="\t",
            quote_char=None,  # see module docstring - literal quotes appear in names
            schema=SCHEMA,
            has_header=True,
        )
        .with_columns(
            # "S2-166376419" -> 166376419
            pl.col("entity_id").str.slice(3).cast(pl.Int32).alias("id"),
            # empty strings and the literal token "null" both mean "missing here"
            pl.col("business_name").fill_null(""),
            pl.col("business_address").fill_null(""),
        )
        .select("id", "business_name", "business_address", "country")
    )


def convert_sources() -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    for split in ("train", "test"):
        for src in (1, 2, 3):
            src_path = DATA / split / f"{split}_source{src}.tsv"
            dst_path = OUT / f"{split}_source{src}.parquet"
            t0 = time.time()
            df = read_source_tsv(src_path)

            # IDs must be unique within a source file - later stages index on them.
            n_unique = df["id"].n_unique()
            if n_unique != df.height:
                raise ValueError(
                    f"{src_path.name}: {df.height - n_unique} duplicate numeric ids"
                )

            df.write_parquet(dst_path, compression="zstd")
            counts[(split, src)] = df.height
            print(
                f"  {dst_path.name:<26} {df.height:>9,} rows  "
                f"{dst_path.stat().st_size / 1e6:>6.1f} MB  {time.time() - t0:>5.1f}s",
                flush=True,
            )
    return counts


def convert_ground_truth() -> None:
    """Explode train_ground_truth.tsv into a long (s1_id, match_id, match_src) table."""
    t0 = time.time()
    gt = pl.read_csv(
        DATA / "train" / "train_ground_truth.tsv",
        separator="\t",
        quote_char=None,
        schema={"source1_entity_id": pl.String, "matched_entity_ids": pl.String},
        has_header=True,
    ).with_columns(
        pl.col("source1_entity_id").str.slice(3).cast(pl.Int32).alias("s1_id"),
        pl.col("matched_entity_ids").fill_null(""),
    )

    # Per-entity summary: how many matches, and is it a singleton.
    summary = gt.select(
        "s1_id",
        pl.when(pl.col("matched_entity_ids") == "")
        .then(0)
        .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)
        .cast(pl.Int16)
        .alias("n_matches"),
    )
    summary.write_parquet(OUT / "train_gt_summary.parquet", compression="zstd")

    # Long form, singletons dropped (they contribute no rows by construction).
    long = (
        gt.filter(pl.col("matched_entity_ids") != "")
        .select(
            "s1_id",
            pl.col("matched_entity_ids").str.split(",").alias("m"),
        )
        .explode("m")
        .select(
            "s1_id",
            pl.col("m").str.slice(3).cast(pl.Int32).alias("match_id"),
            pl.col("m").str.slice(1, 1).cast(pl.Int8).alias("match_src"),
        )
    )
    long.write_parquet(OUT / "train_gt_long.parquet", compression="zstd")

    n_singleton = int((summary["n_matches"] == 0).sum())
    print(
        f"  train_gt_long.parquet      {long.height:>9,} links  "
        f"({n_singleton:,} singleton entities)  {time.time() - t0:>5.1f}s",
        flush=True,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"reading from {DATA}")
    print(f"writing to   {OUT}\n")
    convert_sources()
    convert_ground_truth()
    print("\nphase 0 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
