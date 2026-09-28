"""Shared conventions: paths, the combined match key, and TSV submission writing."""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

# Paths can be overridden with environment variables; the defaults match the layout
# of the submission zip (code/business_entity_resolution/src three levels below ROOT).
ROOT = Path(os.environ.get("ER_ROOT", Path(__file__).resolve().parents[3]))
DATA = Path(os.environ.get(
    "ER_DATA_DIR", ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"))
PARQUET = ROOT / "work" / "parquet"
WORK = ROOT / "work"
OUTPUT = ROOT / "output"

# A numeric entity id is unique only WITHIN a source file: S2-123 and S3-123 can both
# exist. Every cross-source structure therefore keys on `mid`, a single int64 that packs
# the source and the id together. Max observed id is 999,999,995 < 1e9, so this is exact.
SRC_STRIDE = 1_000_000_000


def make_mid(src: int | pl.Expr, id_col: str | pl.Expr = "id") -> pl.Expr:
    """Pack (source, id) into the combined int64 match key."""
    id_expr = pl.col(id_col) if isinstance(id_col, str) else id_col
    src_expr = pl.lit(src) if isinstance(src, int) else src
    return (src_expr.cast(pl.Int64) * SRC_STRIDE + id_expr.cast(pl.Int64)).alias("mid")


def mid_to_entity_id(mid_col: str = "mid") -> pl.Expr:
    """Unpack a `mid` back into the challenge's string form, e.g. 'S2-166376419'."""
    mid = pl.col(mid_col)
    return (
        pl.lit("S")
        + (mid // SRC_STRIDE).cast(pl.String)
        + pl.lit("-")
        + (mid % SRC_STRIDE).cast(pl.String)
    )


def load_source(split: str, src: int) -> pl.DataFrame:
    """Load one prepared source table, with the combined `mid` key attached."""
    return pl.read_parquet(PARQUET / f"{split}_source{src}.parquet").with_columns(
        make_mid(src)
    )


def write_submission(
    pairs: pl.DataFrame,
    s1_ids: pl.Series,
    path: Path,
    id_column: str,
) -> None:
    """Write a submission TSV: exactly one row per S1 entity, tab-separated.

    `pairs` is a long (s1_id, mid) frame; entities absent from it get an empty list.
    Both output files share this format, differing only in the second column name.
    """
    # `s1_id` arrives as Int32 from the source files but Int64 from the candidate
    # frames; align both sides so the left join below cannot fail on dtype.
    s1_ids = s1_ids.cast(pl.Int64)
    pairs = pairs.with_columns(pl.col("s1_id").cast(pl.Int64))

    per_entity = (
        pairs.unique(subset=["s1_id", "mid"])
        .sort(["s1_id", "mid"])
        .group_by("s1_id", maintain_order=True)
        .agg(mid_to_entity_id().str.join(",").alias(id_column))
    )

    # Left join off the full S1 list so singletons appear with an empty field.
    out = (
        pl.DataFrame({"s1_id": s1_ids})
        .join(per_entity, on="s1_id", how="left")
        .with_columns(
            (pl.lit("S1-") + pl.col("s1_id").cast(pl.String)).alias("source1_entity_id"),
            pl.col(id_column).fill_null(""),
        )
        .select("source1_entity_id", id_column)
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
