"""Phase 2 (first pass): normalize every source file once and cache to Parquet.

Normalization is pure and deterministic, so it is computed once here rather than inside
blocking, feature generation and scoring. Files are processed one at a time and released
before the next is read, to stay inside the 15 GB RAM budget.
"""

from __future__ import annotations

import gc
import sys
import time

import polars as pl

from common import PARQUET, make_mid
from normalize import add_normalized_columns

KEEP = [
    "id", "mid", "country",
    "business_name", "business_address",
    "name_norm", "name_core", "name_nospace", "name_skel",
    "addr_norm", "addr_state",
]


def main() -> int:
    splits = sys.argv[1:] or ["train", "test"]
    for split in splits:
        for src in (1, 2, 3):
            dst = PARQUET / f"{split}_source{src}_norm.parquet"
            if dst.exists():
                print(f"  {dst.name:<32} exists, skipping", flush=True)
                continue
            t0 = time.time()
            df = pl.read_parquet(PARQUET / f"{split}_source{src}.parquet")
            df = add_normalized_columns(df).with_columns(make_mid(src))
            df.select(KEEP).write_parquet(dst, compression="zstd")
            print(
                f"  {dst.name:<32} {df.height:>9,} rows  {time.time() - t0:>6.1f}s",
                flush=True,
            )
            del df
            gc.collect()
    print("normalization cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
