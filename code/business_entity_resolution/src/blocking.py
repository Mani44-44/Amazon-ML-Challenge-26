"""Phase 3: multi-channel candidate generation.

Four channels, unioned, then scored and truncated to the top K per S1 entity:

  (a) address token-pair keys  - the workhorse. 95% of true pairs share >=2 address
      tokens, and a key made of two tokens is selective enough to keep a usable df cap.
  (b) name token-pair keys     - carries the ~12% of records whose address is empty or
      digit-free, which channel (a) structurally cannot reach.
  (c) exact `name_nospace`     - cheap, high precision, and catches short names whose
      token count is below the pair-key minimum of 2.
  (d) exact `name_skel`        - the consonant skeleton; weak alone (it lifted the
      transliterated segment only 17.1% -> 19.7%) but free once computed.

Channels (a) and (b) restrict each record to its k rarest JOINT-VOCABULARY tokens: a
token absent from the other side is unmatchable, and typo tokens - which are abundant
here - are exactly the rarest thing in a record, so without the joint-vocabulary filter
they would occupy every slot. Both sides rank by the same S2/S3 document frequency with a
tiebreak on the token string, so the two top-k sets stay aligned.

Candidates are scored by summed IDF over the shared keys and truncated to TOP_K. Nothing
here decides matches; this stage only has to put the true record in the candidate list,
and `candidate_pairs.tsv` is exactly what it emits.

Usage:
    python blocking.py train      # writes candidates + measures recall@K vs ground truth
    python blocking.py test       # writes candidates for submission
"""

from __future__ import annotations

import gc
import os
import sys
import time

import polars as pl

from common import PARQUET, WORK

# Tuned in pairkey_stats.py; see PLAN.md for the measured recall/cost curve.
K_ADDR = 6          # rarest joint-vocab address tokens per record
K_NAME = 6          # rarest joint-vocab name tokens per record
CAP_ADDR = 100      # max S2/S3 frequency of an address pair key
CAP_NAME = 100      # max S2/S3 frequency of a name pair key
CAP_EXACT = 30      # max S2/S3 frequency of an exact-name key
CAP_CROSS = int(os.environ.get("CAP_CROSS", 120))  # max S2/S3 frequency of a name x address key
CROSS_NAME_TOKS = 4 # name tokens per record used in cross keys
# per-channel candidates kept per S1 entity before the union
CH_K = {"addr": int(os.environ.get("K_CH_ADDR", 35)),
        "name": int(os.environ.get("K_CH_NAME", 25)),
        "cross": int(os.environ.get("K_CH_CROSS", 45)),
        "fullx": int(os.environ.get("K_CH_FULLX", 20))}
CH_LABELS = ["addr", "name", "cross", "fullx", "exact"]
CAP_FULL = int(os.environ.get("CAP_FULL", 60))  # max S2/S3 frequency of a whole-name x address key
TOP_K = int(os.environ.get("BLOCK_TOPK", 30))  # candidates kept per S1 entity
S1_CHUNKS = 8       # S1 hash partitions, to bound peak memory

COUNTRIES = ["US", "India", "France"]


# --------------------------------------------------------------------------- #
# Shared pieces
# --------------------------------------------------------------------------- #


def _load(split: str, country: str, col: str, side: str) -> pl.DataFrame:
    if side == "s1":
        return (
            pl.read_parquet(PARQUET / f"{split}_source1_norm.parquet")
            .filter(pl.col("country") == country)
            .select(pl.col("id").cast(pl.Int64).alias("rid"), pl.col(col).alias("txt"))
        )
    return pl.concat(
        [
            pl.read_parquet(PARQUET / f"{split}_source{s}_norm.parquet")
            .filter(pl.col("country") == country)
            .select(pl.col("mid").alias("rid"), pl.col(col).alias("txt"))
            for s in (2, 3)
        ]
    )


def _exploded(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select("rid", pl.col("txt").str.split(" ").alias("tok"))
        .explode("tok")
        .filter(pl.col("tok") != "")
        .unique(subset=["rid", "tok"])
    )


def _joint_vocab(ex1: pl.DataFrame, ex23: pl.DataFrame) -> pl.DataFrame:
    """Tokens occurring on both sides, with their S2/S3 df and a dense integer id."""
    df1 = ex1.group_by("tok").len().rename({"len": "df1"})
    df23 = ex23.group_by("tok").len().rename({"len": "df"})
    return (
        df23.join(df1, on="tok", how="inner")
        .sort("tok")
        .with_row_index("tid")
        .select("tok", "df", pl.col("tid").cast(pl.Int64))
    )


def _pair_keys(ex: pl.DataFrame, vocab: pl.DataFrame, k: int) -> pl.DataFrame:
    """(rid, key, maxdf) for every pair drawn from a record's k rarest joint tokens.

    The pairs are emitted one index combination at a time (k*(k-1)/2 vectorized passes)
    rather than by looping over records in Python. Both produce identical output, but the
    Python version spent ~16 min on a single country in `pairkey_stats.py`, essentially
    all of the runtime, because it crosses the Python boundary once per record.
    """
    topk = (
        ex.join(vocab, on="tok", how="inner")  # inner join = joint-vocabulary filter
        .sort(["rid", "df", "tok"])  # tok breaks df ties identically on both sides
        .group_by("rid", maintain_order=True)
        .agg(pl.col("tid").head(k), pl.col("df").head(k))
        .with_columns(pl.col("tid").list.len().alias("n"))
    )

    frames = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = pl.col("tid").list.get(i), pl.col("tid").list.get(j)
            frames.append(
                topk.filter(pl.col("n") > j).select(
                    "rid",
                    # sorted pair packed into one u64; both ids are < 2^32
                    (
                        pl.min_horizontal(a, b) * 4_294_967_296
                        + pl.max_horizontal(a, b)
                    ).alias("key"),
                    # the rarer token of the pair drives its weight
                    pl.max_horizontal(
                        pl.col("df").list.get(i), pl.col("df").list.get(j)
                    ).alias("maxdf"),
                )
            )
    return pl.concat(frames) if frames else topk.select(
        "rid", pl.lit(0, dtype=pl.Int64).alias("key"), pl.lit(0).alias("maxdf")
    ).clear()


def _cross_tokens(split: str, country: str, side: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Exploded (rid, tok) for name (first CROSS_NAME_TOKS tokens) and address."""
    n = _exploded(_load(split, country, "name_core", side))
    n = (n.with_columns(pl.int_range(pl.len()).over("rid").alias("_i"))
         .filter(pl.col("_i") < CROSS_NAME_TOKS).drop("_i"))
    a = _exploded(_load(split, country, "addr_norm", side))
    return n, a


def _fullname_tokens(split: str, country: str, side: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(rid, tok) where tok is the WHOLE name: exact core name and its consonant skeleton."""
    a = _load(split, country, "name_nospace", side).filter(pl.col("txt").str.len_chars() >= 3)
    b = _load(split, country, "name_skel", side).filter(pl.col("txt").str.len_chars() >= 3)
    n = pl.concat([a.select("rid", pl.col("txt").alias("tok")),
                   b.select("rid", (pl.lit("~") + pl.col("txt")).alias("tok"))]).unique()
    return n, _exploded(_load(split, country, "addr_norm", side))


def _cross_channel(split: str, country: str, n23: int,
                   full: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Name-token x address-token keys.

    India's remaining misses are generic names ('good food', 'om impex') on truncated
    addresses ('b 401 nashik mh'). Neither side is distinctive alone, so neither the name
    nor the address channel can keep them under its cap. Their conjunction is distinctive:
    'om' AND '501', 'good' AND 'ludhiana'. A key is one name token paired with one address
    token. Every address token is kept, not only the k rarest, so a short address matches
    on whatever it still has. Tokens are mapped to integer ids before the cross product so
    the ~150M-row intermediate holds no strings.
    """
    toks = _fullname_tokens if full else _cross_tokens
    n1, a1 = toks(split, country, "s1")
    n2, a2 = toks(split, country, "s23")
    nv = (pl.concat([n1["tok"], n2["tok"]]).unique().to_frame("tok")
          .with_row_index("id").with_columns(pl.col("id").cast(pl.UInt64)))
    av = (pl.concat([a1["tok"], a2["tok"]]).unique().to_frame("tok")
          .with_row_index("id").with_columns(pl.col("id").cast(pl.UInt64)))

    def keyed(n: pl.DataFrame, a: pl.DataFrame) -> pl.DataFrame:
        n = n.join(nv, on="tok").select("rid", pl.col("id").alias("nid"))
        a = a.join(av, on="tok").select("rid", pl.col("id").alias("aid"))
        return (n.join(a, on="rid", how="inner")
                .select("rid", (pl.col("nid") * 4_294_967_296 + pl.col("aid")).alias("key"))
                .unique())

    a = keyed(n1, a1)
    del n1, a1
    b = keyed(n2, a2)
    del n2, a2, nv, av
    gc.collect()
    df = b.group_by("key").len().filter(pl.col("len") <= (CAP_FULL if full else CAP_CROSS))
    a = a.join(df, on="key", how="inner").rename({"len": "maxdf"})
    b = b.join(df, on="key", how="inner").rename({"len": "maxdf"})
    return a, b


def _exact_keys(split: str, country: str, col: str, side: str) -> pl.DataFrame:
    """Exact whole-string keys (name_nospace / name_skel), hashed to u64."""
    df = _load(split, country, col, side).filter(pl.col("txt").str.len_chars() >= 4)
    # hash() is u64 and overflows a signed cast; the exact channels join only against
    # each other, so the key dtype need not match the packed pair keys.
    return df.select("rid", pl.col("txt").hash(seed=17).alias("key"))


# --------------------------------------------------------------------------- #
# Per-country candidate generation
# --------------------------------------------------------------------------- #


def _channel(split: str, country: str, col: str, k: int, cap: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build capped, IDF-weighted pair keys for one channel; returns (s1 side, s23 side)."""
    s1 = _load(split, country, col, "s1")
    s23 = _load(split, country, col, "s23")
    ex1, ex23 = _exploded(s1), _exploded(s23)
    vocab = _joint_vocab(ex1, ex23)
    del s1, s23
    gc.collect()

    a = _pair_keys(ex1, vocab, k)
    b = _pair_keys(ex23, vocab, k)
    del ex1, ex23, vocab
    gc.collect()

    # Cap on the *corpus* frequency of the key, not on its df estimate: a key held by
    # hundreds of S2/S3 records cannot discriminate and would dominate the join cost.
    keep = b.group_by("key").len().filter(pl.col("len") <= cap).drop("len")
    a = a.join(keep, on="key", how="semi")
    b = b.join(keep, on="key", how="semi")
    return a, b


def _weighted_pairs(a: pl.DataFrame, b: pl.DataFrame, n23: int, chunk: int) -> pl.DataFrame:
    """Join one S1 hash-partition against the S2/S3 side, accumulating IDF weight."""
    a = a.filter(pl.col("rid").hash(seed=5) % S1_CHUNKS == chunk)
    if not a.height:
        return pl.DataFrame(schema={"s1_id": pl.Int64, "mid": pl.Int64, "w": pl.Float64})
    return (
        a.join(b.rename({"rid": "mid", "maxdf": "maxdf_b"}), on="key", how="inner")
        .with_columns(
            (pl.lit(float(n23)) / pl.col("maxdf").cast(pl.Float64)).log().alias("idf")
        )
        .group_by(["rid", "mid"])
        .agg(pl.col("idf").sum().alias("w"))
        .rename({"rid": "s1_id"})
    )


def build_country(split: str, country: str) -> pl.DataFrame:
    t0 = time.time()
    n23 = sum(
        pl.read_parquet(PARQUET / f"{split}_source{s}_norm.parquet", columns=["country"])
        .filter(pl.col("country") == country)
        .height
        for s in (2, 3)
    )
    if not n23:
        return pl.DataFrame(schema={"s1_id": pl.Int64, "mid": pl.Int64, "w": pl.Float64})

    # Build every channel's key tables first, then consume them one S1 partition at a
    # time. The earlier version accumulated all channels' pairs before truncating, which
    # for US meant holding ~260M rows (>6 GB) and drove the machine into swap; truncating
    # inside the partition loop keeps only TOP_K rows per entity alive.
    # Every channel keeps its OWN top-K per entity, and the union of those lists is the
    # candidate set. A single top-K over summed weights let long-address neighbours
    # (many shared pair keys) crowd out a true record that matches on a short address or
    # a generic name through only one or two strong keys. Per-channel weights are kept
    # as separate columns because they are also model features.
    channels: list[tuple[str, pl.DataFrame, pl.DataFrame, int]] = []
    for label, col, k, cap in (
        ("addr", "addr_norm", K_ADDR, CAP_ADDR),
        ("name", "name_core", K_NAME, CAP_NAME),
    ):
        a, b = _channel(split, country, col, k, cap)
        print(f"    {country}/{label}: s1_keys={a.height:,} s23_keys={b.height:,}",
              flush=True)
        channels.append((label, a, b, CH_K[label]))

    a, b = _cross_channel(split, country, n23)
    print(f"    {country}/cross: s1_keys={a.height:,} s23_keys={b.height:,}", flush=True)
    channels.append(("cross", a, b, CH_K["cross"]))

    # Whole name x address token: a generic name that many S1 entities share ('good
    # enterprises') is pinned down by one surviving address token of a truncated address
    # ('b 25'), a conjunction far rarer than any single name token x address token.
    a, b = _cross_channel(split, country, n23, full=True)
    print(f"    {country}/fullx: s1_keys={a.height:,} s23_keys={b.height:,}", flush=True)
    channels.append(("fullx", a, b, CH_K["fullx"]))

    # Exact-name channels: cheap, and they reach records the pair keys miss.
    for col in ("name_nospace", "name_skel"):
        a = _exact_keys(split, country, col, "s1")
        b = _exact_keys(split, country, col, "s23")
        keep = b.group_by("key").len().filter(pl.col("len") <= CAP_EXACT).drop("len")
        channels.append(("exact", a.join(keep, on="key", how="semi"),
                         b.join(keep, on="key", how="semi"), 0))
        del keep
        gc.collect()

    parts: list[pl.DataFrame] = []
    for c in range(S1_CHUNKS):
        chunk_parts = []
        for label, a, b, k in channels:
            if label != "exact":
                p = _weighted_pairs(a, b, n23, c)
                p = (p.sort(["s1_id", "w"], descending=[False, True])
                     .group_by("s1_id", maintain_order=True).head(k))
            else:
                sel = a.filter(pl.col("rid").hash(seed=5) % S1_CHUNKS == c)
                p = (sel.join(b.rename({"rid": "mid"}), on="key", how="inner")
                     .select(pl.col("rid").alias("s1_id"), "mid").unique()
                     .with_columns(pl.lit(1.0).alias("w")))
            chunk_parts.append(p.select("s1_id", "mid", pl.col("w").alias(f"w_{label}")))
        if chunk_parts:
            u = chunk_parts[0]
            for p in chunk_parts[1:]:
                u = u.join(p, on=["s1_id", "mid"], how="full", coalesce=True)
            u = (u.group_by(["s1_id", "mid"]).agg(pl.col(f"w_{l}").max() for l in CH_LABELS)
                 .with_columns([pl.col(f"w_{l}").fill_null(0.0).cast(pl.Float32)
                                for l in CH_LABELS])
                 .with_columns((pl.col("w_addr") + pl.col("w_name") + pl.col("w_cross")
                                + pl.col("w_fullx") + 4.0 * pl.col("w_exact")).alias("w")))
            parts.append(u)
        del chunk_parts
        gc.collect()
        print(f"      chunk {c + 1}/{S1_CHUNKS} done ({time.time() - t0:.0f}s)",
              flush=True)

    del channels
    gc.collect()
    out = pl.concat(parts) if parts else pl.DataFrame(
        schema={"s1_id": pl.Int64, "mid": pl.Int64, "w": pl.Float64}
    )
    print(f"    {country}: {out.height:,} candidates "
          f"({out.height / max(out['s1_id'].n_unique(), 1):.1f}/entity, "
          f"{time.time() - t0:.0f}s)", flush=True)
    return out


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    t0 = time.time()

    present = (
        pl.read_parquet(PARQUET / f"{split}_source1_norm.parquet", columns=["country"])
        ["country"].unique().to_list()
    )
    frames = [build_country(split, c) for c in COUNTRIES if c in present]
    cands = pl.concat(frames)

    dst = WORK / "parquet" / os.environ.get("BLOCK_DST", f"{split}_candidates.parquet")
    cands.write_parquet(dst, compression="zstd")
    print(f"  wrote {dst}  ({cands.height:,} rows, {time.time() - t0:.0f}s)")

    if split == "test":
        # candidate_pairs.tsv must be exactly what the model scores, so it is written
        # from the same frame the model will consume.
        from common import OUTPUT, write_submission

        s1_ids = pl.read_parquet(
            PARQUET / "test_source1_norm.parquet", columns=["id"]
        )["id"]
        write_submission(
            cands.select("s1_id", "mid"),
            s1_ids,
            OUTPUT / "candidate_pairs.tsv",
            "candidate_entity_ids",
        )
        print(f"  wrote {OUTPUT / 'candidate_pairs.tsv'}")

    if split == "train":
        from evaluate import candidate_recall

        gt = pl.read_parquet(PARQUET / "train_gt_long.parquet").select(
            "s1_id",
            (
                pl.col("match_src").cast(pl.Int64) * 1_000_000_000
                + pl.col("match_id").cast(pl.Int64)
            ).alias("mid"),
        )
        r = candidate_recall(cands.select("s1_id", "mid"), gt)
        print(f"  recall {r['pair_recall']:.4f}   {r['cands_per_entity']:.1f} cands/entity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
