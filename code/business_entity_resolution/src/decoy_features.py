"""Decoy-aware pair features: what KIND of difference separates two names.

89% of the confident false positives from the first model were constructed decoys, not
records owned by another entity. A decoy copies the address almost verbatim and changes
the name in a meaningful way:

  * substitutes the distinctive token     Cindie Smith Investments -> Zuluaga Smith ...
  * changes the legal form                ZR Finance LLP -> ZR Finance Private Limited
  * adds a real content word              Martin Brothers Limited -> ... Overseas Limited

True-match noise instead typos a token (Payne -> Paydne), abbreviates a suffix, or adds a
generator filler word (Services, Trading, Center, Ventures). Aggregate similarities such
as ratio or token-set ratio score both kinds alike. These features look at each token:

  tok_min_12 / tok_mean_12   for each S1 core token, its best Jaro-Winkler match among
                             the candidate's tokens (typo ~0.8-0.95, substitution ~0.4)
  tok_min_21 / tok_mean_21   the same in the other direction
  n_bad_12 / n_bad_21        tokens with no counterpart above 0.8
  first_sim                  the best match of the S1 name's first token
  extra_filler_min           among the candidate's unmatched tokens, the lowest "filler
                             score". The score is the token's log frequency in S2/S3
                             names relative to S1 names, over the split's own unlabelled
                             records. Generator filler is over-represented in S2/S3; a
                             real content word is not. This is language-agnostic, so it
                             adapts to France without French labels.
  missing_idf_max            IDF of the rarest S1 token with no counterpart
  legal_conflict             both names carry a legal form and the forms are disjoint
  legal_diff                 size of the symmetric difference of legal-form tokens
  addr_equal                 the normalized addresses are identical
  num_first_eq               the first number in each address agrees
  num_extra                  numbers in the candidate's address absent from the S1 one
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

DECOY_FEATURES = [
    "tok_min_12", "tok_mean_12", "n_bad_12", "tok_min_21", "tok_mean_21", "n_bad_21",
    "first_sim", "extra_filler_min", "missing_idf_max",
    "legal_conflict", "legal_diff", "addr_equal", "num_first_eq", "num_extra",
    # classic edit-distance similarities, computed in the same worker pass
    "name_ratio", "name_tsr", "skel_ratio", "addr_ratio", "addr_partial",
]

LEGAL = {
    "private", "limited", "corporation", "incorporated", "company", "llc", "llp", "lp",
    "plc", "pllc", "pc", "opc", "huf", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc",
    "scop", "ei", "scp", "sca", "gie", "selarl", "scm", "sccv", "earl", "gaec", "gmbh",
    "ag", "bv", "nv", "llllp", "lllp",
}
GOOD = 0.8
WORKERS = max(1, (os.cpu_count() or 2) - 2)

_FILLER: dict[str, float] = {}
_IDF: dict[str, float] = {}


def token_stats(split: str) -> tuple[dict[str, float], dict[str, float]]:
    """Filler score and IDF per name token, from the split's own S1/S2/S3 names."""
    from common import PARQUET

    def df_of(paths: list[str]) -> tuple[pl.DataFrame, int]:
        frames = [pl.read_parquet(PARQUET / p, columns=["name_core"]) for p in paths]
        names = pl.concat(frames)
        toks = (names.with_row_index("r")
                .select("r", pl.col("name_core").fill_null("").str.split(" ").alias("t"))
                .explode("t").filter(pl.col("t") != "").unique())
        return toks.group_by("t").len(), names.height

    d1, n1 = df_of([f"{split}_source1_norm.parquet"])
    d23, n23 = df_of([f"{split}_source2_norm.parquet", f"{split}_source3_norm.parquet"])
    j = d23.join(d1, on="t", how="full", coalesce=True, suffix="_1").fill_null(0)
    filler = {
        t: math.log(((c23 + 1) / n23) / ((c1 + 1) / n1))
        for t, c23, c1 in j.select("t", "len", "len_1").iter_rows()
    }
    idf = {t: math.log(n23 / (c23 + 1)) for t, c23 in d23.iter_rows()}
    return filler, idf


def _init(filler: dict[str, float], idf: dict[str, float]) -> None:
    global _FILLER, _IDF
    _FILLER, _IDF = filler, idf


def _best(tok: str, others: list[str], joined: str) -> float:
    if not others:
        return 0.0
    if len(tok) >= 3 and tok in joined:
        return 1.0
    return max(JaroWinkler.normalized_similarity(tok, o) for o in others)


def _pair(n1: str, n2: str, l1: str, l2: str, a1: str, a2: str,
          k1: str, k2: str) -> tuple:
    t1 = n1.split() if n1 else []
    t2 = n2.split() if n2 else []
    j1, j2 = "".join(t1), "".join(t2)

    b12 = [_best(t, t2, j2) for t in t1]
    b21 = [_best(t, t1, j1) for t in t2]
    tok_min_12 = min(b12) if b12 else 0.0
    tok_mean_12 = sum(b12) / len(b12) if b12 else 0.0
    tok_min_21 = min(b21) if b21 else 0.0
    tok_mean_21 = sum(b21) / len(b21) if b21 else 0.0
    n_bad_12 = sum(s < GOOD for s in b12)
    n_bad_21 = sum(s < GOOD for s in b21)
    first_sim = b12[0] if b12 else 0.0

    extra = [t for t, s in zip(t2, b21) if s < GOOD]
    extra_filler_min = min((_FILLER.get(t, 0.0) for t in extra), default=5.0)
    missing = [t for t, s in zip(t1, b12) if s < GOOD]
    missing_idf_max = max((_IDF.get(t, 15.0) for t in missing), default=0.0)

    g1 = {t for t in l1.split() if t in LEGAL}
    g2 = {t for t in l2.split() if t in LEGAL}
    legal_conflict = int(bool(g1) and bool(g2) and not (g1 & g2))
    legal_diff = len(g1 ^ g2)

    d1 = [t for t in a1.split() if t.isdigit()]
    d2 = [t for t in a2.split() if t.isdigit()]
    num_first_eq = -1 if not d1 or not d2 else int(d1[0] == d2[0])
    s1 = set(d1)
    num_extra = sum(t not in s1 for t in set(d2))

    return (tok_min_12, tok_mean_12, n_bad_12, tok_min_21, tok_mean_21, n_bad_21,
            first_sim, extra_filler_min, missing_idf_max, legal_conflict, legal_diff,
            int(a1 == a2 and a1 != ""), num_first_eq, num_extra,
            fuzz.ratio(n1, n2), fuzz.token_set_ratio(n1, n2), fuzz.ratio(k1, k2),
            fuzz.token_set_ratio(a1, a2),
            # a truncated address is a near-subsequence of the full one
            fuzz.partial_token_set_ratio(a1, a2) if a1 and a2 else 0.0)


def _chunk(args: tuple) -> list[tuple]:
    return [_pair(*r) for r in zip(*args)]


def compute(df: pl.DataFrame, filler: dict[str, float], idf: dict[str, float],
            pool: ProcessPoolExecutor | None = None) -> pl.DataFrame:
    """Decoy + edit-distance features for a frame holding the _1/_2 string columns."""
    cols = [df[c].fill_null("").to_list() for c in (
        "name_core_1", "name_core_2", "name_norm_1", "name_norm_2",
        "addr_norm_1", "addr_norm_2", "name_skel_1", "name_skel_2")]
    n = df.height
    step = max(1, math.ceil(n / (WORKERS * 4)))
    parts = [tuple(c[i:i + step] for c in cols) for i in range(0, n, step)]
    if pool is None:
        _init(filler, idf)
        rows = [r for p in parts for r in _chunk(p)]
    else:
        rows = [r for res in pool.map(_chunk, parts) for r in res]
    out = pl.DataFrame(rows, schema=DECOY_FEATURES, orient="row")
    return out.with_columns(pl.all().cast(pl.Float32))


def make_pool(filler: dict[str, float], idf: dict[str, float]) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=WORKERS, initializer=_init,
                               initargs=(filler, idf))
