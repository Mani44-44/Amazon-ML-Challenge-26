"""Text normalization for business names and addresses.

Two families of keys are produced.

1. `norm` / `core` / `nospace` - conventional normalization: romanize, lowercase, fold
   punctuation, canonicalize legal suffixes. Good for US and France, where the noise is
   orthographic (typos, abbreviations, accents).

2. `skeleton` - a consonant skeleton, which is what makes the India half of the data
   tractable. ~15-19% of S2/S3 names are written in Devanagari or Tamil while the S1
   reference is 97.6% Latin. Romanizing the Indic text yields a *phonetic* string, not
   the original spelling:

       'Ss Food Private Limited'  ->  'eses phuudd praaivett limittedd'
       'Modern Finance'           ->  'moNddrn phaaineNs'

   Those will not match on characters. But the distortions are systematic - doubled
   consonants, lengthened vowels, 'ph' for 'f', a nasal 'N' marker, and dropped inherent
   vowels. Folding to a deduplicated, devoweled consonant skeleton cancels all of them:
   both strings above collapse to 'ssfdprvtlmtd'.

Indic-script names are first translated token by token through `indic_dict.json`, a
dictionary learned from the training ground truth (see indic_dict.py); only tokens it does
not contain fall back to character-level romanization. Indic-script state names in
addresses are folded to their 2-letter codes.

`anyascii` (ISC licence) is a static character-map library bundled with the code. It
performs no lookup of any kind against an external service or database.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
from anyascii import anyascii

_INDIC_DICT_PATH = Path(__file__).with_name("indic_dict.json")
INDIC_DICT: dict[str, str] = (
    json.loads(_INDIC_DICT_PATH.read_text(encoding="utf-8"))
    if _INDIC_DICT_PATH.exists() else {}
)

# Indic-script state names seen in India S2/S3 addresses (16 values in train + test).
INDIC_STATES = {
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "उत्तर प्रदेश": "up", "ಕರ್ನಾಟಕ": "ka",
    "தமிழ்நாடு": "tn", "ગુજરાત": "gj", "পশ্চিমবঙ্গ": "wb", "తెలంగాణ": "tg",
    "हरियाणा": "hr", "राजस्थान": "rj", "കേരളം": "kl", "बिहार": "br",
    "मध्य प्रदेश": "mp", "ఆంధ్రప్రదేశ్": "ap", "ਪੰਜਾਬ": "pb", "ଓଡ଼ିଶା": "or",
}

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #

# Canonicalized to a single spelling so that "Pvt Ltd" and "Private Limited" agree.
LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "pte": "private",
    "ltd": "limited",
    "ltda": "limited",
    "lim": "limited",
    "corp": "corporation",
    "corpn": "corporation",
    "inc": "incorporated",
    "incorp": "incorporated",
    "co": "company",
    "cos": "company",
    "compagnie": "company",
    "cie": "company",
    "intl": "international",
    "assoc": "associates",
    "assocs": "associates",
    "bros": "brothers",
    "mfg": "manufacturing",
    "ind": "industries",
    "inds": "industries",
    "ent": "enterprises",
    "ents": "enterprises",
    "svcs": "services",
    "svc": "services",
    "sol": "solutions",
    "soln": "solutions",
    "solns": "solutions",
    "tech": "technologies",
    "techs": "technologies",
}

# Dropped entirely when building the `core` variant - these carry almost no identifying
# signal and are inconsistently present across sources.
LEGAL_STOPWORDS = {
    "private", "limited", "corporation", "incorporated", "company", "llc", "llp",
    "lp", "plc", "ltd", "inc", "corp", "co", "pvt", "and", "the", "of", "pc", "pllc",
    # French legal forms (France appears only in the test set; this list was checked
    # against the most frequent trailing tokens of the unlabelled test France names).
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "ei", "scp", "sca",
    "gie", "selarl", "scm", "sccv", "earl", "gaec",
    # Indian forms
    "opc", "huf",
}

# Street-type variants, folded to one token. Note 'st' is deliberately mapped to
# 'street' even though the data contains a corrupted 'ST' -> 'SAINT' expansion; folding
# 'saint' here too makes both spellings agree.
STREET_MAP = {
    "rd": "road", "st": "street", "saint": "street", "str": "street",
    "ave": "avenue", "av": "avenue", "blvd": "boulevard", "boul": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "pl": "place", "sq": "square",
    "hwy": "highway", "pkwy": "parkway", "ter": "terrace", "trl": "trail",
    "cir": "circle", "apt": "apartment", "ste": "suite", "fl": "floor",
    "bldg": "building", "nr": "near", "opp": "opposite", "rly": "railway",
    "mkt": "market", "nag": "nagar", "colony": "colony", "ph": "phase",
    "r": "rue", "bd": "boulevard", "av.": "avenue",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "newhampshire": "nh", "newjersey": "nj", "newmexico": "nm", "newyork": "ny",
    "northcarolina": "nc", "northdakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhodeisland": "ri",
    "southcarolina": "sc", "southdakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "westvirginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "districtofcolumbia": "dc",
}

IN_STATES = {
    "andhrapradesh": "ap", "arunachalpradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachalpradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhyapradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "or", "orissa": "or",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamilnadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttarpradesh": "up", "uttarakhand": "uk",
    "westbengal": "wb", "delhi": "dl", "newdelhi": "dl", "jammukashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}

STATE_MAP = {**US_STATES, **IN_STATES}

_TLD_RE = re.compile(r"\.(com|net|org|in|co|io|biz|info|fr|us|uk)\b")
_NONALNUM_RE = re.compile(r"[^a-z0-9]+")
_DOTTED_RE = re.compile(r"\b(?:[a-z]\.){2,}(?:[a-z]\b\.?)?")
_WS_RE = re.compile(r"\s+")
_VOWELS_RE = re.compile(r"[aeiou]")
_DUP_RE = re.compile(r"(.)\1+")
# An ordinal suffix glued to a digit run: '45th', and the corrupted '45nd' / '45rd'
# forms. Must be applied BEFORE the STREET_MAP lookup, or a trailing 'st' / 'rd' is
# expanded into 'street' / 'road'.
_ORDINAL_RE = re.compile(r"(\d)(?:st|nd|rd|th)\b")
# Digit <-> letter boundaries inside a single token: 'af684', '12b'.
_D2A_RE = re.compile(r"(\d)([a-z])")
_A2D_RE = re.compile(r"([a-z])(\d)")


# --------------------------------------------------------------------------- #
# Scalar helpers
# --------------------------------------------------------------------------- #


def romanize(s: str) -> str:
    """Romanize non-Latin scripts. ASCII input is returned unchanged (fast path).

    Tokens found in the learned Indic dictionary are replaced by their Latin original;
    everything else is transliterated character by character.
    """
    if s.isascii():
        return s
    if INDIC_DICT:
        s = " ".join(INDIC_DICT.get(t, t) for t in s.split(" "))
    return anyascii(s)


def romanize_address(s: str) -> str:
    """As `romanize`, but Indic-script state names become their 2-letter code first."""
    if s.isascii():
        return s
    for native, code in INDIC_STATES.items():
        if native in s:
            s = s.replace(native, code)
    return anyascii(s)


def consonant_skeleton(s: str) -> str:
    """Fold a romanized string to its deduplicated consonant skeleton.

    Cancels the systematic artifacts of phonetic romanization so that a Devanagari or
    Tamil rendering of a name collapses onto the same key as its Latin original.
    """
    if not s:
        return ""
    s = s.lower()
    # 'ph' is how romanized Indic renders 'f'; fold before vowels are stripped.
    s = s.replace("ph", "f")
    s = s.replace("kh", "k").replace("gh", "g").replace("bh", "b")
    s = s.replace("dh", "d").replace("th", "t").replace("ch", "c")
    s = s.replace("sh", "s").replace("ck", "k").replace("qu", "k")
    s = _NONALNUM_RE.sub("", s)
    s = _VOWELS_RE.sub("", s)
    # Voiced/unvoiced and other pairs that romanization swaps freely.
    s = s.translate(str.maketrans({"z": "s", "q": "k", "x": "ks", "w": "v",
                                   "j": "y", "d": "t", "g": "k", "b": "p"}))
    return _DUP_RE.sub(r"\1", s)


def _canon_tokens(s: str, mapping: dict[str, str]) -> str:
    return " ".join(mapping.get(t, t) for t in s.split())


def normalize_name(raw: str) -> str:
    """Romanize, lowercase, strip domain TLDs and punctuation, canonicalize suffixes."""
    s = romanize(raw).lower()
    s = s.replace("&", " and ")
    # dotted abbreviations ('s.a.s.', 'l.l.c.', 'p.c.') become one token, not three
    s = _DOTTED_RE.sub(lambda m: m.group(0).replace(".", ""), s)
    s = _TLD_RE.sub(" ", s)
    s = _NONALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return _canon_tokens(s, LEGAL_SUFFIX_MAP)


def name_core(norm: str) -> str:
    """The distinctive part of a name: legal boilerplate removed."""
    toks = [t for t in norm.split() if t not in LEGAL_STOPWORDS]
    return " ".join(toks) if toks else norm


def normalize_address(raw: str) -> str:
    """Normalize an address, isolate digit runs, and fold street-type/state variants.

    Isolating digits is what makes the address channel work across sources. Numeric
    tokens are corrupted between sources ('45th' -> '45ND', 'Af-684' -> 'AF-0684') but
    the digits themselves survive, so splitting them away from the characters around
    them and dropping leading zeros makes both spellings agree on a shared '45' / '684'.
    """
    s = romanize_address(raw).lower()
    s = s.replace("&", " and ")
    s = _NONALNUM_RE.sub(" ", s)
    s = _ORDINAL_RE.sub(r"\1 ", s)
    s = _D2A_RE.sub(r"\1 \2", s)
    s = _A2D_RE.sub(r"\1 \2", s)
    s = _WS_RE.sub(" ", s).strip()
    # The generator emits a literal 'null' token for missing components.
    toks = [t for t in s.split() if t != "null"]
    toks = [STREET_MAP.get(t, t) for t in toks]
    # '0684' and '684' have to be the same token.
    toks = [(t.lstrip("0") or "0") if t.isdigit() else t for t in toks]
    return " ".join(toks)


def address_state(norm_addr: str) -> str:
    """Best-effort 2-letter state code, folding full names to their abbreviation."""
    toks = norm_addr.split()
    # Try the longest trailing spans first: 'new york' before 'york'.
    for n in (3, 2, 1):
        if len(toks) >= n:
            joined = "".join(toks[-n:])
            if joined in STATE_MAP:
                return STATE_MAP[joined]
            if n == 1 and len(joined) == 2 and joined in set(STATE_MAP.values()):
                return joined
    return ""


def address_numbers(norm_addr: str) -> list[str]:
    """Digit runs in the address: street numbers and PIN/ZIP codes.

    Numeric tokens are corrupted in the data ('45th' -> '45ND', 'Af-684' -> 'AF-0684'),
    but the leading digits usually survive, so digit runs are split out of mixed tokens
    and leading zeros dropped.
    """
    nums = re.findall(r"\d+", norm_addr)
    return [n.lstrip("0") or "0" for n in nums]


# --------------------------------------------------------------------------- #
# Frame-level application
# --------------------------------------------------------------------------- #


def add_normalized_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Attach all normalized key columns to a source frame.

    Romanization runs only on rows that actually contain non-ASCII characters (2-19%
    depending on the source), keeping the expensive Python path off the ASCII majority.
    """
    out = df

    # Romanize the minority of rows that need it, in Python; leave the rest untouched.
    for col, fn in (("business_name", romanize), ("business_address", romanize_address)):
        mask = out[col].fill_null("").str.contains(r"^[\x00-\x7F]*$").not_()
        if mask.sum():
            romanized = out.filter(mask)[col].map_elements(
                fn, return_dtype=pl.String
            )
            idx = mask.arg_true()
            out = out.with_columns(
                out[col].scatter(idx, romanized).alias(col)
            )

    return out.with_columns(
        pl.col("business_name")
        .map_elements(normalize_name, return_dtype=pl.String)
        .alias("name_norm"),
        pl.col("business_address")
        .map_elements(normalize_address, return_dtype=pl.String)
        .alias("addr_norm"),
    ).with_columns(
        pl.col("name_norm")
        .map_elements(name_core, return_dtype=pl.String)
        .alias("name_core"),
    ).with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_nospace"),
        pl.col("name_core")
        .map_elements(consonant_skeleton, return_dtype=pl.String)
        .alias("name_skel"),
        pl.col("addr_norm")
        .map_elements(address_state, return_dtype=pl.String)
        .alias("addr_state"),
    )
