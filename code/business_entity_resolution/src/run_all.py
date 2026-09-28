"""End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

Each step is a separate script, so that a step can be re-run on its own. This file only
runs them in order, with any per-step environment settings, and stops at the first
failure. The cross-encoder steps need a CUDA GPU (developed on a 6 GB RTX 4050 laptop GPU).

Usage:  python run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

# (command, extra environment)
STEPS = [
    (["prepare_data.py"], {}),                 # TSV -> Parquet, compact integer ids
    (["indic_dict.py"], {}),                   # Indic-script -> Latin token map from train GT
    (["normalize_all.py"], {}),                # normalized name / address keys, cached
    (["blocking.py", "train"], {}),            # candidate generation (train, for fitting)
    (["blocking.py", "test"], {}),
    (["stage_a.py", "features", "train"], {}), # cheap features over the full union
    (["stage_a.py", "features", "test"], {}),
    (["stage_a.py", "fit"], {}),               # 2-fold stage A ranker
    (["stage_a.py", "score", "train"], {}),    # out-of-fold scores + cut to top N
    (["stage_a.py", "score", "test"], {}),     # mean of fold models + cut (final candidates)
    (["stage_b.py", "features", "train"], {}), # full pair features on the cut set
    (["stage_b.py", "features", "test"], {}),
    (["stage_b.py", "fit"], {}),               # 2-fold stage B scorer
    (["stage_b.py", "score", "train"], {}),    # out-of-fold scores (support for stage C)
    (["stage_b.py", "score", "test"], {}),     # writes candidate_pairs.tsv (the cut set)
    (["stage_c.py", "features", "train"], {}), # collective evidence from each entity's records
    (["stage_c.py", "features", "test"], {}),
    (["stage_c.py", "fit"], {}),               # 2-fold stage C scorer (seed 0)
    (["stage_c.py", "score", "train"], {}),    # out-of-fold validation report
    (["stage_c.py", "score", "test"], {}),     # stage-C-only matching_results.tsv (v7)
    # two more stage C models for the bag (different seed and feature fraction)
    (["-c", "import stage_c as c; c.fit(); c.score('train'); c.score('test')"],
     {"ER_MODEL_TAG": "_s1", "ER_SEED": "1", "ER_FF": "0.7"}),
    (["-c", "import stage_c as c; c.fit(); c.score('train'); c.score('test')"],
     {"ER_MODEL_TAG": "_s2", "ER_SEED": "2", "ER_FF": "0.9"}),
    # cross-encoder on the uncertain band (GPU)
    (["xenc.py", "prep"], {}),                 # text pairs: train halves A/B, test
    (["xenc.py", "train"], {"XE_HALF": "A"}),  # model A, fine-tuned on half A
    (["xenc.py", "predict"], {"XE_HALF": "A"}),  # scores half B (held out) and test
    (["xenc.py", "train"], {"XE_HALF": "B"}),  # model B, fine-tuned on half B
    (["xenc.py", "predict"], {"XE_HALF": "B", "XE_TEST_ONLY": "1"}),  # scores test
    (["final_blend.py"], {}),                  # bag + cross-encoder blend -> matching_results.tsv
]


def main() -> int:
    t0 = time.time()
    for args, extra in STEPS:
        label = " ".join(args) + (f"  {extra}" if extra else "")
        print(f"\n=== {label}  ({time.time() - t0:.0f}s) ===", flush=True)
        env = {**os.environ, **extra, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        rc = subprocess.call([sys.executable, *args], env=env)
        if rc:
            print(f"step failed with exit code {rc}: {label}")
            return rc
    print(f"\npipeline complete ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
