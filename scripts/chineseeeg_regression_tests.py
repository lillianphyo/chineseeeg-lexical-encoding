"""Pre-specified test of lab01/.claude/prds/exp-chineseeeg-overlap-regression-2026-10.md, fixed before any EEG fit.

    python scripts/chineseeeg_regression_tests.py runs/chineseeeg_regression_locked/results.csv

delta_r(L) = r(N+L) - r(N) on the locked-test runs, per subject.
H1: delta_r(L) > 0, one-sided Wilcoxon over the 7 Gate-1 subjects, alpha 0.05.
Secondary (no claim): strokes = r(N) - r(N-strokes); embedding = r(N+L+E) - r(N+L).
Amendment before any dev or locked run (side-agent review): a band whose chosen penalty is the grid maximum is
switched off, so its delta is set to exactly 0; zeros count half for and half against (zero_method="zsplit").
"""
import sys

import pandas as pd
from scipy.stats import wilcoxon

from zuco_llm_align.chineseeeg import GATE1_PASS
from zuco_llm_align.chineseeeg_regression import GRID

ALPHA = 0.05


def deltas(results: pd.DataFrame, split: str = "locked") -> pd.DataFrame:
    rows = results[results.split == split]
    r = rows.pivot_table(index="subject", columns="model", values="r")
    alphas = rows.set_index(["subject", "model"]).alphas.str.split().map(lambda a: [float(x) for x in a])

    def off(model):
        return alphas.xs(model, level="model").map(lambda a: a[-1] >= GRID[-1])

    return pd.DataFrame({"lexical": (r["N+L"] - r["N"]).mask(off("N+L"), 0.0), "strokes": r["N"] - r["N-strokes"],
                         "embedding": (r["N+L+E"] - r["N+L"]).mask(off("N+L+E"), 0.0)})


def tests(results: pd.DataFrame) -> dict:
    d = deltas(results)
    missing = sorted(set(GATE1_PASS) - set(d.index))
    if missing:
        raise ValueError(f"subjects missing from the locked results: {missing}")
    out = {"n_subjects": len(d)}
    for name, v in d.items():
        p = float(wilcoxon(v, alternative="greater", zero_method="zsplit").pvalue)
        out.update({f"{name}_mean": float(v.mean()), f"{name}_positive": int((v > 0).sum()), f"{name}_zero": int((v == 0).sum()), f"{name}_p": p})
    out["h1_pass"] = out["lexical_p"] < ALPHA
    out["per_subject"] = d.round(5)
    return out


if __name__ == "__main__":
    result = tests(pd.read_csv(sys.argv[1]))
    print(result.pop("per_subject").to_string())
    for k, v in result.items():
        print(f"{k}: {v}")
