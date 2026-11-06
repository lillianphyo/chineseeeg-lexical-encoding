"""Gate 1 of lab01/.claude/prds/exp-chineseeeg-char-decoding-2026-10.md: are the interpolated character onsets right?

    python scripts/chineseeeg_timing_check.py --subjects sub-04,sub-05 --out runs/chineseeeg_gate1

Checks 1 and 2 use only characters at row position >= 5, whose onsets are interpolated, never the marked first one.
1. The 0-0.4 s peak-to-trough of the occipital ERP from row-pace onsets is over 3x the jittered-onset floor
   (onsets moved uniformly by +-half a character).
2. It is larger than with fixed 0.35 s onsets and than the jittered floor.
3. Data sanity, not a timing test (the first character is marked exactly): the first character of a row is told apart from the rest from 0-0.4 s EEG,
   cross-validated by run (AUC). Only train and validation runs are read.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from zuco_llm_align.chineseeeg import SFREQ, load_run, run_id, run_roles, runs_of

OCCIPITAL = ("E70", "E71", "E74", "E75", "E76", "E82", "E83")
ERP_WINDOW = (-13, 64)
ONSET = -ERP_WINDOW[0]
EARLY = (ONSET, ONSET + 26)
VISUAL = (ONSET + 5, ONSET + 13)
LATE_POSITION = 5


def peak_to_trough(erp: np.ndarray, span: tuple[int, int] = EARLY) -> float:
    seg = erp[span[0]:span[1]]
    return float(seg.max() - seg.min())


def visual_peak_ms(erp: np.ndarray) -> float:
    seg = np.abs(erp[EARLY[0]:EARLY[1]] - erp[EARLY[0]:EARLY[1]].mean())
    return 1000 * float(seg.argmax()) / SFREQ


def subject_check(root, subject: str, max_runs: int | None, seed: int = 0) -> dict:
    roles, rng = run_roles(), np.random.default_rng(seed)
    erps = {"row_pace": [], "fixed_0.35": [], "jittered": []}
    late = {k: [] for k in erps}
    features, labels, groups = [], [], []
    picks = None
    runs = [(s, r, stem) for s, r, stem in runs_of(Path(root), subject)
            if roles.get(run_id(s, r)) in {"train", "val"} and Path(f"{stem}_eeg.eeg").exists()][:max_runs]
    for session, run, stem in runs:
        eeg, meta = load_run(root, stem, session, run, ERP_WINDOW)
        fixed, fixed_meta = load_run(root, stem, session, run, ERP_WINDOW, pace=0.35)
        if picks is None:
            names = pd.read_csv(f"{stem}_channels.tsv", sep="\t").name.tolist()
            picks = [names.index(c) for c in OCCIPITAL if c in names]
        k = np.array([m[2] for m in meta])
        shift = np.round(rng.uniform(-0.5, 0.5, len(meta)) * np.array([m[3] for m in meta]) / 1000 * SFREQ).astype(int)
        occ = eeg[:, picks].mean(axis=1)
        jit = np.stack([np.roll(o, -s) for o, s in zip(occ, shift)])
        k_fixed = np.array([m[2] for m in fixed_meta])
        for name, x, kk in (("row_pace", occ, k), ("fixed_0.35", fixed[:, picks].mean(axis=1), k_fixed), ("jittered", jit, k)):
            erps[name].append(x)
            late[name].append(x[kk >= LATE_POSITION])
        features.append(eeg[:, :, EARLY[0]:EARLY[1]:2].reshape(len(eeg), -1))
        labels.append(k == 0)
        groups.append(np.full(len(eeg), run_id(session, run)))
        print(f"  {subject} {session} run {run}: {len(eeg)} characters", flush=True)
    erp = {n: np.concatenate(v).mean(axis=0) for n, v in erps.items()}
    late_p2t = {n: peak_to_trough(np.concatenate(v).mean(axis=0)) for n, v in late.items()}
    x, y, g = np.concatenate(features), np.concatenate(labels), np.concatenate(groups)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.01, class_weight="balanced"))
    auc = cross_val_score(model, x, y, groups=g, cv=GroupKFold(n_splits=min(5, len(np.unique(g)))), scoring="roc_auc")
    return {
        "subject": subject, "runs": [f"{s}-{r}" for s, r, _ in runs], "n_characters": int(len(y)),
        "visual_peak_ms": visual_peak_ms(erp["row_pace"]),
        "p2t_all_uV": {n: peak_to_trough(e) for n, e in erp.items()}, "p2t_position_ge5_uV": late_p2t,
        "first_char_auc_mean": float(auc.mean()), "first_char_auc_folds": auc.round(3).tolist(),
        "erp": {n: e.round(4).tolist() for n, e in erp.items()},
    }


def verdict(r: dict) -> dict:
    """All three must hold for the subject to enter decoding; 1 and 2 use only interpolated characters (position >= 5)."""
    late = r["p2t_position_ge5_uV"]
    return {
        "1_response_3x_jitter_floor": late["row_pace"] > 3 * late["jittered"],
        "2_row_pace_beats_fixed_and_jitter": late["row_pace"] > late["fixed_0.35"] and late["row_pace"] > late["jittered"],
        "3_first_char_auc_above_0.55": r["first_char_auc_mean"] > 0.55,
    }


def plot(results: list[dict], out: Path) -> None:
    t = 1000 * (np.arange(*ERP_WINDOW)) / SFREQ
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.5), squeeze=False)
    for ax, r in zip(axes[0], results):
        for name, style in (("row_pace", "-"), ("fixed_0.35", "--"), ("jittered", ":")):
            ax.plot(t, r["erp"][name], style, label=name)
        ax.axvline(0, color="k", lw=0.5)
        ax.set(title=f"{r['subject']} ({r['n_characters']} chars)", xlabel="ms from character onset", ylabel="occipital µV")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=os.environ.get("CHINESEEEG_ROOT", str(Path.home() / "datasets/ds004952")))
    p.add_argument("--subjects", default="sub-04,sub-05")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--out", default="runs/chineseeeg_gate1")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = [subject_check(args.root, s, args.max_runs) for s in args.subjects.split(",")]
    for r in results:
        r["verdict"] = verdict(r)
        print(json.dumps({k: v for k, v in r.items() if k != "erp"}, indent=2), flush=True)
    (out / "gate1.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot(results, out / "gate1_erp.png")


if __name__ == "__main__":
    main()
