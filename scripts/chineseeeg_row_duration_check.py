"""Verify the character step from the row markers alone (no EEG).

    python scripts/chineseeeg_row_duration_check.py --root ~/datasets/ds004952 [--out runs/row_duration_check.txt]

play_novel.py (eye-tracker loop) displays each highlighted character for SHIFT_TIME after rebuilding the text
stimuli, sends ROWS at the first character's onset and ROWE after the last character's display. That predicts
row duration = n * SHIFT_TIME + (n - 1) * overhead: slope SHIFT_TIME + overhead, intercept -overhead. The
non-slip loop without the eye tracker would give slope SHIFT_TIME and intercept 0. The corrected step inside a
row is (ROWE - ROWS - SHIFT_TIME) / (n - 1); the old span / n step is shown beside it.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from zuco_llm_align.chineseeeg import SHIFT_TIME, highlighted, runs_of, units_of

MAX_ROW_S = 20.0


def row_durations(events: pd.DataFrame, units: list[str]) -> list[tuple[int, float]]:
    """(highlighted characters, ROWE - ROWS) per row; empty when the markers do not match the text."""
    starts = events.loc[events.trial_type == "ROWS", "onset"].to_numpy()
    ends = events.loc[events.trial_type == "ROWE", "onset"].to_numpy()
    if not len(starts) == len(ends) == len(units):
        return []
    return [(len(highlighted(u)), float(e - s)) for s, e, u in zip(starts, ends, units) if highlighted(u)]


def fit_row_model(rows: pd.DataFrame) -> dict:
    a = np.c_[rows.n, np.ones(len(rows))]
    (slope, intercept), *_ = np.linalg.lstsq(a, rows.dur, rcond=None)
    residual = rows.dur - a @ (slope, intercept)
    return {"slope": float(slope), "intercept": float(intercept), "residual_sd": float(residual.std()),
            "overhead_from_slope": float(slope - SHIFT_TIME), "overhead_from_intercept": float(-intercept), "rows": int(len(rows))}


def per_n_table(rows: pd.DataFrame) -> pd.DataFrame:
    multi = rows[rows.n > 1]
    g = multi.groupby("n").dur
    return pd.DataFrame({"rows": g.size(), "mean_duration": g.mean(), "sd": g.std(),
                         "span_over_n": g.mean() / g.mean().index,
                         "corrected_step": ((multi.dur - SHIFT_TIME) / (multi.n - 1)).groupby(multi.n).mean()})


def collect(root: Path) -> pd.DataFrame:
    out = []
    for subject in sorted(p.name for p in (root / "derivatives/preproc/filtered_0.5_30").glob("sub-*")):
        for session, run, stem in runs_of(root, subject):
            events = pd.read_csv(f"{stem}_events.tsv", sep="\t")
            out += [(subject, session, run, n, d) for n, d in row_durations(events, units_of(root, session, run))]
    rows = pd.DataFrame(out, columns=["subject", "session", "run", "n", "dur"])
    return rows[rows.dur < MAX_ROW_S]


def report(rows: pd.DataFrame) -> str:
    fit = fit_row_model(rows)
    lines = [f"{fit['rows']} rows from {rows.subject.nunique()} subjects; n = {rows.n.min()}-{rows.n.max()} characters",
             f"duration = {fit['slope']:.4f} * n + ({fit['intercept']:+.4f}) s, residual SD {fit['residual_sd'] * 1000:.0f} ms",
             f"overhead: {fit['overhead_from_slope']:.4f} s from the slope, {fit['overhead_from_intercept']:.4f} s from the intercept",
             f"eye-tracker loop predicts slope {SHIFT_TIME} + overhead and intercept -overhead; non-slip loop predicts slope {SHIFT_TIME}, intercept 0"]
    for subject, g in rows.groupby("subject"):
        f = fit_row_model(g)
        lines.append(f"  {subject}: slope {f['slope']:.4f} intercept {f['intercept']:+.4f} rows {f['rows']}")
    lines.append(per_n_table(rows).round(4).to_string())
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=os.environ.get("CHINESEEEG_ROOT", str(Path.home() / "datasets/ds004952")))
    p.add_argument("--out", default="runs/row_duration_check.txt")
    args = p.parse_args()
    text = report(collect(Path(args.root)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
