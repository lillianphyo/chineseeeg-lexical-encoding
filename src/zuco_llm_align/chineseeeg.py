"""ChineseEEG (OpenNeuro ds004952) continuous EEG, character onsets and run roles from the authors' preprocessed derivative.

Only row onsets (ROWS) and offsets (ROWE) are marked. Each non-punctuation character is highlighted in turn by one
PsychoPy routine (`experiment/play_novel.py`) of 0.35 s plus about 0.08 s of stimulus rebuild, so character onsets
are spread over each row at that row's own pace (see char_onsets). Runs are split by their order (run_roles).
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

PUNCTUATION = frozenset(["\n", "。", "，", "！", "？", "：", "；", "“", "”", "、", "《", "》", ".", "（", "）", "…", "·"])
SFREQ = 64.0
PACE_RANGE = (0.30, 0.60)
SHIFT_TIME = 0.35
EYE_CHANNELS = (1, 8, 14, 17, 21, 25, 32, 125, 126, 127, 128)
GATE1_PASS = ("sub-04", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-14")
RUN_ORDER = [("GarnettDream", r) for r in range(1, 19)] + [("LittlePrince", r) for r in range(2, 8)]
PREPROC = "derivatives/preproc/filtered_0.5_30"


def highlighted(unit: str) -> list[str]:
    return [c for c in unit if c not in PUNCTUATION]


def run_id(session: str, run: int) -> int:
    return run + (100 if session == "LittlePrince" else 0)


def run_roles() -> dict[int, str]:
    """Fixed 3:1:1 train/val/test by position in the global run order; LittlePrince run 1 is left out."""
    return {run_id(s, r): ("train", "train", "train", "val", "test")[i % 5] for i, (s, r) in enumerate(RUN_ORDER)}


def locked_roles() -> dict[int, str]:
    """Locked-test read: train runs as before, test runs in the scored ("val") slot; validation runs left out."""
    return {k: "val" if v == "test" else v for k, v in run_roles().items() if v != "val"}


def char_onsets(events: pd.DataFrame, units: list[str], pace: float | None = None):
    """(onset s, character, row, position in row) per highlighted character, and the number of rows dropped.

    play_novel.py (eye-tracker loop) sends ROWS at the onset of the first highlighted character and ROWE after the
    last one's SHIFT_TIME display, so with `pace` None the n onsets of a row are spaced by
    (ROWE - ROWS - SHIFT_TIME) / (n - 1). A row whose span per character falls outside PACE_RANGE is dropped.
    A fixed `pace` is only for the timing check.
    """
    starts = events.loc[events.trial_type == "ROWS", "onset"].to_numpy()
    ends = events.loc[events.trial_type == "ROWE", "onset"].to_numpy()
    if not len(starts) == len(ends) == len(units):
        raise ValueError(f"{len(starts)} ROWS / {len(ends)} ROWE rows for {len(units)} text units")
    out, dropped = [], 0
    for row, (start, end, unit) in enumerate(zip(starts, ends, units)):
        chars = highlighted(unit)
        if not chars:
            continue
        row_pace = (end - start) / len(chars)
        if not PACE_RANGE[0] <= row_pace <= PACE_RANGE[1]:
            dropped += 1
            continue
        step = pace if pace is not None else (end - start - SHIFT_TIME) / (len(chars) - 1) if len(chars) > 1 else 0.0
        out += [(start + k * step, c, row, k) for k, c in enumerate(chars)]
    return out, dropped


def fixed_header(text: str, vhdr: Path) -> str:
    """The header with DataFile / MarkerFile pointing at the files beside it (GarnettDream headers say 'GranettDream')."""
    stem = vhdr.with_suffix("")
    return re.sub(r"^(DataFile|MarkerFile)=.*$", lambda m: f"{m[1]}={stem}.{'eeg' if m[1] == 'DataFile' else 'vmrk'}",
                  text, flags=re.M)


def read_raw(path) -> mne.io.BaseRaw:
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        header = Path(tmp) / path.name
        header.write_text(fixed_header(path.read_text(encoding="utf-8"), path.resolve()), encoding="utf-8")
        return mne.io.read_raw_brainvision(header, preload=True, verbose=False)


def epoch_raw(raw: mne.io.BaseRaw, onsets_s: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """(n, channels, samples) µV at SFREQ, samples `window` relative to each onset; NaN where off the recording."""
    data = raw.copy().resample(SFREQ, verbose=False).get_data() * 1e6
    starts = np.round(np.asarray(onsets_s) * SFREQ).astype(int) + window[0]
    width = window[1] - window[0]
    out = np.full((len(starts), data.shape[0], width), np.nan, dtype=np.float32)
    inside = (starts >= 0) & (starts + width <= data.shape[1])
    for i in np.flatnonzero(inside):
        out[i] = data[:, starts[i]:starts[i] + width]
    return out


def runs_of(root: Path, subject: str):
    for session, run in RUN_ORDER:
        stem = root / PREPROC / subject / f"ses-{session}" / "eeg" / f"{subject}_ses-{session}_task-reading_run-{run:02d}"
        if Path(f"{stem}_eeg.vhdr").exists():
            yield session, run, stem


def units_of(root, session: str, run: int) -> list[str]:
    path = Path(root) / f"derivatives/novels/segmented_novel/{session}/segmented_Chinense_novel_run_{run}.xlsx"
    return pd.read_excel(path).Chinese_text.astype(str).tolist()


def load_run(root, stem: Path, session: str, run: int, window: tuple[int, int], pace: float | None = None):
    """Character epochs of one run and their (character, row, position, per-character ms); rows off the recording dropped."""
    events = pd.read_csv(f"{stem}_events.tsv", sep="\t")
    onsets, _ = char_onsets(events, units_of(root, session, run), pace)
    eeg = epoch_raw(read_raw(f"{stem}_eeg.vhdr"), np.array([o for o, *_ in onsets]), window)
    starts = events.loc[events.trial_type == "ROWS", "onset"].to_numpy()
    ends = events.loc[events.trial_type == "ROWE", "onset"].to_numpy()
    n = {}
    for _, _, row, k in onsets:
        n[row] = k + 1
    meta = [(c, row, k, 1000 * (ends[row] - starts[row]) / n[row]) for _, c, row, k in onsets]
    keep = ~np.isnan(eeg).any(axis=(1, 2))
    return eeg[keep], [m for m, ok in zip(meta, keep) if ok]




def load_char_table(path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Character -> static bert-base-chinese token embedding, and character -> Unihan stroke count (-1 unknown)."""
    t = np.load(path)
    chars = [str(c) for c in t["chars"]]
    return dict(zip(chars, t["vectors"].astype(np.float32))), dict(zip(chars, t["strokes"].astype(int).tolist()))
