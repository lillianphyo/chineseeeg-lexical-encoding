import mne
import numpy as np
import pandas as pd
import pytest

from zuco_llm_align import chineseeeg
from zuco_llm_align.chineseeeg import (
    SFREQ,
    SHIFT_TIME,
    char_onsets,
    EYE_CHANNELS,
    epoch_raw,
    fixed_header,
    highlighted,
    load_char_table,
    locked_roles,
    run_id,
    run_roles,
)

UNITS = ["全世界的狼", "同的习性，在", "集合。"]


def _events(rows, overhead=0.08, gap=0.06, start=2.0):
    """ROWS at the first character's onset, ROWE after the last character's 0.35 s display; each character
    costs SHIFT_TIME + the stimulus rebuild `overhead` (play_novel.py, eye-tracker loop)."""
    out, t = [], start
    for n in rows:
        length = n * SHIFT_TIME + (n - 1) * overhead
        out += [("ROWS", t), ("ROWE", t + length)]
        t += length + gap
    return pd.DataFrame({"onset": [o for _, o in out], "duration": 0.0, "trial_type": [k for k, _ in out]})


def test_highlighted_drops_the_presentation_codes_punctuation_only():
    assert highlighted("同的习性，在") == ["同", "的", "习", "性", "在"]
    assert highlighted("“1*—”。") == ["1", "*", "—"]


def test_char_onsets_step_is_the_row_span_minus_the_last_display_over_n_minus_1():
    ev = _events([5, 5, 2], overhead=0.08)
    onsets, dropped = char_onsets(ev, UNITS)
    assert dropped == 0
    assert [c for _, c, _, _ in onsets] == list("全世界的狼同的习性在集合")
    assert [k for _, _, _, k in onsets[:6]] == [0, 1, 2, 3, 4, 0]
    assert onsets[0][0] == pytest.approx(2.0)
    assert onsets[1][0] == pytest.approx(2.0 + 0.43)
    assert onsets[4][0] == pytest.approx(2.0 + 4 * 0.43)
    assert onsets[5][0] == pytest.approx(2.0 + 5 * SHIFT_TIME + 4 * 0.08 + 0.06)
    assert onsets[11][0] - onsets[10][0] == pytest.approx(0.43)


def test_char_onsets_of_a_single_character_row_is_the_row_start():
    ev = _events([1, 5, 2])
    onsets, _ = char_onsets(ev, ["狼", "同的习性，在", "集合。"])
    assert onsets[0] == (pytest.approx(2.0), "狼", 0, 0)


def test_char_onsets_at_a_fixed_pace_drift_from_the_row_pace():
    ev = _events([5, 5, 2])
    fixed, _ = char_onsets(ev, UNITS, pace=0.35)
    assert fixed[4][0] == pytest.approx(2.0 + 4 * 0.35)


def test_char_onsets_drop_rows_whose_pace_is_implausible():
    ev = _events([5, 5, 2])
    ev.loc[3, "onset"] += 3.0
    onsets, dropped = char_onsets(ev, UNITS)
    assert dropped == 1 and {r for _, _, r, _ in onsets} == {0, 2}


def test_char_onsets_reject_a_run_whose_rows_do_not_match_the_text():
    with pytest.raises(ValueError, match="rows"):
        char_onsets(_events([5, 5]), UNITS)


def test_run_roles_follow_the_global_run_order_and_skip_the_excluded_run():
    roles = run_roles()
    assert run_id("LittlePrince", 1) not in roles
    assert [roles[run_id("GarnettDream", r)] for r in range(1, 6)] == ["train", "train", "train", "val", "test"]
    assert roles[run_id("LittlePrince", 2)] == ("train", "train", "train", "val", "test")[18 % 5]
    assert len(roles) == 24


def _raw(seconds=20.0, n_ch=3, sfreq=256.0, offset=0.0):
    t = np.arange(int(seconds * sfreq)) / sfreq
    data = np.stack([t + 100 * c + offset for c in range(n_ch)]) * 1e-6
    return mne.io.RawArray(data, mne.create_info([f"E{c + 1}" for c in range(n_ch)], sfreq, "eeg"), verbose=False)


def test_epoch_raw_resamples_and_cuts_windows_in_microvolts():
    raw = _raw()
    eeg = epoch_raw(raw, np.array([2.0, 5.0]), (0, 8))
    assert eeg.shape == (2, 3, 8) and eeg.dtype == np.float32
    assert eeg[0, 0, 0] == pytest.approx(2.0, abs=0.05)
    assert eeg[1, 2, 1] == pytest.approx(200 + 5.0 + 1 / SFREQ, abs=0.05)


def test_epoch_raw_marks_windows_past_the_recording_as_missing():
    eeg = epoch_raw(_raw(seconds=10.0), np.array([-0.1, 9.9]), (0, 8))
    assert np.isnan(eeg).all(axis=(1, 2)).tolist() == [True, True]


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    for ses, run in (("GarnettDream", 1), ("GarnettDream", 4)):
        text = root / f"derivatives/novels/segmented_novel/{ses}"
        text.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Chinese_text": UNITS}).to_excel(text / f"segmented_Chinense_novel_run_{run}.xlsx", index=False)
        eeg = root / f"derivatives/preproc/filtered_0.5_30/sub-04/ses-{ses}/eeg"
        eeg.mkdir(parents=True, exist_ok=True)
        stem = f"sub-04_ses-{ses}_task-reading_run-{run:02d}"
        _events([5, 5, 2]).to_csv(eeg / f"{stem}_events.tsv", sep="\t", index=False)
        (eeg / f"{stem}_eeg.vhdr").write_text("")
    monkeypatch.setattr(chineseeeg, "read_raw", lambda path: _raw(offset=1000.0 * ("run-04" in str(path))))
    return root


def test_runs_of_yields_the_recorded_runs_in_global_order(dataset):
    runs = list(chineseeeg.runs_of(dataset, "sub-04"))
    assert [(s, r) for s, r, _ in runs] == [("GarnettDream", 1), ("GarnettDream", 4)]
    assert runs[1][2].name == "sub-04_ses-GarnettDream_task-reading_run-04"


def test_load_run_returns_one_epoch_per_highlighted_character_with_row_and_position(dataset):
    _, _, stem = list(chineseeeg.runs_of(dataset, "sub-04"))[1]
    eeg, meta = chineseeeg.load_run(dataset, stem, "GarnettDream", 4, (0, 8))
    chars = [c for u in UNITS for c in highlighted(u)]
    assert eeg.shape == (len(chars), 3, 8)
    assert [m[0] for m in meta] == chars
    assert [m[1] for m in meta] == [0] * 5 + [1] * 5 + [2] * 2
    assert [m[2] for m in meta] == list(range(5)) + list(range(5)) + [0, 1]
    assert eeg[0, 0, 0] > 1000


def test_load_run_drops_characters_whose_window_leaves_the_recording(dataset, monkeypatch):
    monkeypatch.setattr(chineseeeg, "read_raw", lambda path: _raw(seconds=4.0))
    _, _, stem = next(chineseeeg.runs_of(dataset, "sub-04"))
    eeg, meta = chineseeeg.load_run(dataset, stem, "GarnettDream", 1, (0, 8))
    assert 0 < len(meta) < 12 and len(eeg) == len(meta)


def test_fixed_header_points_data_and_marker_files_at_the_files_beside_it(tmp_path):
    vhdr = tmp_path / "sub-04_ses-GarnettDream_task-reading_run-01_eeg.vhdr"
    text = "[Common Infos]\nDataFile=sub-04_ses-GranettDream_task-reading_run-01_eeg.eeg\nMarkerFile=x.vmrk\nDataFormat=BINARY\n"
    out = fixed_header(text, vhdr)
    assert f"DataFile={tmp_path / 'sub-04_ses-GarnettDream_task-reading_run-01_eeg.eeg'}" in out
    assert f"MarkerFile={tmp_path / 'sub-04_ses-GarnettDream_task-reading_run-01_eeg.vmrk'}" in out
    assert "DataFormat=BINARY" in out


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "char_table.npz"
    np.savez(path, chars=np.array(["全", "世", "界"]), vectors=np.eye(3, 4, dtype=np.float32), strokes=np.array([6, 5, -1]))
    return path


def test_load_char_table_maps_characters_to_vectors_and_strokes(table):
    vectors, strokes = load_char_table(table)
    assert vectors["世"].tolist() == [0, 1, 0, 0] and strokes == {"全": 6, "世": 5, "界": -1}







def test_locked_roles_score_the_test_runs_and_never_load_validation_runs():
    dev, locked = run_roles(), locked_roles()
    assert {k for k, v in locked.items() if v == "val"} == {k for k, v in dev.items() if v == "test"}
    assert {k for k, v in locked.items() if v == "train"} == {k for k, v in dev.items() if v == "train"}
    assert not {k for k, v in dev.items() if v == "val"} & set(locked)
    assert {run_id("GarnettDream", 5), run_id("GarnettDream", 10), run_id("GarnettDream", 15), run_id("LittlePrince", 3)} == {
        k for k, v in locked.items() if v == "val"}
