import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("chineseeeg_timing_check", Path(__file__).parents[1] / "scripts" / "chineseeeg_timing_check.py")
gate1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate1)


def _erp(peak_ms):
    erp = np.zeros(gate1.ERP_WINDOW[1] - gate1.ERP_WINDOW[0])
    erp[gate1.ONSET + round(peak_ms / 1000 * gate1.SFREQ)] = 3.0
    erp[gate1.ONSET + 20] = -1.0
    return erp


def test_peak_to_trough_and_visual_peak_read_the_early_window():
    erp = _erp(125)
    assert gate1.peak_to_trough(erp) == 4.0
    assert gate1.visual_peak_ms(erp) == 125.0


def test_verdict_passes_only_when_all_three_checks_hold():
    good = {"visual_peak_ms": 125.0, "first_char_auc_mean": 0.7,
            "p2t_position_ge5_uV": {"row_pace": 2.0, "fixed_0.35": 1.0, "jittered": 0.5}}
    assert all(gate1.verdict(good).values())
    bad = {**good, "p2t_position_ge5_uV": {"row_pace": 1.0, "fixed_0.35": 2.0, "jittered": 0.5}}
    assert not gate1.verdict(bad)["2_row_pace_beats_fixed_and_jitter"]
    weak = {**good, "p2t_position_ge5_uV": {"row_pace": 1.0, "fixed_0.35": 0.5, "jittered": 0.5}}
    assert not gate1.verdict(weak)["1_response_3x_jitter_floor"]
