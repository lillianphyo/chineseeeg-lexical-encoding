import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zuco_llm_align.chineseeeg import EYE_CHANNELS
from zuco_llm_align.chineseeeg_regression import CHAR_LAGS, layout, run_stats

_spec = importlib.util.spec_from_file_location(
    "chineseeeg_paper_assets", Path(__file__).parents[1] / "scripts" / "chineseeeg_paper_assets.py")
assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assets)

N_TARGETS = 128 - len(EYE_CHANNELS)
MODELS = ("N", "N+L", "N-strokes", "N+L+E", "N-eye")


def _results(rows):
    return pd.DataFrame([{"subject": s, "split": "locked", "model": m, "r": r, "alphas": "100 100 1000"}
                         for s, by_model in rows.items() for m, r in by_model.items()])


def test_subject_table_gives_gains_and_relative_sizes():
    t = assets.subject_table(_results({"sub-01": {"N": 0.4, "N+L": 0.41, "N-strokes": 0.39, "N+L+E": 0.405, "N-eye": 0.1}}))
    row = t.loc["sub-01"]
    assert row.delta == pytest.approx(0.01)
    assert row.strokes == pytest.approx(0.01)
    assert row.embedding == pytest.approx(-0.005)
    assert row.pct_n == pytest.approx(2.5)
    assert row.pct_neye == pytest.approx(10.0)


def test_subject_table_ignores_dev_rows():
    res = pd.concat([_results({"sub-01": dict.fromkeys(MODELS, 0.3)}),
                     _results({"sub-02": dict.fromkeys(MODELS, 0.3)}).assign(split="dev0")])
    assert list(assets.subject_table(res).index) == ["sub-01"]


def test_locked_runs_reads_the_test_runs_of_the_locked_split():
    res = pd.DataFrame([{"subject": "sub-01", "split": "dev0", "test_runs": "1"},
                        {"subject": "sub-01", "split": "locked", "test_runs": "5,10,103"}])
    assert assets.locked_runs(res, "sub-01") == ["5", "10", "103"]


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    lo, hi = assets.bootstrap_ci(values, n=500)
    assert lo < values.mean() < hi
    assert (lo, hi) == assets.bootstrap_ci(values, n=500)


def test_bootstrap_ci_of_a_constant_is_that_constant():
    assert assets.bootstrap_ci(pd.Series([2.0] * 4), n=50) == (2.0, 2.0)


def test_perm_z_uses_sample_sd():
    assert assets.perm_z(3.0, np.array([0.0, 2.0])) == pytest.approx(2.0 / np.sqrt(2))


def test_lags_ms_span_minus_100_to_800():
    ms = assets.lags_ms()
    assert len(ms) == len(CHAR_LAGS)
    assert ms[0] == pytest.approx(-93.75) and ms[-1] == pytest.approx(796.875)


def test_window_mean_averages_lags_and_channels_inside_only():
    ms = np.array([0.0, 100.0, 200.0, 300.0])
    k = np.zeros((2, 4, 3))
    k[:, 1:3, :2] = 1.0
    k[1, 1:3, 0] = 3.0
    np.testing.assert_allclose(assets.window_mean(k, ms, (100, 200), np.array([True, True, False])), [1.0, 2.0])


def test_window_mean_map_averages_subjects_and_lags():
    ms = np.array([0.0, 100.0])
    k = np.stack([np.ones((2, 3)), 3 * np.ones((2, 3))])
    np.testing.assert_allclose(assets.window_mean_map(k, ms, (0, 100)), [2.0, 2.0, 2.0])


def test_gfp_peak_ms_finds_the_largest_response():
    ms = np.array([0.0, 100.0, 200.0])
    k = np.zeros((2, 3, 4))
    k[:, 2] = -1.0
    k[:, 1] = 0.5
    assert assets.gfp_peak_ms(k, ms) == 200.0


@pytest.mark.parametrize(("x", "digits", "text"), [(5.1354e-5, 1, "5.1"), (1.63e-4, 1, "16.3"), (-2e-6, 2, "-0.20")])
def test_sci_scales_to_1e_minus_5(x, digits, text):
    assert assets.sci(x, digits=digits) == text


@pytest.mark.parametrize(("n", "text"), [(140, "140"), (22298, "22\\,298"), (2500000, "2\\,500\\,000")])
def test_thousands_separates_with_thin_spaces(n, text):
    assert assets.thousands(n) == text


def test_macros_write_one_newcommand_per_value():
    assert assets.macros({"a": "1", "bB": "x"}) == "\\newcommand{\\a}{1}\n\\newcommand{\\bB}{x}\n"


def test_table_tex_shortens_subject_and_scales_delta():
    t = pd.DataFrame({"r_n": [0.4], "r_neye": [0.1], "delta": [3e-5], "pct_n": [0.0075], "pct_neye": [0.03]},
                     index=["sub-07"])
    tex = assets.table_tex(t, {"sub-07": 4.21}, {"sub-07": 0.005})
    assert "\nS07 & 0.400 & 0.100 & 3.0 & 0.007 & 4.2 & 0.005 \\\\\n" in tex
    assert tex.startswith("\\begin{tabular}") and tex.endswith("\\bottomrule\n\\end{tabular}\n")


def _locked_dir(root: Path, subjects=("sub-01", "sub-02")):
    rng = np.random.default_rng(0)
    width = layout()["E"][-1] + 1
    deltas = {}
    for i, s in enumerate(subjects):
        groups = {}
        for g in range(5):
            x = rng.standard_normal((300, width))
            groups[f"g{g}"] = run_stats(x, x[:, :N_TARGETS] + rng.standard_normal((300, N_TARGETS)))
        np.savez(root / f"stats_{s}.npz", **{f"{g}_{k}": getattr(st, k) for g, st in groups.items()
                                             for k in ("G", "B", "sx", "sy", "syy", "n")})
        np.savez(root / f"kernels_{s}.npz", lexical=rng.standard_normal((2, len(CHAR_LAGS), N_TARGETS)),
                 features=np.array(["log_frequency", "surprisal"]), lags=np.arange(len(CHAR_LAGS)))
        pd.DataFrame({"subject": s, "perm": range(5), "delta_r": rng.standard_normal(5) * 1e-5}).to_csv(
            root / f"perms_{s}.csv", index=False)
        deltas[s] = (i + 1) * 1e-5
        (root / f"snapshot_{s}.json").write_text(json.dumps({"permutation": {"observed_delta_r": deltas[s], "p": 0.005},
                                                             "n_chars": {"1": 100, "5": 40 + i}, "condition": {"L": 2.0}}))
    rows = []
    for s in subjects:
        r = {"N": 0.3, "N+L": 0.3 + deltas[s], "N-strokes": 0.29, "N+L+E": 0.3 + deltas[s], "N-eye": 0.1}
        alphas = {"N": "100 100", "N+L": "100 100 1000", "N-strokes": "100 100", "N+L+E": "100 100 1000 1000", "N-eye": "100"}
        rows += [{"subject": s, "split": "locked", "model": m, "r": v, "alphas": alphas[m], "test_runs": "5"}
                 for m, v in r.items()]
    pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
    (root / "h1.txt").write_text("﻿n_subjects: 2\nlexical_p: 0.25\nstrokes_mean: 1e-5\nstrokes_positive: 2\n"
                                 "strokes_p: 0.25\nembedding_mean: 0.0\nembedding_positive: 0\nembedding_p: 1.0\n",
                                 encoding="utf-8")


def _nulls_dir(root: Path, subjects=("sub-01", "sub-02"), observed=None):
    rng = np.random.default_rng(1)
    for i, s in enumerate(subjects):
        obs = (observed or {}).get(s, (i + 1) * 1e-5)
        snap = {"nulls": {}}
        for kind, p in (("shuffle", 0.005), ("circular", 0.02 * (i + 1))):
            name = f"perms_{s}.csv" if kind == "shuffle" else f"perms_{kind}_{s}.csv"
            pd.DataFrame({"subject": s, "perm": range(5), "delta_r": -1e-5 + 1e-8 * rng.standard_normal(5),
                          "alpha_l": 1e5, "delta_r_reselected": 1e-7 * rng.standard_normal(5)}).to_csv(root / name, index=False)
            snap["nulls"][kind] = {"n": 5, "observed_delta_r": obs, "p": p, "p_reselected": p * 2}
        (root / f"snapshot_{s}.json").write_text(json.dumps(snap))


def test_null_table_reads_both_kinds_and_both_penalty_rules(tmp_path):
    _nulls_dir(tmp_path)
    n = assets.null_table(tmp_path, {"sub-01": 1e-5, "sub-02": 2e-5})
    assert list(n.index) == [("sub-01", "shuffle"), ("sub-01", "circular"), ("sub-02", "shuffle"), ("sub-02", "circular")]
    row = n.loc[("sub-02", "circular")]
    assert row.p == 0.04 and row.p_reselected == 0.08 and row.observed == 2e-5
    assert row.null_mean == pytest.approx(-1e-5, abs=2e-6)
    assert abs(row.null_mean_reselected) < 1e-6


def test_null_table_refuses_a_pass_whose_observed_gain_differs(tmp_path):
    _nulls_dir(tmp_path, observed={"sub-02": 3e-5})
    with pytest.raises(ValueError, match="sub-02 shuffle"):
        assets.null_table(tmp_path, {"sub-01": 1e-5, "sub-02": 2e-5})


def test_nulls_tex_has_one_row_per_reader_with_both_nulls(tmp_path):
    _nulls_dir(tmp_path)
    tex = assets.nulls_tex(assets.null_table(tmp_path, {"sub-01": 1e-5, "sub-02": 2e-5}))
    assert "Within-run shuffle" in tex and "Circular shift" in tex
    assert "\nS02 & 2.0 & -1.0 & 0.005 & 0.010 & -1.0 & 0.040 & 0.080 \\\\" in tex
    assert tex.count("\\\\\n") == 4 and tex.endswith("\\bottomrule\n\\end{tabular}\n")


def test_nulls_macros_count_floors_and_bound_the_reselected_means(tmp_path):
    _nulls_dir(tmp_path)
    m = assets.nulls_macros(assets.null_table(tmp_path, {"sub-01": 1e-5, "sub-02": 2e-5}))
    assert m["nullsShuffleFloor"] == "2" and m["nullsShuffleFloorRe"] == "0"
    assert m["nullsCircularFloor"] == "0" and m["nullsCircularFloorRe"] == "0"
    assert m["nullsFloorAllMin"] == "0" and m["nullsCircNeg"] == "2" and m["nullsMaxP"] == "0.08"
    assert m["nullsReNarrower"] == "0"
    assert float(m["nullsReMeanMin"]) <= float(m["nullsReMeanMax"]) and abs(float(m["nullsReMeanMax"])) < 0.1


def _gate_dir(root: Path, values: dict):
    root.mkdir()
    rows = [{"subject": s, "p2t_position_ge5_uV": {"row_pace": r, "fixed_0.35": f, "jittered": j}, "first_char_auc_mean": a}
            for s, (r, f, j, a) in values.items()]
    (root / "gate1.json").write_text(json.dumps(rows))


def test_gate_values_recompute_all_three_criteria(tmp_path):
    _gate_dir(tmp_path / "g", {"sub-01": (3.0, 0.4, 0.5, 0.8), "sub-02": (0.62, 0.1, 0.21, 0.8),
                               "sub-03": (2.0, 2.1, 0.1, 0.8), "sub-04": (3.0, 0.4, 0.5, 0.55)})
    g = assets.gate_values(tmp_path / "g")
    assert g.passes.to_dict() == {"sub-01": True, "sub-02": False, "sub-03": False, "sub-04": False}


def test_gate_tex_sets_selection_and_corrected_values_side_by_side(tmp_path):
    _gate_dir(tmp_path / "a", {"sub-01": (3.0, 0.4, 0.5, 0.8), "sub-13": (0.53, 0.17, 0.21, 0.67)})
    _gate_dir(tmp_path / "b", {"sub-01": (3.2, 0.4, 0.4, 0.8), "sub-13": (0.62, 0.17, 0.20, 0.67)})
    tex = assets.gate_tex(assets.gate_values(tmp_path / "a"), assets.gate_values(tmp_path / "b"))
    assert "\nS01 & 3.00 & 0.40 & 0.50 & 0.80 & yes & 3.20 & 0.40 & yes \\\\" in tex
    assert "\nS13 & 0.53 & 0.17 & 0.21 & 0.67 & no & 0.62 & 0.20 & yes \\\\" in tex
    assert "Span$/n$ onsets (selection)" in tex and tex.endswith("\\end{tabular}\n")


def test_main_writes_the_gate_table_when_both_gate_folders_are_present(tmp_path):
    d, out = tmp_path / "locked", tmp_path / "paper"
    d.mkdir()
    _locked_dir(d)
    for name in assets.GATE_DIRS:
        _gate_dir(tmp_path / name, {"sub-01": (3.0, 0.4, 0.5, 0.8)})
    assets.main(d, out)
    assert (out / "table_gate.tex").read_text().count("yes") == 2
    assert "\\newcommand{\\strokesOverLex}{" in (out / "numbers.tex").read_text()


def test_main_with_a_nulls_pass_writes_the_nulls_table_and_macros(tmp_path):
    d, nulls, out = tmp_path / "locked", tmp_path / "nulls", tmp_path / "paper"
    d.mkdir()
    nulls.mkdir()
    _locked_dir(d)
    _nulls_dir(nulls)
    assets.main(d, out, None, nulls)
    assert (out / "table_nulls.tex").read_text().startswith("\\begin{tabular}")
    numbers = (out / "numbers.tex").read_text()
    assert "\\newcommand{\\nullsShuffleFloor}{2}" in numbers and "\\newcommand{\\nullsMaxP}{0.08}" in numbers


def test_main_writes_numbers_table_and_figures(tmp_path):
    d, out = tmp_path / "locked", tmp_path / "paper"
    d.mkdir()
    _locked_dir(d)
    assets.main(d, out)
    numbers = (out / "numbers.tex").read_text()
    assert "\\newcommand{\\nSubj}{2}" in numbers
    assert "\\newcommand{\\dRmean}{1.5}" in numbers
    assert "\\newcommand{\\dRp}{0.2500}" in numbers
    assert "\\newcommand{\\permFloor}{2}" in numbers
    assert "\\newcommand{\\nullNeg}{" in numbers and "\\newcommand{\\nullMin}{" in numbers
    assert "\\newcommand{\\charsMin}{140}" in numbers and "\\newcommand{\\testCharsMax}{41}" in numbers
    assert "\nS01 & 0.300 & 0.100 & 1.0" in (out / "table_subjects.tex").read_text()
    for name in ("fig_delta.pdf", "fig_kernels.pdf", "fig_topo.pdf"):
        assert (out / name).stat().st_size > 0


def test_per_channel_delta_is_zero_when_models_match(tmp_path):
    d = tmp_path / "locked"
    d.mkdir()
    _locked_dir(d, ("sub-01",))
    groups = assets.load_stats(d / "stats_sub-01.npz")
    same = {"N": [100.0, 100.0], "N+L": [100.0, 100.0, 1e30]}
    np.testing.assert_allclose(assets.per_channel_delta(groups, same), 0.0, atol=1e-6)


def test_feature_deltas_are_zero_when_the_lexical_band_is_switched_off(tmp_path):
    d = tmp_path / "locked"
    d.mkdir()
    _locked_dir(d, ("sub-01",))
    groups = assets.load_stats(d / "stats_sub-01.npz")
    off = {"N": [100.0, 100.0], "N+L": [100.0, 100.0, 1e30]}
    deltas = assets.feature_deltas(groups, off)
    assert set(deltas) == {"log_frequency", "surprisal"}
    assert all(abs(v) < 1e-6 for v in deltas.values())


def test_feature_deltas_credit_the_feature_that_carries_the_signal():
    rng = np.random.default_rng(1)
    cols = layout()
    freq = cols["L"][:len(CHAR_LAGS)]
    width = cols["E"][-1] + 1
    groups = {}
    for g in range(5):
        x = rng.standard_normal((400, width))
        y = np.repeat(x[:, freq[:1]], N_TARGETS, axis=1) + 0.1 * rng.standard_normal((400, N_TARGETS))
        groups[f"g{g}"] = run_stats(x, y)
    deltas = assets.feature_deltas(groups, {"N": [1.0, 1.0], "N+L": [1.0, 1.0, 1.0]})
    assert deltas["log_frequency"] > 0.5
    assert deltas["log_frequency"] > deltas["surprisal"] + 0.4


def test_read_summary_collects_the_h1_items_of_one_read(tmp_path):
    d = tmp_path / "locked"
    d.mkdir()
    _locked_dir(d)
    s = assets.read_summary(d)
    assert s["n"] == 2 and s["positive"] == 2 and s["floor"] == 2
    assert s["mean"] == pytest.approx(1.5e-5)
    assert s["p"] == 0.25 and s["strokes_positive"] == 2 and s["embedding_p"] == 1.0


def test_reads_tex_puts_the_two_reads_side_by_side(tmp_path):
    d = tmp_path / "locked"
    d.mkdir()
    _locked_dir(d)
    first = assets.read_summary(d)
    second = dict(first, mean=2e-5, positive=1)
    tex = assets.reads_tex(first, second)
    assert "Mean $\\Delta R$" in tex and "& 1.5 & 2.0 \\\\" in tex
    assert "& 2/2 & 1/2 \\\\" in tex
    assert "Strokes $p$ & 0.250 & 0.250 \\\\" in tex and "Embedding readers $> 0$ & 0/2 & 0/2 \\\\" in tex
    assert ";" not in tex and ":" not in tex
    assert tex.startswith("\\begin{tabular}") and tex.endswith("\\end{tabular}\n")


def test_main_with_a_first_read_writes_the_reads_table_and_feature_macros(tmp_path):
    d, first, out = tmp_path / "locked", tmp_path / "first", tmp_path / "paper"
    d.mkdir()
    first.mkdir()
    _locked_dir(d)
    _locked_dir(first)
    assets.main(d, out, first)
    assert "& 1.5 & 1.5 \\\\" in (out / "table_reads.tex").read_text()
    numbers = (out / "numbers.tex").read_text()
    assert "\\newcommand{\\looFreqPos}{" in numbers and "\\newcommand{\\looSurpMean}{" in numbers
