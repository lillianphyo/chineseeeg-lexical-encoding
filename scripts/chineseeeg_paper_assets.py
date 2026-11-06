"""Numbers, tables and figures for the ICSIE 2027 lexical-encoding paper, all read from the locked regression run.

    python scripts/chineseeeg_paper_assets.py runs/chineseeeg_regression_locked_corrected paper \
        [runs/locked_first_read_old_onsets] [runs/chineseeeg_regression_locked_nulls]

The optional third path is the first locked read; with it, table_reads.tex sets both reads side by side.
The optional fourth path is the post hoc surrogate pass (shuffle and circular nulls, fixed and re-selected lexical
penalty); with it, table_nulls.tex and the nulls* macros are written, after checking that its observed gains equal
those of the locked read.
The per-feature gains (looFreq*, looSurp* macros) are post hoc: one lexical feature at a time with the N+L penalties.
table_gate.tex is written when the two timing-gate folders sit next to the locked read (GATE_DIRS): the values that
selected the readers (span / n onsets) beside the re-run with the corrected onsets, verdicts recomputed with the
criteria of the paper.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from zuco_llm_align.chineseeeg import EYE_CHANNELS, SFREQ
from zuco_llm_align.chineseeeg_regression import CHAR_LAGS, RunStats, fit, layout, pearson, total

FEATURES = ("log_frequency", "surprisal")
NULL_KINDS = ("shuffle", "circular")
P_FLOOR = 0.005
TRAIN_GROUPS, TEST_GROUP = ("g0", "g1", "g2", "g3"), "g4"
N400 = (300, 500)
GATE_DIRS = ("gate1_first_old_onsets", "gate1_corrected")
GATE_RATIO, GATE_AUC = 3.0, 0.55
ACCENT, NEUTRAL, INK, MUTED, LIGHT = "#2a5db0", "#8a8f98", "#1f2328", "#6e7781", "#9cc0ec"


def subject_table(res: pd.DataFrame) -> pd.DataFrame:
    r = res[res.split == "locked"].pivot_table(index="subject", columns="model", values="r")
    out = pd.DataFrame({"r_n": r["N"], "r_neye": r["N-eye"], "delta": r["N+L"] - r["N"],
                        "strokes": r["N"] - r["N-strokes"], "embedding": r["N+L+E"] - r["N+L"]})
    out["pct_n"] = 100 * out.delta / out.r_n
    out["pct_neye"] = 100 * out.delta / out.r_neye
    return out


def locked_runs(res: pd.DataFrame, subject: str) -> list[str]:
    row = res[(res.split == "locked") & (res.subject == subject)].iloc[0]
    return str(row.test_runs).split(",")


def bootstrap_ci(values: pd.Series, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = [values.sample(len(values), replace=True, random_state=rng.integers(1 << 31)).mean() for _ in range(n)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def perm_z(observed: float, null: np.ndarray) -> float:
    return float((observed - null.mean()) / null.std(ddof=1))


def lags_ms() -> np.ndarray:
    return CHAR_LAGS / SFREQ * 1000


def window_mean(kernels: np.ndarray, ms: np.ndarray, window: tuple[float, float], channels: np.ndarray) -> np.ndarray:
    """Per-subject mean of (subject, lag, channel) kernels over a latency window and a channel mask."""
    inside = (ms >= window[0]) & (ms <= window[1])
    return kernels[:, inside][:, :, channels].mean(axis=(1, 2))


def gfp_peak_ms(kernels: np.ndarray, ms: np.ndarray) -> float:
    return float(ms[int(np.argmax(np.sqrt((kernels ** 2).mean(axis=2)).mean(axis=0)))])


def load_stats(path: Path) -> dict[str, RunStats]:
    z = np.load(path)
    groups = sorted({k.split("_")[0] for k in z})
    return {g: RunStats(*(z[f"{g}_{k}"].astype(np.float64) for k in ("G", "B", "sx", "sy", "syy")), int(z[f"{g}_n"]))
            for g in groups}


def model_r(groups: dict[str, RunStats], cols: list[np.ndarray], alphas) -> np.ndarray:
    """Per-channel held-out r of the model made of the column blocks `cols`, one penalty per block, trained on g0-g3 and scored on g4."""
    train, test = total([groups[g] for g in TRAIN_GROUPS]), groups[TEST_GROUP]
    idx = np.concatenate(cols)
    penalty = np.concatenate([np.full(len(c), a) for c, a in zip(cols, alphas)])
    return pearson(fit(train.take(idx), penalty), test.take(idx))


def per_channel_delta(groups: dict[str, RunStats], alphas: dict[str, list[float]]) -> np.ndarray:
    cols = layout()
    nuisance = [cols["C"], cols["eye"]]
    return model_r(groups, [*nuisance, cols["L"]], alphas["N+L"]) - model_r(groups, nuisance, alphas["N"])


def feature_deltas(groups: dict[str, RunStats], alphas: dict[str, list[float]]) -> dict[str, float]:
    """Channel-mean held-out gain of each lexical feature on its own, with the N+L penalties (post hoc)."""
    cols = layout()
    nuisance = [cols["C"], cols["eye"]]
    r_n = model_r(groups, nuisance, alphas["N"])
    *a_n, a_l = alphas["N+L"]
    n_lag = len(CHAR_LAGS)
    feature_cols = {"log_frequency": cols["L"][:n_lag], "surprisal": cols["L"][n_lag:]}
    return {f: float(np.nanmean(model_r(groups, [*nuisance, c], [*a_n, a_l]) - r_n)) for f, c in feature_cols.items()}


def load_h1(d: Path) -> dict[str, str]:
    return dict(line.split(": ") for line in (d / "h1.txt").read_text(encoding="utf-8-sig").splitlines() if ": " in line)


def read_summary(d: Path) -> dict:
    """The pre-specified items of one locked read: lexical gain, controls and permutation floor."""
    t = subject_table(pd.read_csv(d / "results.csv"))
    h1 = load_h1(d)
    lo, hi = bootstrap_ci(t.delta)
    p = [json.loads((d / f"snapshot_{s}.json").read_text())["permutation"]["p"] for s in t.index]
    return {"n": len(t), "mean": float(t.delta.mean()), "lo": lo, "hi": hi, "positive": int((t.delta > 0).sum()),
            "p": float(h1["lexical_p"]), "floor": sum(v <= 0.005 for v in p),
            **{f"{k}_{m}": (int if m == "positive" else float)(h1[f"{k}_{m}"]) for k in ("strokes", "embedding") for m in ("mean", "positive", "p")}}


def reads_tex(first: dict, second: dict) -> str:
    def frac(s, k):
        return f"{s[k]}/{s['n']}"
    rows = [("Mean $\\Delta R$ ($\\times10^{-5}$)", lambda s: sci(s["mean"])),
            ("Bootstrap 95\\,\\% CI", lambda s: f"{sci(s['lo'])} to {sci(s['hi'])}"),
            ("Readers with $\\Delta R > 0$", lambda s: frac(s, "positive")),
            ("One-sided Wilcoxon $p$", lambda s: f"{s['p']:.4f}"),
            ("Readers at the permutation floor", lambda s: frac(s, "floor"))]
    for control in ("strokes", "embedding"):
        rows += [(f"{control.capitalize()} mean", lambda s, c=control: sci(s[f"{c}_mean"])),
                 (f"{control.capitalize()} readers $> 0$", lambda s, c=control: frac(s, f"{c}_positive")),
                 (f"{control.capitalize()} $p$", lambda s, c=control: f"{s[f'{c}_p']:.3f}")]
    body = "\n".join(f"{name} & {f(first)} & {f(second)} \\\\" for name, f in rows)
    return ("\\begin{tabular}{lrr}\n\\toprule\n & First read & Second read \\\\\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}\n")


def null_table(d: Path, observed: dict[str, float]) -> pd.DataFrame:
    """Post hoc surrogate pass, per reader and null kind: p with the fixed and the re-selected lexical penalty
    and the surrogate means under both rules. Refuses a pass whose observed gains differ from the locked read."""
    rows = []
    for s, obs in observed.items():
        nulls = json.loads((d / f"snapshot_{s}.json").read_text())["nulls"]
        for kind in NULL_KINDS:
            if nulls[kind]["observed_delta_r"] != obs:
                raise ValueError(f"{s} {kind}: observed gain {nulls[kind]['observed_delta_r']} differs from the locked read {obs}")
            perms = pd.read_csv(d / (f"perms_{s}.csv" if kind == "shuffle" else f"perms_{kind}_{s}.csv"))
            rows.append({"subject": s, "kind": kind, "observed": obs, "p": nulls[kind]["p"], "p_reselected": nulls[kind]["p_reselected"],
                         "null_mean": perms.delta_r.mean(), "null_mean_reselected": perms.delta_r_reselected.mean(),
                         "null_sd": perms.delta_r.std(), "null_sd_reselected": perms.delta_r_reselected.std()})
    return pd.DataFrame(rows).set_index(["subject", "kind"])


def nulls_tex(n: pd.DataFrame) -> str:
    def cells(s, kind):
        row = n.loc[(s, kind)]
        return f"{sci(row.null_mean)} & {row.p:.3f} & {row.p_reselected:.3f}"
    subjects = n.index.get_level_values("subject").unique()
    rows = [f"{s.replace('sub-', 'S')} & {sci(n.loc[(s, 'shuffle')].observed)} & {cells(s, 'shuffle')} & {cells(s, 'circular')} \\\\"
            for s in subjects]
    head = ("\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{Within-run shuffle} & \\multicolumn{3}{c}{Circular shift} \\\\\n"
            "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n"
            "Reader & $\\Delta R$ & null mean & $p$ & $p_\\text{re}$ & null mean & $p$ & $p_\\text{re}$ \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def nulls_macros(n: pd.DataFrame) -> dict[str, str]:
    floor = {f"nulls{k.capitalize()}Floor{suffix}": str(int((n.xs(k, level='kind')[col] <= P_FLOOR).sum()))
             for k in NULL_KINDS for suffix, col in (("", "p"), ("Re", "p_reselected"))}
    shuffle = n.xs("shuffle", level="kind")
    return {**floor, "nullsReMeanMin": sci(n.null_mean_reselected.min()), "nullsReMeanMax": sci(n.null_mean_reselected.max()),
            "nullsCircNeg": str(int((n.xs("circular", level="kind").null_mean < 0).sum())),
            "nullsReNarrower": str(int((shuffle.null_sd_reselected < shuffle.null_sd).sum())),
            "nullsMaxP": f"{n[['p', 'p_reselected']].to_numpy().max():.2f}",
            "nullsFloorAllMin": str(int(((n[['p', 'p_reselected']] <= P_FLOOR).all(axis=1)).groupby(level='subject').all().sum()))}


def gate_values(folder: Path) -> pd.DataFrame:
    rows = [{"subject": r["subject"], **r["p2t_position_ge5_uV"], "auc": r["first_char_auc_mean"]}
            for f in sorted(folder.glob("gate1*.json")) for r in json.loads(f.read_text())]
    g = pd.DataFrame(rows).rename(columns={"fixed_0.35": "fixed_035"}).set_index("subject").sort_index()
    g["passes"] = ((g.row_pace > GATE_RATIO * g.jittered) & (g.row_pace > g.fixed_035) & (g.row_pace > g.jittered)
                   & (g.auc > GATE_AUC))
    return g


def gate_tex(first: pd.DataFrame, corrected: pd.DataFrame) -> str:
    def mark(ok):
        return "yes" if ok else "no"
    rows = [f"{s.replace('sub-', 'S')} & {a.row_pace:.2f} & {a.fixed_035:.2f} & {a.jittered:.2f} & {a.auc:.2f} & {mark(a.passes)}"
            f" & {corrected.loc[s].row_pace:.2f} & {corrected.loc[s].jittered:.2f} & {mark(corrected.loc[s].passes)} \\\\"
            for s, a in first.iterrows()]
    head = ("\\begin{tabular}{lrrrrlrrl}\n\\toprule\n"
            " & \\multicolumn{5}{c}{Span$/n$ onsets (selection)} & \\multicolumn{3}{c}{Corrected onsets} \\\\\n"
            "\\cmidrule(lr){2-6} \\cmidrule(lr){7-9}\n"
            "Reader & Row & Fixed & Jit. & AUC & Pass & Row & Jit. & Pass \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def sci(x: float, exp: int = -5, digits: int = 1) -> str:
    return f"{x / 10 ** exp:.{digits}f}"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "\\,")


def macros(values: dict[str, str]) -> str:
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in values.items())


def table_tex(t: pd.DataFrame, z: dict[str, float], p: dict[str, float]) -> str:
    rows = [f"{s.replace('sub-', 'S')} & {row.r_n:.3f} & {row.r_neye:.3f} & {sci(row.delta)} & {row.pct_n:.3f} & "
            f"{z[s]:.1f} & {p[s]:.3f} \\\\" for s, row in t.iterrows()]
    head = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            "Reader & $r$(N) & $r$(N$-$eye) & $\\Delta R$ & \\% & $z$ & $p$ \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def figures(t, nulls, kernels, ms, posterior, delta_ch, montage_info, out: Path):
    """nulls: {label: {subject: surrogate gains}}, drawn as strips beside each reader's observed gain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    x = np.arange(len(t))
    rng = np.random.default_rng(0)
    width = 0.6 / len(nulls)
    for i, ((label, by_subject), colour) in enumerate(zip(nulls.items(), (NEUTRAL, LIGHT))):
        centre = x - 0.3 + width * (i + 0.5)
        for xi, s in zip(centre, t.index):
            v = by_subject[s] * 1e5
            ax.scatter(xi + rng.uniform(-width / 3, width / 3, len(v)), v, s=1.5, color=colour, lw=0,
                       label=label if s == t.index[0] else None)
    ax.scatter(x, t.delta * 1e5, marker="D", s=14, color=ACCENT, zorder=3, label="Observed")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(x, [s.replace("sub-", "S") for s in t.index])
    ax.set_ylabel(r"$\Delta R$ (lexical) $\times 10^{-5}$")
    ax.legend(frameon=False, fontsize=6, markerscale=2, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig_delta.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.8), sharey=True)
    for f, name in enumerate(("Log frequency", "Surprisal")):
        trace = kernels[f][:, :, posterior].mean(axis=2) * 1e3
        for row in trace:
            ax[f].plot(ms, row, color=NEUTRAL, lw=0.5, alpha=0.7)
        ax[f].plot(ms, trace.mean(axis=0), color=ACCENT, lw=1.4)
        ax[f].axvspan(*N400, color=ACCENT, alpha=0.08, lw=0)
        ax[f].axhline(0, color=MUTED, lw=0.6)
        ax[f].set_title(name, fontsize=8)
        ax[f].set_xlabel("Lag (ms)")
    ax[0].set_ylabel(r"Posterior weight $\times 10^{-3}$")
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig_kernels.pdf")
    plt.close(fig)

    surprisal_map = window_mean_map(kernels[1], ms, N400)
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.7))
    for a, data, title in ((ax[0], delta_ch * 1e5, r"$\Delta R$ ($\times 10^{-5}$)"),
                           (ax[1], surprisal_map * 1e3, r"Surprisal, 300–500 ms ($\times 10^{-3}$)")):
        im, _ = mne.viz.plot_topomap(data, montage_info, axes=a, show=False, cmap="RdBu_r", contours=0,
                                     sensors=False)
        a.set_title(title, fontsize=8)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.savefig(out / "fig_topo.pdf", bbox_inches="tight")
    plt.close(fig)


def window_mean_map(kernels: np.ndarray, ms: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    inside = (ms >= window[0]) & (ms <= window[1])
    return kernels[:, inside].mean(axis=(0, 1))


def main(d: Path, out: Path, first: Path | None = None, nulls_dir: Path | None = None):
    import mne
    res = pd.read_csv(d / "results.csv")
    t = subject_table(res)
    subjects = list(t.index)
    lo, hi = bootstrap_ci(t.delta)
    full = {s: json.loads((d / f"snapshot_{s}.json").read_text()) for s in subjects}
    snaps = {s: full[s]["permutation"] for s in subjects}
    test_runs = {s: {int(r) for r in locked_runs(res, s)} for s in subjects}
    chars = [sum(full[s]["n_chars"].values()) for s in subjects]
    test_chars = [sum(v for k, v in full[s]["n_chars"].items() if int(k) in test_runs[s]) for s in subjects]
    runs = [len(full[s]["n_chars"]) for s in subjects]
    nulls = {s: pd.read_csv(d / f"perms_{s}.csv").delta_r.to_numpy() for s in subjects}
    z = {s: perm_z(snaps[s]["observed_delta_r"], nulls[s]) for s in subjects}
    null_means = [nulls[s].mean() for s in subjects]
    p = {s: snaps[s]["p"] for s in subjects}
    h1 = load_h1(d)

    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    targets = [f"E{c}" for c in range(1, 129) if c not in EYE_CHANNELS]
    info = mne.create_info(targets, SFREQ, "eeg").set_montage(montage)
    y = np.array([montage.get_positions()["ch_pos"][c][1] for c in targets])
    frontal, posterior = y > np.percentile(y, 75), y < np.percentile(y, 25)

    locked = res[res.split == "locked"]
    alphas = {s: {m: [float(a) for a in locked[(locked.subject == s) & (locked.model == m)].alphas.iloc[0].split()]
                  for m in ("N", "N+L")} for s in subjects}
    stats = {s: load_stats(d / f"stats_{s}.npz") for s in subjects}
    delta_ch = np.array([per_channel_delta(stats[s], alphas[s]) for s in subjects])
    group_ch = np.nanmean(delta_ch, axis=0)
    loo = pd.DataFrame([feature_deltas(stats[s], alphas[s]) for s in subjects], index=subjects)

    ms = lags_ms()
    kernels = [np.array([np.load(d / f"kernels_{s}.npz")["lexical"][f] for s in subjects]) for f in range(len(FEATURES))]
    surprisal_n400 = window_mean(kernels[1], ms, N400, posterior)
    freq_peak = gfp_peak_ms(kernels[0], ms)
    freq_at_peak = kernels[0][:, int(np.argmin(np.abs(ms - freq_peak)))][:, posterior].mean(axis=1)

    values = {
        "nSubj": str(len(subjects)),
        "charsMin": thousands(min(chars)), "charsMax": thousands(max(chars)), "testCharsMin": thousands(min(test_chars)),
        "testCharsMax": thousands(max(test_chars)), "runsMin": str(min(runs)), "runsMax": str(max(runs)),
        "condL": f"{max(full[s]['condition']['L'] for s in subjects):.1f}",
        "dRmean": sci(t.delta.mean(), digits=1), "dRlo": sci(lo), "dRhi": sci(hi),
        "dRmin": sci(t.delta.min()), "dRmax": sci(t.delta.max(), digits=1),
        "dRpos": str(int((t.delta > 0).sum())), "dRp": f"{float(h1['lexical_p']):.4f}",
        "pctNmin": f"{t.pct_n.min():.3f}", "pctNmax": f"{t.pct_n.max():.2f}",
        "pctNeyemin": f"{t.pct_neye.min():.3f}", "pctNeyemax": f"{t.pct_neye.max():.2f}",
        "rNmin": f"{t.r_n.min():.2f}", "rNmax": f"{t.r_n.max():.2f}",
        "rNeyemin": f"{t.r_neye.min():.2f}", "rNeyemax": f"{t.r_neye.max():.2f}",
        "nullNeg": str(sum(m < 0 for m in null_means)), "nullMin": sci(min(null_means)), "nullMax": sci(max(null_means)),
        "permFloor": str(sum(v <= 0.005 for v in p.values())), "zmin": f"{min(z.values()):.1f}", "zmax": f"{max(z.values()):.1f}",
        "strokesMean": sci(float(h1["strokes_mean"])), "strokesPos": h1["strokes_positive"], "strokesP": f"{float(h1['strokes_p']):.3f}",
        "embMean": sci(float(h1["embedding_mean"])), "embPos": h1["embedding_positive"], "embP": f"{float(h1['embedding_p']):.3f}",
        "dRfrontal": sci(group_ch[frontal].mean()), "dRposterior": sci(group_ch[posterior].mean()),
        "postGtFront": str(int((np.nanmean(delta_ch[:, posterior], 1) > np.nanmean(delta_ch[:, frontal], 1)).sum())),
        "surpPeak": f"{gfp_peak_ms(kernels[1], ms):.0f}", "surpNeg": str(int((surprisal_n400 < 0).sum())),
        "freqPeak": f"{freq_peak:.0f}", "freqPosAtPeak": str(int((freq_at_peak > 0).sum())),
        "looFreqMean": sci(loo.log_frequency.mean()), "looFreqPos": str(int((loo.log_frequency > 0).sum())),
        "looSurpMean": sci(loo.surprisal.mean()), "looSurpPos": str(int((loo.surprisal > 0).sum())),
    }
    values["strokesOverLex"] = f"{float(h1['strokes_mean']) / t.delta.mean():.1f}"
    out.mkdir(parents=True, exist_ok=True)
    strips = {"Shuffle with fixed penalty": nulls}
    if nulls_dir is not None:
        n = null_table(nulls_dir, {s: snaps[s]["observed_delta_r"] for s in subjects})
        values.update(nulls_macros(n))
        (out / "table_nulls.tex").write_text(nulls_tex(n), encoding="utf-8")
        strips["Shuffle with re-selected penalty"] = {
            s: pd.read_csv(nulls_dir / f"perms_{s}.csv").delta_r_reselected.to_numpy() for s in subjects}
    gate_dirs = [d.parent / g for g in GATE_DIRS]
    if all(g.is_dir() for g in gate_dirs):
        (out / "table_gate.tex").write_text(gate_tex(*(gate_values(g) for g in gate_dirs)), encoding="utf-8")
    (out / "numbers.tex").write_text(macros(values), encoding="utf-8")
    (out / "table_subjects.tex").write_text(table_tex(t, z, p), encoding="utf-8")
    if first is not None:
        (out / "table_reads.tex").write_text(reads_tex(read_summary(first), read_summary(d)), encoding="utf-8")
    figures(t, strips, kernels, ms, posterior, group_ch, info, out)


if __name__ == "__main__":
    paths = [Path(a) for a in sys.argv[1:5]]
    main(*paths[:2], *paths[2:], *[None] * (4 - len(paths)))
