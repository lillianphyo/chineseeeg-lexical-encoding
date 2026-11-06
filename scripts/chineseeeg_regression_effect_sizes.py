"""Effect sizes for a passed H1 (lab01/.claude/prds/exp-chineseeeg-overlap-regression-2026-10.md, effect-size reporting rule).

    python scripts/chineseeeg_regression_effect_sizes.py runs/chineseeeg_regression_locked
"""
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from zuco_llm_align.chineseeeg import EYE_CHANNELS, SFREQ
from zuco_llm_align.chineseeeg_regression import (CHAR_LAGS, MODELS, RunStats, fit, model_index, pearson,
                                                  penalty_of, total)

d = Path(sys.argv[1])
res = pd.read_csv(d / "results.csv")
res = res[res.split == "locked"]
r = res.pivot_table(index="subject", columns="model", values="r")
subjects = list(r.index)
delta = r["N+L"] - r["N"]
rng = np.random.default_rng(0)
boot = [delta.sample(len(delta), replace=True, random_state=rng.integers(1 << 31)).mean() for _ in range(10000)]
print("(1) per-subject delta r:", delta.round(6).to_dict())
print(f"    mean {delta.mean():.6f}, bootstrap 95% CI [{np.percentile(boot, 2.5):.6f}, {np.percentile(boot, 97.5):.6f}]")
print("(2) % of r(N):", (100 * delta / r["N"]).round(4).to_dict())
print("    % of r(N-eye):", (100 * delta / r["N-eye"]).round(4).to_dict())
print("    r(N):", r["N"].round(4).to_dict(), " r(N-eye):", r["N-eye"].round(4).to_dict())

print("(3) permutation:")
for s in subjects:
    perm = pd.read_csv(d / f"perms_{s}.csv").delta_r
    snap = json.loads((d / f"snapshot_{s}.json").read_text())["permutation"]
    print(f"    {s}: observed {snap['observed_delta_r']:.6f}, null mean {perm.mean():.6f} sd {perm.std():.6f}, "
          f"z {(snap['observed_delta_r'] - perm.mean()) / perm.std():.1f}, p {snap['p']:.4f}")

montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
pos = montage.get_positions()["ch_pos"]
targets = [c for c in range(1, 129) if c not in EYE_CHANNELS]
y = np.array([pos[f"E{c}"][1] for c in targets])
frontal = y > np.percentile(y, 75)
posterior = y < np.percentile(y, 25)


def load(s):
    z = np.load(d / f"stats_{s}.npz")
    groups = sorted({k.split("_")[0] for k in z})
    return {g: RunStats(*(z[f"{g}_{k}"].astype(np.float64) for k in ("G", "B", "sx", "sy", "syy")), int(z[f"{g}_n"])) for g in groups}


print("(4) per-channel delta r (refit from group sums, penalties from results.csv):")
per_ch = {}
for s in subjects:
    g = load(s)
    train, test = total([g[k] for k in ("g0", "g1", "g2", "g3")]), g["g4"]
    rr = {}
    for m in ("N", "N+L"):
        idx, bands = model_index(MODELS[m])
        alphas = [float(a) for a in res[(res.subject == s) & (res.model == m)].alphas.iloc[0].split()]
        rr[m] = pearson(fit(train.take(idx), penalty_of(bands, alphas)), test.take(idx))
    dch = rr["N+L"] - rr["N"]
    per_ch[s] = dch
    top = np.argsort(dch)[::-1][:5]
    print(f"    {s}: channel mean {np.nanmean(dch):.6f} (csv {delta[s]:.6f}); frontal quarter {np.nanmean(dch[frontal]):.6f}, "
          f"posterior quarter {np.nanmean(dch[posterior]):.6f}; top E{[targets[i] for i in top]}")
m = np.nanmean(np.array(list(per_ch.values())), axis=0)
print(f"    group: frontal {m[frontal].mean():.6f}, posterior {m[posterior].mean():.6f}, top5 E{[targets[i] for i in np.argsort(m)[::-1][:5]]}")

print("(5) lexical kernels (channel-mean |w| peak and signed mean at peak, over posterior quarter):")
lags_ms = CHAR_LAGS / SFREQ * 1000
for f, name in enumerate(("log_frequency", "surprisal")):
    ws = np.array([np.load(d / f"kernels_{s}.npz")["lexical"][f] for s in subjects])
    gfp = np.sqrt((ws ** 2).mean(axis=2)).mean(axis=0)
    peak = int(np.argmax(gfp))
    signed = ws[:, peak, :][:, posterior].mean(axis=1)
    print(f"    {name}: GFP peak {lags_ms[peak]:.0f} ms; posterior mean at peak per subject {np.round(signed, 5).tolist()}")
    win = (lags_ms >= 300) & (lags_ms <= 500)
    print(f"      300-500 ms posterior mean per subject {np.round(ws[:, win][:, :, posterior].mean(axis=(1, 2)), 5).tolist()}")
