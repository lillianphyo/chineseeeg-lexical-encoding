# ChineseEEG lexical encoding

Code, plan and locked-run outputs for *A Small, Consistent Lexical Effect in Continuous Chinese Reading EEG: Overlap-Corrected Encoding with Run-Level Splits and Training-Only Penalty Selection* (ICSIE 2027 submission).

An overlap-corrected, banded-ridge encoding model of continuous EEG from ChineseEEG ([OpenNeuro ds004952 v1.2.2](https://openneuro.org/datasets/ds004952), CC0). It tests whether character log frequency (SUBTLEX-CH) and GPT-2 character surprisal add held-out accuracy over a nuisance model (character onset, row position, row boundaries, stroke count, 11 periocular channels). Runs are split by order (`i mod 5`); group 4 is the locked test set. It was scored twice: once with onsets at row span / n (2026-10-31) and once, after the advisor's review, with the corrected step (span − 0.35 s) / (n − 1) (2026-11-03). Both reads are kept and reported.

## Layout

| Path | Content |
|---|---|
| `src/zuco_llm_align/chineseeeg.py` | loader, character onsets from the row markers and the presentation code's 0.35 s display, run split |
| `src/zuco_llm_align/chineseeeg_regression.py` | lag design, per-run sufficient statistics, banded ridge, held-out Pearson r; raises if a run is on both sides of a split |
| `scripts/chineseeeg_char_table.py`, `scripts/chineseeeg_surprisal.py` | character table and surprisal (text only) |
| `scripts/chineseeeg_timing_check.py` | timing gate (which readers enter) |
| `scripts/chineseeeg_row_duration_check.py` | row-duration regression that verifies the character step from the markers (`runs/row_duration_check.txt`) |
| `scripts/run_chineseeeg_regression.py` | dev / locked runs; `--null shuffle circular --reselect` adds the post hoc surrogate nulls |
| `scripts/chineseeeg_regression_tests.py` | the pre-specified H1 test (`h1.txt`) |
| `scripts/chineseeeg_regression_effect_sizes.py` | effect-size report (`effect_sizes.txt`) |
| `scripts/chineseeeg_paper_assets.py` | every number, table and figure in the paper (`<second read> paper <first read> <surrogate pass>`) |
| `scripts/*_chineseeeg_regression.ps1` | launcher used on the lab workstation (writes `LOCKED.txt`) |
| `docs/plan.md` | experiment plan with all amendments, in the order they were made |
| `runs/gate1_first_old_onsets/`, `runs/gate1_corrected/` | timing gate verdicts with span / n onsets (2026-10-28) and with the corrected onsets (2026-11-03) |
| `runs/locked_first_read_old_onsets/` | first locked read, span / n onsets (results, permutations, kernels, per-group statistics, snapshots) |
| `runs/chineseeeg_regression_locked_corrected/` | second locked read, corrected onsets; the paper's numbers |
| `runs/chineseeeg_regression_locked_nulls/` | post hoc surrogate pass over the same runs (`--null shuffle circular --reselect`): `perms_*.csv`, `perms_circular_*.csv` with fixed and re-selected penalty; observed gains identical to the second read (Table 3) |
| `paper/` | manuscript source |

The package keeps its original name (`zuco_llm_align`) because the code was developed in a larger project; this repository is a snapshot of the ChineseEEG part. Source commits: locked run 66cef61, effect sizes 4754ed3, paper assets 4197797 and later (development repository, private).

## Run

```bash
conda env create -f environment.yml && conda activate zuco-llm-align
pip install -e .
python -m pytest tests                      # no data needed
python scripts/run_chineseeeg_regression.py --smoke --root <ds004952> ...   # see --help
python scripts/chineseeeg_paper_assets.py runs/chineseeeg_regression_locked_corrected paper \
    runs/locked_first_read_old_onsets runs/chineseeeg_regression_locked_nulls
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The full run needs ds004952 (preprocessed derivative, about 53 GB) and SUBTLEX-CH; it took about 50 CPU-minutes per reader.
