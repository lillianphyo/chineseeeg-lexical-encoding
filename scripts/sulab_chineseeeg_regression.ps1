# ChineseEEG (ds004952) overlap-corrected lexical regression (lab01/.claude/prds/exp-chineseeeg-overlap-regression-2026-10.md),
# CPU only, one runner call per Gate-1 subject. -Mode dev: 4-fold CV over groups 0-3 (locked runs never read).
# -Mode dev2: dev iteration 2 (0.5-8 Hz, robust z clipped at 5; PRD 2026-10-30), same split as dev.
# -Mode dev-corrected / locked-corrected: same as dev2 / locked but with the corrected character step (2026-11-02);
# the code decides the step, these modes only keep the outputs apart.
# -Mode locked-nulls: post hoc sensitivity run for the advisor's review (2026-11-03): the locked runs are read a third time
# only to build shuffle and circular-shift surrogates with and without lexical-penalty re-selection; observed r unchanged.
# -Mode locked: dev-2 settings (0.5-8 Hz, clip 5), fit on groups 0-3, score the locked-test runs once with 200 lexical permutations, then the
# pre-specified H1 test (h1.txt). Locked writes LOCKED.txt first and exits at once after DONE.txt.
# A subject whose snapshot exists is skipped, so a killed task resumes per subject.
# Scheduled task via scripts\register_chineseeeg_regression.ps1 -Mode dev|locked.
param([ValidateSet("dev", "dev2", "locked", "dev-corrected", "locked-corrected", "locked-nulls")][string]$Mode = "dev")
$env:KMP_DUPLICATE_LIB_OK = "TRUE"; $env:PYTHONUTF8 = "1"; $env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "src;."
$py = "C:\Users\shimamura_lab\anaconda3\envs\zuco-blackwell\python.exe"
$repo = "D:\year2\zuco-llm-alignment"
$out = @{ dev = "runs/chineseeeg_regression_dev"; dev2 = "runs/chineseeeg_regression_dev2"; locked = "runs/chineseeeg_regression_locked"; "dev-corrected" = "runs/chineseeeg_regression_dev_corrected"; "locked-corrected" = "runs/chineseeeg_regression_locked_corrected"; "locked-nulls" = "runs/chineseeeg_regression_locked_nulls" }[$Mode]
$subjects = "sub-04", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-14"
Set-Location $repo
New-Item -ItemType Directory -Force -Path $out | Out-Null
if (Test-Path "$out/DONE.txt") { exit 0 }
if ($Mode -like "locked*") {
    if (Test-Path "$out/LOCKED.txt") { "RESUMED $(Get-Date -Format s)" | Out-File "$out/LOCKED.txt" -Append }
    else { "LOCKED-TEST READ STARTED $(Get-Date -Format s) git $(git rev-parse --short HEAD)" | Out-File "$out/LOCKED.txt" }
}
$data = "--root F:\datasets\ds004952 --table F:\datasets\ds004952_chartable\char_table.npz --surprisal F:\datasets\ds004952_chartable\surprisal.npz --subtlex F:\datasets\subtlex_ch\SUBTLEX-CH-CHR"
$flag = @{ dev = ""; dev2 = "--high 8 --clip 5"; locked = "--locked-test --high 8 --clip 5"; "dev-corrected" = "--high 8 --clip 5"; "locked-corrected" = "--locked-test --high 8 --clip 5"; "locked-nulls" = "--locked-test --high 8 --clip 5 --null shuffle circular --reselect" }[$Mode]
foreach ($s in $subjects) {
    if (Test-Path "$out/snapshot_$s.json") { continue }
    $t0 = Get-Date
    & $py ((("scripts/run_chineseeeg_regression.py $data $flag --subject $s --out $out/results.csv") -split " ") | Where-Object { $_ }) 2>&1 | Out-File "$out/run.log" -Append -Encoding utf8
    "$s $($t0.ToString('s')) $([math]::Round(((Get-Date) - $t0).TotalMinutes, 1)) min" | Out-File "$out/timing.txt" -Append
}
if ($Mode -like "locked*") { & $py scripts/chineseeeg_regression_tests.py "$out/results.csv" 2>&1 | Out-File "$out/h1.txt" -Encoding utf8 }
"DONE $(Get-Date -Format s)" | Out-File "$out/DONE.txt"
