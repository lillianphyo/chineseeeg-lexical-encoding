# Registers and starts the ChineseEEG regression worker as a scheduled task on sulab (PowerShell 5.1, run as shimamura_lab):
#   powershell -NoProfile -ExecutionPolicy Bypass -File D:\year2\zuco-llm-alignment\scripts\register_chineseeeg_regression.ps1 -Mode dev
# The 8 h limit (16 h for locked-nulls, two surrogate kinds with re-selection) is about twice the estimate per mode; a killed task resumes per subject on the next start or boot.
param([ValidateSet("dev", "dev2", "locked", "dev-corrected", "locked-corrected", "locked-nulls")][string]$Mode = "dev")
$task = "chineseeeg_regression_$Mode"
$repo = "D:\year2\zuco-llm-alignment"
$out = "chineseeeg_regression_$($Mode -replace '-', '_')"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File $repo\scripts\sulab_chineseeeg_regression.ps1 -Mode $Mode"
$hours = if ($Mode -eq "locked-nulls") { 16 } else { 8 }
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours $hours) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId "shimamura_lab" -LogonType S4U -RunLevel Limited
Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $task -Action $action -Principal $principal -Settings $settings | Out-Null
if (-not (Select-String -Path D:\year2\resume_queue.txt -Pattern "^$task\|" -Quiet)) {
    Add-Content D:\year2\resume_queue.txt "$task|$repo\runs\$out\DONE.txt"
}
Start-ScheduledTask -TaskName $task
Start-Sleep 90
Get-ScheduledTask -TaskName $task | Select-Object TaskName, State
Get-Content "$repo\runs\$out\run.log" -Tail 3 -ErrorAction SilentlyContinue
