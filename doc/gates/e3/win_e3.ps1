# E3 on Windows (CUDA build): the PowerShell counterpart of mixed_gate.sh (gates 2 and 3, mixed
# precision on CUDA: bf16 operands on the tensor cores) and of gate 1 (every option off: E2's
# ddp_gate references and the single-device differential, through doc\gates\e2\win_gates.ps1).
# Run from the source tree after `cargo +1.94.1 build --release -j 2 --features cuda --examples`
# (and, for -Part off, the 0.5.0 tree built the same way with E2's probe in its examples):
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File doc\gates\e3\win_e3.ps1 -Part gate
#   powershell -NoProfile -ExecutionPolicy Bypass -File doc\gates\e3\win_e3.ps1 -Part off -Ref <folder of the 0.5.0 tree>
#
# Every process runs on 2 logical CPUs (affinity mask 3) at below-normal priority; at most 2
# ranks. Logs (UTF-8): doc\gates\e3\logs\win-cuda-mixed.log and win-cuda-off.log.
param([string]$Part = 'gate', [string]$Ref = '')
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['*:Encoding'] = 'utf8'
$me = [System.Diagnostics.Process]::GetCurrentProcess()
$me.ProcessorAffinity = [IntPtr]3
$me.PriorityClass = 'BelowNormal'
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$logs = "$PSScriptRoot\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
Set-Location $root
$E = "$root\target\release\examples"
$rv = (& rustc +1.94.1 --version) -replace ' [(].*', ''
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
$gpu = ((& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) | Select-Object -First 1).Trim()
$script:fail = 0

if ($Part -eq 'gate') {
    $work = "$root\runs\e3-win"
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $work | Out-Null
    $log = "$logs\win-cuda-mixed.log"
    function Say([string]$s) { Add-Content -Path $log -Value $s }
    Set-Content -Path $log -Value "# E3 gates 2 and 3 (mixed precision), win-cuda: $gpu; $cpu; $rv; Windows $([Environment]::OSVersion.Version) x86_64; $(Get-Date -Format o)"
    Say '# every process on 2 logical CPUs at below-normal priority; ranks share the GPU; mixed results are compared with f32 by a tolerance, never byte for byte'
    function Final([string]$d) { ((Get-Content "$d\record.txt") | Where-Object { $_ -like 'final *' }) -replace '^final ', '' }
    function Rec([string]$d) { [regex]::Replace([regex]::Replace((Get-Content "$d\record.txt")[0], 'libm=[0-9a-f]+', 'libm=<sha256>'), 'kernels-[0-9a-f]+', 'kernels-<hex>') }
    function One([string]$n, [string[]]$a) { & "$E\ddp_gate.exe" --device cuda --out "$work\$n" @a *> "$work\$n.out"; return ($LASTEXITCODE -eq 0) }
    function Ranks([string]$n, [int]$w, [string[]]$a, [string]$out = '') {
        if ($out -eq '') { $out = "$work\$n" }
        & "$E\launch.exe" --nproc-per-node $w --log-dir "$work\$n\logs" -- "$E\ddp_gate.exe" --device cuda --ranks-per-device $w --out $out @a *> "$work\$n.out"
        return ($LASTEXITCODE -eq 0)
    }
    function Steps([string]$d) { (Get-Content "$d\rank-0.steps") -join "`n" }
    function Same([string]$x, [string]$y) { ((Steps $x) -eq (Steps $y)) -and ((Final $x) -eq (Final $y)) }
    function YesNo([bool]$b) { if ($b) { 'yes' } else { $script:fail = 1; 'NO' } }

    Say '## Gate 2: run to run and restart, per key (12 steps, M 2)'
    foreach ($cfg in @(@('one', 1, @()), @('ddp', 2, @()), @('zero', 2, @('--zero', '2:flat')))) {
        $n = $cfg[0]; $w = $cfg[1]; $extra = $cfg[2]
        $base = @('--micro', '2', '--mixed') + $extra
        foreach ($run in @('a', 'b')) {
            $ok = if ($w -eq 1) { One "$n-$run" $base } else { Ranks "$n-$run" $w $base }
            if (-not $ok) { Say "$n $run : FAILED"; $script:fail = 1 }
        }
        Say "$n (W $w $($extra -join ' ')): final weights sha256 $(Final "$work\$n-a"); run twice, every per-step hash and the final weights the same: $(YesNo (Same "$work\$n-a" "$work\$n-b"))"
        $r = "$n-resume"
        if ($w -eq 1) { $ok = (One $r ($base + @('--steps', '7', '--ckpt-every', '3'))) -and (One $r ($base + @('--resume'))) }
        else { $ok = (Ranks $r $w ($base + @('--steps', '7', '--ckpt-every', '3'))) -and (Ranks "$r-2" $w ($base + @('--resume')) "$work\$r") }
        if (-not $ok) { Say "$n resume: FAILED"; $script:fail = 1 }
        $uniq = (Get-Content "$work\$r\rank-0.steps" | Sort-Object -Unique | Sort-Object { [int]($_ -split ' ')[0] }) -join "`n"
        Say "${n}: 7 steps with checkpoints every 3, a new launch resuming from step 6 to 12: every per-step hash and the final weights the uninterrupted run's: $(YesNo (($uniq -eq (Steps "$work\$n-a")) -and ((Final "$work\$r") -eq (Final "$work\$n-a"))))"
    }
    Say "W 2 data-parallel against one process (the deterministic mode's identity within the mixed key): $(YesNo (Same "$work\ddp-a" "$work\one-a"))"
    Say "W 2 sharded (stage 2, flat) against W 2 data-parallel: $(YesNo (Same "$work\zero-a" "$work\ddp-a"))"
    Say "key and record (one process): $(Rec "$work\one-a")"
    Say "key and record (sharded): $(Rec "$work\zero-a")"
    $null = Ranks 'fast-a' 2 @('--micro', '4', '--mixed', '--fast'); $null = Ranks 'fast-b' 2 @('--micro', '4', '--mixed', '--fast')
    Say "fast mode, W 2, M 4, run twice: the same per-step hashes: $(if (Same "$work\fast-a" "$work\fast-b") { 'yes' } else { 'no' }) (an observation; the fast mode makes no such promise); key: $(Rec "$work\fast-a")"
    if (One 'f32-resume' @('--micro', '2', '--ckpt-dir', "$work\one-resume\ckpt", '--resume')) { Say 'f32 resume of the mixed checkpoint: NOT REFUSED'; $script:fail = 1 }
    else { Say "f32 resume of the mixed checkpoint: refused ($(if (Select-String -Quiet 'resume refused' "$work\f32-resume.out") { 'resume refused: another key' } else { 'another error' }))" }

    Say '## Gate 3: the loss against f32 (one process, M 4, 200 steps)'
    foreach ($x in @(@('f32-long', @()), @('mixed-long', @('--mixed')), @('mixed8-long', @('--mixed', '--loss-scale', '8')))) {
        if (-not (One $x[0] (@('--micro', '4', '--steps', '200', '--losses') + $x[1]))) { Say "$($x[0]): FAILED"; $script:fail = 1 }
    }
    function Losses([string]$n) { $h = @{}; foreach ($l in (Get-Content "$work\$n.out")) { if ($l -match '^loss (\d+) (\S+)$') { $h[[int]$Matches[1]] = [double]$Matches[2] } }; $h }
    $f = Losses 'f32-long'
    function Mean($h, [int]$lo, [int]$hi) { $s = 0.0; $c = 0; foreach ($k in $lo..$hi) { if ($h.ContainsKey($k)) { $s += $h[$k]; $c++ } }; '{0:f4}' -f ($s / [math]::Max($c, 1)) }
    foreach ($n in @('mixed-long', 'mixed8-long')) {
        $m = Losses $n
        $out = foreach ($range in @(@(1, 12), @(1, 200))) {
            $mx = 0.0; $at = 0; $rm = 0.0
            foreach ($k in $range[0]..$range[1]) { if ($f.ContainsKey($k) -and $m.ContainsKey($k)) { $d = [math]::Abs($m[$k] - $f[$k]); if ($d -ge $mx) { $mx = $d; $at = $k }; $rr = $d / [math]::Abs($f[$k]); if ($rr -gt $rm) { $rm = $rr } } }
            "steps $($range[0])-$($range[1]) largest |difference| {0:e3} at step {1} (relative {2:e3})" -f $mx, $at, $rm
        }
        $skipped = (Select-String -Path "$work\$n.out" -Pattern '^skipped' | Measure-Object).Count
        Say "$n against f32: $($out -join '; '); skipped steps $skipped"
    }
    foreach ($n in @('f32-long', 'mixed-long', 'mixed8-long')) {
        $h = Losses $n
        Say "mean loss by window, ${n}: steps 1-12 $(Mean $h 1 12), 13-50 $(Mean $h 13 50), 51-100 $(Mean $h 51 100), 101-150 $(Mean $h 101 150), 151-200 $(Mean $h 151 200)"
    }
    $status = if ($script:fail -eq 0) { 'all checks passed' } else { 'FAILURES' }
    Say "# ended $(Get-Date -Format o), $status"
    Get-Content $log
    exit $script:fail
}

if ($Part -eq 'off') {
    $log = "$logs\win-cuda-off.log"
    Set-Content -Path $log -Value "# E3 gate 1 (every option off) on Windows: $gpu; $cpu; $rv; $(Get-Date -Format o)"
    $e2 = "$root\doc\gates\e2"
    $recorded = Get-Content "$e2\logs\win-ddp-gate.log"
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$e2\win_gates.ps1" -Part ddpgate *> "$root\runs\e3-off-ddpgate.out"
    $now = Get-Content "$e2\logs\win-ddp-gate.log"
    $pick = { param($lines) $lines | ForEach-Object { if ($_ -match '^([^#].*?)( \(.*?\))?: final weights sha256 ([0-9a-f]{64})') { "$($Matches[1]) $($Matches[3])" } } }
    $a = & $pick $recorded; $b = & $pick $now
    $same = (($a -join "`n") -eq ($b -join "`n")) -and $a.Count -gt 0
    if (-not $same) { $script:fail = 1 }
    Add-Content $log "ddp_gate references (doc\gates\e2\win_gates.ps1 -Part ddpgate) against 0.5.0's recorded log: $($b.Count) final-weights lines, $(if ($same) { 'all identical' } else { 'DIFFERENT' })"
    $now | Where-Object { $_ -match 'final weights sha256|resume' } | ForEach-Object { Add-Content $log "  $_" }
    Set-Content "$e2\logs\win-ddp-gate.log" $recorded
    if ($Ref -ne '') {
        & powershell -NoProfile -ExecutionPolicy Bypass -File "$e2\win_gates.ps1" -Part diff -Ref $Ref *> "$root\runs\e3-off-diff.out"
        $d = Get-Content "$e2\logs\win-single-device.log"
        Add-Content $log "single-device differential (doc\gates\e2\win_gates.ps1 -Part diff, CPU and CUDA) against the 0.5.0 tree:"
        $d | Where-Object { $_ -match 'identical|decoder adamw' } | ForEach-Object { Add-Content $log "  $_" }
        if (($d | Where-Object { $_ -match 'identical: yes' }).Count -ne 2) { $script:fail = 1 }
    }
    $status = if ($script:fail -eq 0) { 'all identical' } else { 'FAILURES' }
    Add-Content $log "# ended $(Get-Date -Format o), $status"
    Get-Content $log
    exit $script:fail
}
