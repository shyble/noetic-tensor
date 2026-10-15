# C1 on Windows (CUDA build): the PowerShell counterpart of gate.sh (gate 2, the reduced-precision
# product on CUDA) and of gate 1 (the product off: E2's ddp_gate references and the single-device
# differential, through doc\gates\e2\win_gates.ps1). Run from the source tree after
# `cargo +1.94.1 test --release -j 2 --features cuda --lib --no-run` and
# `cargo +1.94.1 build --release -j 2 --features cuda --examples`:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File doc\gates\c1\win_c1.ps1 -Part gate
#   powershell -NoProfile -ExecutionPolicy Bypass -File doc\gates\c1\win_c1.ps1 -Part off -Ref <folder of the 0.5.0 tree>
#
# Every process runs on 2 logical CPUs (affinity mask 3) at below-normal priority. Logs:
# doc\gates\c1\logs\win-cuda-half.log (gate 2) and win-cuda-off.log (gate 1).
param([string]$Part = 'gate', [string]$Ref = '')
$ErrorActionPreference = 'Continue'
# UTF-8 throughout: the native executables' output is read, and the logs written, as UTF-8 (the logs
# hold Σ, × and ᵀ).
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
    $work = "$root\runs\c1-gate-win"
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $work | Out-Null
    $log = "$logs\win-cuda-half.log"
    function Say([string]$s) { Add-Content -Path $log -Value $s }
    Set-Content -Path $log -Value "# C1 gate 2 (the reduced-precision product), win-cuda: $gpu; $cpu; $rv; Windows $([Environment]::OSVersion.Version) x86_64; $(Get-Date -Format o)"
    Say '# every process on 2 logical CPUs at below-normal priority; half results are compared with f32 by a tolerance, never byte for byte'
    function G([string]$name, [string[]]$a) {
        & "$E\half_gemm_gate.exe" --device cuda --out "$work\$name" @a *> "$work\$name.out"
        return ($LASTEXITCODE -eq 0)
    }
    function Fin([string]$name) { (Get-Content "$work\$name\final.txt")[0] }
    function Same([string]$x, [string]$y) { (Get-FileHash "$x").Hash -eq (Get-FileHash "$y").Hash }
    function Losses([string]$file) {
        $h = @{}
        foreach ($l in (Get-Content $file)) { if ($l -match '^loss (\d+) (\S+)$') { $h[[int]$Matches[1]] = [double]$Matches[2] } }
        $h
    }
    function Mean($h, [int]$lo, [int]$hi) { $s = 0.0; $n = 0; foreach ($k in $lo..$hi) { if ($h.ContainsKey($k)) { $s += $h[$k]; $n++ } }; '{0:f4}' -f ($s / [math]::Max($n, 1)) }

    Say '## 1. Run to run (200 steps, two processes)'
    if (-not ((G 'a' @('--precision', 'half', '--steps', '200')) -and (G 'b' @('--precision', 'half', '--steps', '200')))) { Say 'runs: FAILED'; $script:fail = 1 }
    $r = if (Same "$work\a\steps" "$work\b\steps") { 'every per-step weight hash the same (200 steps)' } else { $script:fail = 1; 'DIFFERENT' }
    Say "half, run A and run B: final weights sha256 $(Fin 'a') and $(Fin 'b'); $r"
    $key = (Get-Content "$work\a.out" | Where-Object { $_ -like 'half_gemm_gate:*' } | Select-Object -First 1) -replace '.* key ', ''
    Say "key: $([regex]::Replace($key, 'kernels-[0-9a-f]+', 'kernels-<hex>'))"

    Say '## 2. Restart'
    if (-not ((G 'r' @('--precision', 'half', '--steps', '107', '--ckpt-every', '50')) -and (G 'r' @('--precision', 'half', '--steps', '200', '--ckpt-every', '50', '--resume')))) { Say 'restart: FAILED'; $script:fail = 1 }
    $uniq = Get-Content "$work\r\steps" | Sort-Object -Unique | Sort-Object { [int]($_ -split ' ')[0] }
    $r = if (($uniq -join "`n") -eq ((Get-Content "$work\a\steps") -join "`n") -and (Select-String -Quiet 'resumed at step 100' "$work\r.out")) { 'every per-step hash the uninterrupted run''s' } else { $script:fail = 1; 'DIFFERENT' }
    Say "half, 107 steps with checkpoints every 50, a new process resuming from step 100 to 200: final weights sha256 $(Fin 'r'); $r"
    if (G 'resume-f32' @('--precision', 'f32', '--steps', '200', '--ckpt-dir', "$work\r\ckpt", '--resume')) { Say 'f32 resume of the half checkpoint: NOT REFUSED'; $script:fail = 1 } else { Say "f32 resume of the half checkpoint: refused ($(if (Select-String -Quiet 'resume refused' "$work\resume-f32.out") { 'resume refused: another key' } else { 'another error' }))" }

    Say '## 3. A larger decoder, run twice (width 256, 4 blocks, MLP 1024, batch 32, 20 steps)'
    $big = @('--precision', 'half', '--steps', '20', '--width', '256', '--blocks', '4', '--mlp', '1024', '--batch', '32', '--lr', '0.003')
    if (-not ((G 'big1' $big) -and (G 'big2' $big))) { Say 'large: FAILED'; $script:fail = 1 }
    $r = if (Same "$work\big1\steps" "$work\big2\steps") { 'every per-step hash the same' } else { $script:fail = 1; 'DIFFERENT' }
    Say "half: final weights sha256 $(Fin 'big1'); $r"

    Say '## 4. The loss against f32 (the 200 steps of 1)'
    if (-not (G 'f' @('--precision', 'f32', '--steps', '200'))) { Say 'f32: FAILED'; $script:fail = 1 }
    $x = Losses "$work\f\losses"; $y = Losses "$work\a\losses"
    $m = 0.0; $s = 0; $rm = 0.0; $m10 = 0.0
    foreach ($k in $x.Keys) { if ($y.ContainsKey($k)) { $d = [math]::Abs($y[$k] - $x[$k]); $rr = $d / [math]::Abs($x[$k]); if ($d -ge $m) { $m = $d; $s = $k }; if ($rr -gt $rm) { $rm = $rr }; if ($k -le 10 -and $d -gt $m10) { $m10 = $d } } }
    Say ("half against f32: largest |loss difference| {0:e3} at step {1} (relative {2:e3}) over 200 steps; over steps 1-10 {3:e3}" -f $m, $s, $rm, $m10)
    Say "mean loss over steps 1-10: f32 $(Mean $x 1 10), half $(Mean $y 1 10); over steps 191-200: f32 $(Mean $x 191 200), half $(Mean $y 191 200)"
    foreach ($k in @(1, 40, 80, 120, 160, 200)) { Say ("  step {0,3}: f32 {1:f6} half {2:f6} difference {3:e3}" -f $k, $x[$k], $y[$k], ($y[$k] - $x[$k])) }

    Say '## 5. Wall time of training (width 512, 4 blocks, MLP 2048, batch 64, 10 steps, no per-step hash; 3 runs each, in rotation)'
    $times = @{ 'f32' = @(); 'half' = @() }
    foreach ($rep in 1..3) {
        foreach ($p in @('f32', 'half')) {
            if (G "t-$p-$rep" @('--precision', $p, '--steps', '10', '--width', '512', '--blocks', '4', '--mlp', '2048', '--batch', '64', '--lr', '0.001', '--no-hash')) {
                $w = [double]((Get-Content "$work\t-$p-$rep.out" | Where-Object { $_ -like 'wall *' }) -replace '^wall ([0-9.]+) seconds.*', '$1')
                $times[$p] += $w
                Say "$p run $rep : $w s"
            } else { Say "timing $p $rep : FAILED"; $script:fail = 1 }
        }
    }
    foreach ($p in @('f32', 'half')) { Say "$p : median $(($times[$p] | Sort-Object)[1]) s" }
    Say "parameters per model: $((Get-Content "$work\t-half-1.out" | Select-Object -First 1) -replace '^half_gemm_gate: (\d+) parameters.*', '$1')"

    Say '## library tests and throughput (the lib test binary: half_matmul, ignored included)'
    # The library's test binary: the newest <lib name>-*.exe in target\release\deps, the lib name
    # read from Cargo.toml's [lib] section.
    $toml = Get-Content "$root\Cargo.toml"
    $at = [array]::IndexOf($toml, '[lib]')
    $lib = ($toml[($at + 1)..($toml.Count - 1)] | Where-Object { $_ -match '^name = "(.+)"' } | Select-Object -First 1) -replace '^name = "(.+)"', '$1'
    $exe = (Get-ChildItem "$root\target\release\deps\$lib-*.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    $out = & $exe 'half_matmul' --include-ignored --test-threads=1 --nocapture 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { $script:fail = 1 }
    $out | Where-Object { $_ -match 'half matmul|matmul throughput|test result' } | ForEach-Object { $_ -replace '^test \S+ \.\.\. ', '' } | Sort-Object -Unique | ForEach-Object { Say $_ }
    $status = if ($script:fail -eq 0) { 'all checks passed' } else { 'FAILURES' }
    Say "# ended $(Get-Date -Format o), $status"
    Get-Content $log
    exit $script:fail
}

if ($Part -eq 'off') {
    # Gate 1 on CUDA: E2's ddp_gate references (their log is compared with E2's recorded one) and the
    # single-device differential against the 0.5.0 tree in $Ref.
    $log = "$logs\win-cuda-off.log"
    Set-Content -Path $log -Value "# C1 gate 1 (the reduced-precision product off) on Windows: $gpu; $cpu; $rv; $(Get-Date -Format o)"
    $e2 = "$root\doc\gates\e2"
    $recorded = Get-Content "$e2\logs\win-ddp-gate.log"
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$e2\win_gates.ps1" -Part ddpgate *> "$root\runs\c1-off-ddpgate.out"
    $now = Get-Content "$e2\logs\win-ddp-gate.log"
    $pick = { param($lines) $lines | ForEach-Object { if ($_ -match '^([^#].*?)( \(.*?\))?: final weights sha256 ([0-9a-f]{64})') { "$($Matches[1]) $($Matches[3])" } } }
    $a = & $pick $recorded; $b = & $pick $now
    $same = (($a -join "`n") -eq ($b -join "`n")) -and $a.Count -gt 0
    if (-not $same) { $script:fail = 1 }
    Add-Content $log "ddp_gate references (doc\gates\e2\win_gates.ps1 -Part ddpgate) against 0.5.0's recorded log: $($b.Count) final-weights lines, $(if ($same) { 'all identical' } else { 'DIFFERENT' })"
    $now | Where-Object { $_ -match 'final weights sha256|resume|nodes' } | ForEach-Object { Add-Content $log "  $_" }
    # Restore the recorded E2 log in this copy (the run above rewrote it).
    Set-Content "$e2\logs\win-ddp-gate.log" $recorded
    if ($Ref -ne '') {
        & powershell -NoProfile -ExecutionPolicy Bypass -File "$e2\win_gates.ps1" -Part diff -Ref $Ref *> "$root\runs\c1-off-diff.out"
        $d = Get-Content "$e2\logs\win-single-device.log"
        Add-Content $log "single-device differential (doc\gates\e2\win_gates.ps1 -Part diff, CPU and CUDA) against the 0.5.0 tree:"
        $d | Where-Object { $_ -match 'identical|^#' } | ForEach-Object { Add-Content $log "  $_" }
        if (-not ($d | Where-Object { $_ -match 'identical: yes' })) { $script:fail = 1 }
    }
    $status = if ($script:fail -eq 0) { 'all identical' } else { 'FAILURES' }
    Add-Content $log "# ended $(Get-Date -Format o), $status"
    Get-Content $log
    exit $script:fail
}
