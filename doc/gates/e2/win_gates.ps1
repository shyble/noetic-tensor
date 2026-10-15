# The E2 gates on Windows (CUDA build), the PowerShell counterpart of run_gates.sh,
# ddp_gate_refs.sh and single_device_diff.sh (no POSIX shell there). Run from the source tree
# after `cargo +1.94.1 test --release -j 1 --features cuda --lib --no-run` and
# `cargo +1.94.1 build --release -j 1 --features cuda --examples`:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File doc\gates\e2\win_gates.ps1 -Part gates      # every nn::dist test, CPU and CUDA
#   powershell -NoProfile -ExecutionPolicy Bypass -File doc\gates\e2\win_gates.ps1 -Part ddpgate    # examples/ddp_gate.rs on CUDA and the CPU
#   powershell -NoProfile -ExecutionPolicy Bypass -File doc\gates\e2\win_gates.ps1 -Part diff -Ref <folder of the reference tree>
#
# Logs go to doc\gates\e2\logs\win-*.log, with local paths and short hexadecimal ids replaced by
# <path> and <hex> (the sha256 values are kept whole).
param([string]$Part = 'gates', [string]$Ref = '')
$ErrorActionPreference = 'Continue'
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$logs = "$PSScriptRoot\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
Set-Location $root
function Clean([string[]]$lines) {
    $lines | Where-Object { $_ -notmatch '^\s*(Compiling|Finished|Running|Blocking|Fresh|Doc-tests)' } | ForEach-Object {
        $l = $_.Replace($root, '.')
        $l = [regex]::Replace($l, '[A-Za-z]:\\U[s]ers\\\S*', '<path>')
        [regex]::Replace($l, '\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}\b', '<hex>')
    }
}
function Header([string]$what) {
    $rv = (& rustc +1.94.1 --version) -replace ' [(].*', ''
    $os = [Environment]::OSVersion.Version
    $now = Get-Date -Format o
    "# $what"
    "# $rv; Windows $os x86_64; $now"
}
$E = "$root\target\release\examples"

if ($Part -eq 'gates') {
    # The library's test executable: the newest <lib name>-*.exe in target\release\deps (the lib
    # name is the second `name =` of Cargo.toml, under [lib]).
    $lib = (Select-String -Path "$root\Cargo.toml" -Pattern '^name = "(.+)"' | Select-Object -Last 1).Matches[0].Groups[1].Value
    $exe = (Get-ChildItem "$root\target\release\deps\$lib-*.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    Remove-Item Env:\DIST_TEST_MAX_PROCS -ErrorAction SilentlyContinue
    $out = & $exe 'nn::dist' --test-threads=1 --nocapture 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    (@(Header 'E2 gates, win-cuda (every nn::dist test: CPU gates and CUDA gates, no process cap)') + (Clean $out) + "# exit $code") | Set-Content "$logs\win-cuda.log"
    exit $code
}

if ($Part -eq 'ddpgate') {
    $work = "$root\runs\ddp-gate-win"
    if (Test-Path $work) { Remove-Item -Recurse -Force $work }
    New-Item -ItemType Directory -Force $work | Out-Null
    $log = @(Header 'examples/ddp_gate.rs references, win: ranks sharing the device (CUDA: the one GPU)')
    $fail = 0
    function Final($dir) { ((Get-Content "$dir\record.txt") | Where-Object { $_ -like 'final *' }) -replace '^final ', '' }
    foreach ($device in 'cuda', 'cpu') {
        $configs = [ordered]@{ 'gate' = @(); 'absent' = @('--absent'); 'idle-after' = @('--idle', 'after'); 'idle-torch' = @('--idle', 'torch'); 'clip' = @('--clip', '1.0'); 'sgd' = @('--opt', 'sgd') }
        if ($device -eq 'cpu') { $configs = [ordered]@{ 'gate' = @() } }
        foreach ($name in $configs.Keys) {
            $opts = $configs[$name]
            foreach ($micro in 2, 4) {
                $ref = "$work\$device-$name-m$micro-ref"
                & "$E\ddp_gate.exe" --device $device --micro $micro @opts --out $ref *> "$ref.out"
                if ($LASTEXITCODE -ne 0) { $log += "$device $name M $micro reference: FAILED"; $fail = 1; continue }
                $log += "$device $name M $micro reference (1 process, $micro accumulation micro-steps): final weights sha256 $(Final $ref)"
                foreach ($w in 2, 4) {
                    if ($micro % $w -ne 0) { continue }
                    $run = "$work\$device-$name-m$micro-w$w"
                    & "$E\launch.exe" --nproc-per-node $w --log-dir "$run\logs" -- "$E\ddp_gate.exe" --device $device --ranks-per-device $w --micro $micro @opts --out $run *> "$run.out"
                    if ($LASTEXITCODE -ne 0) { $log += "$device $name M $micro W ${w}: FAILED"; $fail = 1; continue }
                    $same = 'yes'
                    $want = Get-Content "$ref\rank-0.steps" -Raw
                    foreach ($r in 0..($w - 1)) { if ((Get-Content "$run\rank-$r.steps" -Raw) -ne $want) { $same = 'NO' } }
                    if ((Final $run) -ne (Final $ref)) { $same = 'NO' }
                    if ($same -ne 'yes') { $fail = 1 }
                    $log += "$device $name M $micro W $w ($w ranks sharing the device): final weights sha256 $(Final $run); every per-step hash and the final weights equal the reference: $same"
                }
            }
        }
    }
    # Resume on CUDA: 7 steps with checkpoints every 3, relaunched to step 12 from step 6.
    $run = "$work\cuda-resume"
    & "$E\launch.exe" --nproc-per-node 2 --log-dir "$run\logs1" -- "$E\ddp_gate.exe" --device cuda --ranks-per-device 2 --steps 7 --ckpt-every 3 --out $run *> "$run.out"
    & "$E\launch.exe" --nproc-per-node 2 --log-dir "$run\logs2" -- "$E\ddp_gate.exe" --device cuda --ranks-per-device 2 --resume --out $run *>> "$run.out"
    $steps = (Get-Content "$run\rank-1.steps" | Sort-Object -Unique | Sort-Object { [int]($_ -split ' ')[0] }) -join "`n"
    $want = ((Get-Content "$work\cuda-gate-m2-ref\rank-0.steps") -join "`n")
    $same = if (($steps -eq $want) -and ((Final $run) -eq (Final "$work\cuda-gate-m2-ref")) -and (Select-String -Quiet 'resumed at step 6' "$run\logs2\rank-0.log")) { 'yes' } else { 'NO' }
    if ($same -ne 'yes') { $fail = 1 }
    $log += "cuda resume W 2 M 2 (checkpoint of step 6, relaunched to step 12): final weights sha256 $(Final $run); equals the reference and every per-step hash: $same"
    $rec = (Get-Content "$work\cuda-gate-m2-w2\logs\rank-0.log" | Where-Object { $_ -like 'record *' } | Select-Object -First 1)
    $rec = $rec -replace 'libm=[0-9a-f]+', 'libm=<sha256>' -replace 'kernels-[0-9a-f]+', 'kernels-<hex>'
    $log += "run record: $rec"
    $log += "# ended $(Get-Date -Format o), $(if ($fail -eq 0) { 'all identical' } else { 'FAILURES' })"
    $log | Set-Content "$logs\win-ddp-gate.log"
    exit $fail
}

if ($Part -eq 'diff') {
    # The single-device probe and the copy task, in the reference tree ($Ref, built there the same
    # way, with doc\gates\e2\single_device_probe.rs copied into its examples) and in this one.
    $log = @(Header "E2 single-device differential, win: the reference tree against this tree")
    $same = 'yes'
    foreach ($device in 'cpu', 'cuda') {
        $o = @{}
        foreach ($t in @{ 'ref' = $Ref; 'new' = $root }.GetEnumerator()) {
            $ex = "$($t.Value)\target\release\examples"
            $lines = @(& "$ex\single_device_probe.exe" $device 2>&1 | ForEach-Object { "$_" })
            $lines += 'copy_task:'
            if ($device -eq 'cpu') { $lines += @(& "$ex\copy_task.exe" 2>&1 | ForEach-Object { "$_" }) } else { $lines += @(& "$ex\copy_task.exe" $device 2>&1 | ForEach-Object { "$_" }) }
            $o[$t.Key] = $lines
        }
        $log += "## $device, reference"; $log += $o['ref']
        $log += "## $device, this tree"; $log += $o['new']
        $id = (($o['ref'] -join "`n") -eq ($o['new'] -join "`n"))
        if (-not $id) { $same = 'NO' }
        $log += "## $device identical: $(if ($id) { 'yes' } else { 'NO' })"
    }
    (Clean $log) | Set-Content "$logs\win-single-device.log"
    if ($same -ne 'yes') { exit 1 }
    exit 0
}
