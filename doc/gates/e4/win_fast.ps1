# E4 gate 2 on Windows (the PowerShell counterpart of fast_gate.sh's loss comparison, for world
# size 4): examples/ddp_gate.rs in the fast mode against the deterministic mode on the CPU
# reference backend and on CUDA (every rank sharing the machine's one GPU). Run from the source
# tree after `cargo +1.94.1 build --release -j 2 --features cuda --examples`:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File doc\gates\e4\win_fast.ps1 [-Devices cpu,cuda]
#
# The script, and every process it starts, runs on 2 logical CPUs at below-normal priority (the
# machine is shared): 4 ranks share those 2 CPUs. Logs go to doc\gates\e4\logs\win-fast.log. The
# fast mode is not reproducible bit for bit: its losses are compared with a tolerance.
param([string]$Devices = 'cpu,cuda')
$ErrorActionPreference = 'Continue'
$me = [System.Diagnostics.Process]::GetCurrentProcess()
$me.ProcessorAffinity = [IntPtr]3
$me.PriorityClass = 'BelowNormal'
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$logs = "$PSScriptRoot\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
Set-Location $root
$E = "$root\target\release\examples"
$work = "$root\runs\e4-win-fast"
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $work | Out-Null
$log = "$logs\win-fast.log"
$script:fail = 0
function Say([string]$s) { Add-Content -Path $log -Value $s }
$rv = (& rustc +1.94.1 --version) -replace ' [(].*', ''
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
Set-Content -Path $log -Value "# E4 gate 2 on Windows (the fast mode): $rv; Windows $([Environment]::OSVersion.Version) x86_64; $cpu; $(Get-Date -Format o)"
Say "# every process on 2 logical CPUs (affinity mask 3) at below-normal priority; ranks share the device; fast-mode results are compared with a tolerance, never byte for byte"

function Final([string]$dir) { ((Get-Content "$dir\record.txt") | Where-Object { $_ -like 'final *' }) -replace '^final ', '' }
function Record([string]$dir) {
    $l = (Get-Content "$dir\record.txt")[0]
    $l = [regex]::Replace($l, 'libm=[0-9a-f]+', 'libm=<sha256>')
    [regex]::Replace($l, 'kernels-[0-9a-f]+', 'kernels-<hex>')
}
function Losses([string]$file) {
    $h = @{}
    foreach ($l in (Get-Content $file)) { if ($l -match '^loss (\d+) (\S+)$') { $h[[int]$Matches[1]] = [double]$Matches[2] } }
    $h
}
function LossDiff([string]$a, [string]$b) {
    $x = Losses $a; $y = Losses $b
    $m = 0.0; $s = 0; $rm = 0.0; $n = 0
    foreach ($k in $x.Keys) {
        if ($y.ContainsKey($k)) {
            $d = [math]::Abs($y[$k] - $x[$k]); $r = $d / [math]::Abs($x[$k]); $n++
            if ($d -ge $m) { $m = $d; $s = $k }
            if ($r -gt $rm) { $rm = $r }
        }
    }
    "largest |loss difference| {0:e3} at step {1}, largest relative {2:e3}, over {3} steps" -f $m, $s, $rm, $n
}
# Ref name device args: one process; Ranks name device W args: W processes under the launcher.
function Ref([string]$name, [string]$device, [string[]]$a) {
    $out = "$work\$name"
    & "$E\ddp_gate.exe" --device $device --losses --out $out @a *> "$out.out"
    if ($LASTEXITCODE -ne 0) { Say "$name : FAILED"; $script:fail = 1; return $false }
    Get-Content "$out.out" | Where-Object { $_ -like 'loss *' } | Set-Content "$out\losses"
    $true
}
function Ranks([string]$name, [string]$device, [int]$w, [string[]]$a) {
    $out = "$work\$name"
    & "$E\launch.exe" --nproc-per-node $w --log-dir "$out\logs" -- "$E\ddp_gate.exe" --device $device --ranks-per-device $w --losses --out $out @a *> "$out.out"
    if ($LASTEXITCODE -ne 0) { Say "$name : FAILED"; $script:fail = 1; return $false }
    Get-Content "$out\logs\rank-0.log" | Where-Object { $_ -like 'loss *' } | Set-Content "$out\losses"
    $true
}
function Synced([string]$dir) {
    if ((Get-Content "$dir\record.txt")[0] -match 'weight_checks=\d+ ok') { 'every per-step weight check passed on every rank' } else { $script:fail = 1; 'WEIGHT CHECKS FAILED' }
}

foreach ($device in $Devices.Split(',')) {
    Say "## $device : the gate's model, 12 steps, M 4"
    if (-not (Ref "$device-det-ref" $device @('--micro', '4'))) { continue }
    Say "deterministic M 4, 1 process: final weights sha256 $(Final "$work\$device-det-ref")"
    foreach ($w in @(2, 4)) {
        if (Ranks "$device-det-w$w" $device $w @('--micro', '4')) {
            $eq = if ((Final "$work\$device-det-w$w") -eq (Final "$work\$device-det-ref")) { 'equal to 1 process, as the deterministic mode promises' } else { $script:fail = 1; 'NOT EQUAL TO 1 PROCESS' }
            Say "deterministic M 4 W $w : final weights sha256 $(Final "$work\$device-det-w$w") ($eq)"
        }
        foreach ($kb in @('25600', '16')) {
            $name = "$device-fast-w$w-b$kb"
            if (Ranks $name $device $w @('--micro', '4', '--fast', '--bucket-kb', $kb)) {
                $same = if ((Final "$work\$name") -eq (Final "$work\$device-det-ref")) { 'the same bits as the deterministic run' } else { 'other bits than the deterministic run' }
                Say "fast M 4 W $w buckets $kb KiB: final weights sha256 $(Final "$work\$name") ($same); against the deterministic losses: $(LossDiff "$work\$device-det-ref\losses" "$work\$name\losses"); $(Synced "$work\$name")"
            }
        }
    }
    Say "fast-mode record ($device, M 4 W 4, 16 KiB buckets): $(Record "$work\$device-fast-w4-b16")"
    Say "## $device : the gate's model, 200 steps, M 4"
    if ((Ref "$device-det-long" $device @('--micro', '4', '--steps', '200'))) {
        foreach ($w in @(2, 4)) {
            $name = "$device-fast-long-w$w"
            if (Ranks $name $device $w @('--micro', '4', '--steps', '200', '--fast')) {
                Say "fast W $w, 200 steps: final weights sha256 $(Final "$work\$name"); against the deterministic losses: $(LossDiff "$work\$device-det-long\losses" "$work\$name\losses"); $(Synced "$work\$name")"
            }
        }
        $d = Losses "$work\$device-det-long\losses"; $f = Losses "$work\$device-fast-long-w4\losses"
        foreach ($k in @(1, 50, 100, 150, 200)) { if ($d.ContainsKey($k) -and $f.ContainsKey($k)) { Say ("  step {0,3}: deterministic {1:f6} fast W 4 {2:f6} difference {3:e3}" -f $k, $d[$k], $f[$k], ($f[$k] - $d[$k])) } }
    }
}
$status = if ($script:fail -eq 0) { 'all checks passed' } else { 'FAILURES' }
Say "# ended $(Get-Date -Format o), $status"
Get-Content $log
exit $script:fail
