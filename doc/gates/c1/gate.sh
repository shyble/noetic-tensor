#!/bin/sh
# C1 gate 2: the reduced-precision product (`MatmulPrecision::Half`) on one device, through
# examples/half_gemm_gate.rs, and its library tests (tolerance on fixed shapes, run to run in a
# process, the key) and throughput. Kind (ii): the half results are compared with the f32 ones
# by a stated tolerance, never byte for byte, and never pooled with them.
#
# 1. Run to run: two runs of 200 steps (the gate's decoder: width 32, 2 blocks, MLP 64, batch 2),
#    each its own process: every per-step weight hash the same.
# 2. Restart: a run of 107 steps with checkpoints every 50, then a new process resuming from
#    step 100 to 200: every per-step hash the uninterrupted run's. An f32 resume of the half
#    checkpoint is refused (the key differs).
# 3. A larger decoder (width 256, 4 blocks, MLP 1024, batch 32: full 64×64 tiles), 20 steps,
#    run twice: the same hashes.
# 4. The loss against f32 over the 200 steps of 1 (a tolerance).
# 5. Wall time of training (width 512, 4 blocks, MLP 2048, batch 64, 10 steps), f32 against
#    half, 3 runs each in rotation; and the library's throughput test.
#
#   cargo build --release --features metal --examples
#   doc/gates/c1/gate.sh mac-metal metal metal     # label, device, cargo feature
set -eu
label=$1
device=$2
feature=$3
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
E="$root/target/release/examples"
work="$root/runs/c1-gate-$label"
log="$here/logs/$label-half.log"
rm -rf "$work"
mkdir -p "$work" "$here/logs"
if command -v taskpolicy > /dev/null; then prio="taskpolicy -b"; else prio="nice -n 10"; fi
fail=0
say() { echo "$*" >> "$log"; }
{
  echo "# C1 gate 2 (the reduced-precision product), $label: device $device"
  echo "# $(rustc --version | sed 's/ (.*//'); $(uname -sm); $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# half results are compared with f32 by a tolerance, never byte for byte"
} > "$log"
G() { name=$1; shift; $prio "$E/half_gemm_gate" --device "$device" --out "$work/$name" "$@" > "$work/$name.out" 2>&1; }
fin() { head -1 "$work/$1/final.txt"; }
key() { sed -n 's/.* key \(.*\)$/\1/p' "$work/$1.out" | head -1 | sed 's/kernels-[0-9a-f]*/kernels-<hex>/g'; }

say "## 1. Run to run (200 steps, two processes)"
G a --precision half --steps 200 && G b --precision half --steps 200 || { say "runs: FAILED"; fail=1; }
if cmp -s "$work/a/steps" "$work/b/steps"; then r="every per-step weight hash the same (200 steps)"; else r="DIFFERENT"; fail=1; fi
say "half, run A and run B: final weights sha256 $(fin a) and $(fin b); $r"
say "key: $(key a)"

say "## 2. Restart"
G r --precision half --steps 107 --ckpt-every 50 && G r --precision half --steps 200 --ckpt-every 50 --resume || { say "restart: FAILED"; fail=1; }
grep -q "resumed at step 100" "$work/r.out" || { say "restart: did not resume at step 100"; fail=1; }
# Steps 101-107 ran twice (before and after the restart): one line per step means the same hash.
if sort -u "$work/r/steps" | sort -n | cmp -s - "$work/a/steps"; then r="every per-step hash the uninterrupted run's"; else r="DIFFERENT"; fail=1; fi
say "half, 107 steps with checkpoints every 50, a new process resuming from step 100 to 200: final weights sha256 $(fin r); $r"
if G resume-f32 --precision f32 --steps 200 --ckpt-dir "$work/r/ckpt" --resume; then say "f32 resume of the half checkpoint: NOT REFUSED"; fail=1; else say "f32 resume of the half checkpoint: refused ($(grep -o 'taken under the key[^(]*([^)]*)' "$work/resume-f32.out" | sed 's/.*(\(.*\))/\1/'))"; fi

say "## 3. A larger decoder, run twice (width 256, 4 blocks, MLP 1024, batch 32, 20 steps)"
G big1 --precision half --steps 20 --width 256 --blocks 4 --mlp 1024 --batch 32 --lr 0.003 && G big2 --precision half --steps 20 --width 256 --blocks 4 --mlp 1024 --batch 32 --lr 0.003 || { say "large: FAILED"; fail=1; }
if cmp -s "$work/big1/steps" "$work/big2/steps"; then r="every per-step hash the same"; else r="DIFFERENT"; fail=1; fi
say "half: final weights sha256 $(fin big1); $r"

say "## 4. The loss against f32 (the 200 steps of 1)"
G f --precision f32 --steps 200 || { say "f32: FAILED"; fail=1; }
awk 'NR == FNR { a[$2] = $3; next } ($2 in a) { d = $3 - a[$2]; if (d < 0) d = -d; r = d / (a[$2] < 0 ? -a[$2] : a[$2]); if (d >= m) { m = d; s = $2 }; if (r > rm) rm = r; if ($2 <= 10 && d > m10) m10 = d; n++ }
     END { printf "largest |loss difference| %.3e at step %d (relative %.3e) over %d steps; over steps 1-10 %.3e\n", m, s, rm, n, m10 }' "$work/f/losses" "$work/a/losses" | sed 's/^/half against f32: /' >> "$log"
mean() { awk -v lo="$2" -v hi="$3" '$2 >= lo && $2 <= hi { s += $3; n++ } END { printf "%.4f", s / n }' "$1"; }
say "mean loss over steps 1-10: f32 $(mean "$work/f/losses" 1 10), half $(mean "$work/a/losses" 1 10); over steps 191-200: f32 $(mean "$work/f/losses" 191 200), half $(mean "$work/a/losses" 191 200)"
awk 'NR == FNR { a[$2] = $3; next } ($2 in a) && ($2 == 1 || $2 % 40 == 0) { printf "  step %3d: f32 %.6f half %.6f difference %+.3e\n", $2, a[$2], $3, $3 - a[$2] }' "$work/f/losses" "$work/a/losses" >> "$log"

say "## 5. Wall time of training (width 512, 4 blocks, MLP 2048, batch 64, 10 steps; 3 runs each, in rotation; background priority)"
for rep in 1 2 3; do
  for p in f32 half; do
    G "t-$p-$rep" --precision "$p" --steps 10 --width 512 --blocks 4 --mlp 2048 --batch 64 --lr 0.001 || { say "timing $p $rep: FAILED"; fail=1; continue; }
    say "$p run $rep: $(sed -n 's/^wall \([0-9.]*\) seconds.*/\1/p' "$work/t-$p-$rep.out") s"
  done
done
for p in f32 half; do
  say "$p: median $(grep "^$p run " "$log" | sed 's/.*: \([0-9.]*\) s$/\1/' | sort -n | sed -n 2p) s"
done
say "parameters per model: $(sed -n 's/^half_gemm_gate: \([0-9]*\) parameters.*/\1/p' "$work/t-half-1.out")"
say "## library tests and throughput (cargo test --features $feature --lib half_matmul -- --include-ignored)"
(cd "$root" && CARGO_BUILD_JOBS=1 $prio cargo test --release -j 1 --features "$feature" --lib half_matmul -- --include-ignored --test-threads 1 --nocapture 2>&1) | grep -E "half matmul|matmul throughput|test result" | sed 's/\.\.\. //' | sort -u >> "$log" || fail=1
say "# ended $(date -u +%Y-%m-%dT%H:%M:%SZ), $([ $fail -eq 0 ] && echo 'all checks passed' || echo 'FAILURES')"
cat "$log"
exit $fail
