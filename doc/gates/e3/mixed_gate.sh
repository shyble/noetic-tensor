#!/bin/sh
# E3 gates 2 and 3: mixed precision (examples/ddp_gate.rs --mixed: the products at
# MatmulPrecision::Half, f32 vars and AdamW state, a static loss scale) on one GPU, ranks sharing it.
# Mixed results are compared with f32 by a stated tolerance, never byte for byte.
#
# Gate 2, run to run and restart, per key (the gate's model, 12 steps, M 2 global micro-steps):
#   one process; W 2 data-parallel; W 2 sharded (stage 2, flat); each run twice (separate
#   launches) with every per-step hash the same, and resumed from a checkpoint of step 6 in a new
#   launch with every per-step hash the uninterrupted run's. W 2 is also compared with one process
#   (the deterministic mode's identity, within the mixed key). The fast mode (W 2, M 4) is run
#   twice and keyed apart (an observation: the fast mode makes no such promise). An f32 resume of
#   a mixed checkpoint is refused; a job with one rank mixed and one at f32 is refused.
# Gate 3, the loss against f32 (one process, M 4, 200 steps): mixed with loss scales 2^0 and 2^8.
#
#   cargo build --release --features metal --examples
#   doc/gates/e3/mixed_gate.sh mac-metal metal
set -eu
label=$1
device=$2
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
E="$root/target/release/examples"
work="$root/runs/e3-$label"
log="$here/logs/$label-mixed.log"
rm -rf "$work"
mkdir -p "$work" "$here/logs"
if command -v taskpolicy > /dev/null; then prio="taskpolicy -b"; else prio="nice -n 10"; fi
fail=0
say() { echo "$*" >> "$log"; }
{
  echo "# E3 gates 2 and 3 (mixed precision), $label: device $device, ranks sharing the device, at most 2 processes at once"
  echo "# $(rustc --version | sed 's/ (.*//'); $(uname -sm); $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# mixed results are compared with f32 by a tolerance, never byte for byte"
} > "$log"
final() { sed -n 's/^final //p' "$1/record.txt"; }
rec() { head -1 "$1/record.txt" | sed 's/libm=[0-9a-f]*/libm=<sha256>/; s/kernels-[0-9a-f]*/kernels-<hex>/g; s/metal-Apple_[A-Za-z0-9_]*-mathmode/metal-<the GPU name>-mathmode/'; }
# one NAME ARGS...: one process; ranks NAME W ARGS...: W processes under the launcher.
# (POSIX functions share the caller's variables: these use their own names.)
one() { o_n=$1; shift; $prio "$E/ddp_gate" --device "$device" --out "$work/$o_n" "$@" > "$work/$o_n.out" 2>&1; }
ranks() { r_n=$1; r_w=$2; shift 2; $prio "$E/launch" --nproc-per-node "$r_w" --log-dir "$work/$r_n/logs" -- "$E/ddp_gate" --device "$device" --ranks-per-device "$r_w" --out "$work/$r_n" "$@" > "$work/$r_n.out" 2>&1; }
same() { cmp -s "$1/rank-0.steps" "$2/rank-0.steps" && [ "$(final "$1")" = "$(final "$2")" ]; }
check() { if "$@"; then echo yes; else fail=1; echo NO; fi; }

say "## Gate 2: run to run and restart, per key (12 steps, M 2)"
for cfg in "one:1:" "ddp:2:" "zero:2:--zero 2:flat"; do
  name=${cfg%%:*}; rest=${cfg#*:}; w=${rest%%:*}; extra=${rest#*:}
  for run in a b; do
    # shellcheck disable=SC2086
    if [ "$w" = 1 ]; then one "$name-$run" --micro 2 --mixed $extra; else ranks "$name-$run" "$w" --micro 2 --mixed $extra; fi || { say "$name $run: FAILED"; fail=1; }
  done
  say "$name (W $w$( [ -n "$extra" ] && echo ", $extra")): final weights sha256 $(final "$work/$name-a"); run twice, every per-step hash and the final weights the same: $(check same "$work/$name-a" "$work/$name-b")"
  # Restart: 7 steps with checkpoints every 3, then a new launch resuming from step 6 to 12.
  r="$name-resume"
  # shellcheck disable=SC2086
  if [ "$w" = 1 ]; then one "$r" --micro 2 --mixed $extra --steps 7 --ckpt-every 3 && one "$r" --micro 2 --mixed $extra --resume; else ranks "$r" "$w" --micro 2 --mixed $extra --steps 7 --ckpt-every 3 && ranks "$r-2" "$w" --micro 2 --mixed $extra --resume --out "$work/$r"; fi || { say "$name resume: FAILED"; fail=1; }
  ok=yes
  sort -u "$work/$r/rank-0.steps" | sort -n | cmp -s - "$work/$name-a/rank-0.steps" || ok=NO
  [ "$(final "$work/$r")" = "$(final "$work/$name-a")" ] || ok=NO
  [ $ok = yes ] || fail=1
  say "$name: 7 steps with checkpoints every 3, a new launch resuming from step 6 to 12: every per-step hash and the final weights the uninterrupted run's: $ok"
done
say "W 2 data-parallel against one process (the deterministic mode's identity within the mixed key): $(check same "$work/ddp-a" "$work/one-a")"
say "W 2 sharded (stage 2, flat) against W 2 data-parallel: $(check same "$work/zero-a" "$work/ddp-a")"
say "key and record (one process): $(rec "$work/one-a")"
say "key and record (sharded): $(rec "$work/zero-a")"
ranks fast-a 2 --micro 4 --mixed --fast && ranks fast-b 2 --micro 4 --mixed --fast || { say "fast: FAILED"; fail=1; }
say "fast mode, W 2, M 4, run twice: the same per-step hashes: $(if same "$work/fast-a" "$work/fast-b"; then echo yes; else echo no; fi) (an observation; the fast mode makes no such promise); key: $(rec "$work/fast-a")"
# Refusals: an f32 resume of the mixed checkpoint; a job with one rank mixed.
if one f32-resume --micro 2 --ckpt-dir "$work/one-resume/ckpt" --resume; then say "f32 resume of the mixed checkpoint: NOT REFUSED"; fail=1; else say "f32 resume of the mixed checkpoint: refused ($(grep -o '([^()]*resume refused)' "$work/f32-resume.out" | head -1))"; fi
port=$((20000 + $$ % 20000))
for n in 0 1; do
  if [ $n -eq 0 ]; then m="--mixed"; else m=""; fi
  # shellcheck disable=SC2086
  $prio "$E/launch" --nproc-per-node 1 --nnodes 2 --node-rank $n --master-addr 127.0.0.1 --master-port $port --no-auth --log-dir "$work/mixjob/logs-$n" -- "$E/ddp_gate" --device "$device" --micro 2 --job-key "mixed job" $m > "$work/mixjob-$n.out" 2>&1 &
done
wait || true
refused=$(cat "$work"/mixjob/logs-*/rank-*.log | grep -c "refused the job" || true)
stepped=$(cat "$work"/mixjob/logs-*/rank-*.log | grep -c "^step " || true)
if [ "$refused" -ge 2 ] && [ "$stepped" -eq 0 ]; then r="refused on both ranks before step 0"; else r="NOT REFUSED"; fail=1; fi
say "a job with rank 0 mixed and rank 1 at f32 (the same job key): $r"

say "## Gate 3: the loss against f32 (one process, M 4, 200 steps)"
one f32-long --micro 4 --steps 200 --losses && one mixed-long --micro 4 --steps 200 --losses --mixed && one mixed8-long --micro 4 --steps 200 --losses --mixed --loss-scale 8 || { say "long runs: FAILED"; fail=1; }
for n in f32-long mixed-long mixed8-long; do grep '^loss ' "$work/$n.out" > "$work/$n/losses"; done
diffs() {
  awk -v lo="$3" -v hi="$4" 'NR == FNR { a[$2] = $3; next } ($2 in a) && $2 >= lo && $2 <= hi { d = $3 - a[$2]; if (d < 0) d = -d; r = d / (a[$2] < 0 ? -a[$2] : a[$2]); if (d >= m) { m = d; s = $2 }; if (r > rm) rm = r }
       END { printf "largest |difference| %.3e at step %d (relative %.3e)", m, s, rm }' "$1" "$2"
}
mean() { awk -v lo="$2" -v hi="$3" '$2 >= lo && $2 <= hi { s += $3; n++ } END { printf "%.5f", s / n }' "$1"; }
for n in mixed-long mixed8-long; do
  say "$n against f32: steps 1-12 $(diffs "$work/f32-long/losses" "$work/$n/losses" 1 12); steps 1-200 $(diffs "$work/f32-long/losses" "$work/$n/losses" 1 200); skipped steps $(grep -c '^skipped' "$work/$n.out" || true)"
done
say "mean loss, steps 1-10: f32 $(mean "$work/f32-long/losses" 1 10), mixed 2^0 $(mean "$work/mixed-long/losses" 1 10), mixed 2^8 $(mean "$work/mixed8-long/losses" 1 10); steps 191-200: f32 $(mean "$work/f32-long/losses" 191 200), mixed 2^0 $(mean "$work/mixed-long/losses" 191 200), mixed 2^8 $(mean "$work/mixed8-long/losses" 191 200)"
for n in f32-long mixed-long mixed8-long; do
  say "mean loss by window, $n: steps 1-12 $(mean "$work/$n/losses" 1 12), 13-50 $(mean "$work/$n/losses" 13 50), 51-100 $(mean "$work/$n/losses" 51 100), 101-150 $(mean "$work/$n/losses" 101 150), 151-200 $(mean "$work/$n/losses" 151 200)"
done
awk 'NR == FNR { a[$2] = $3; next } FILENAME == ARGV[2] { b[$2] = $3; next } ($2 in a) && ($2 == 1 || $2 % 25 == 0) { printf "  step %3d: f32 %.6f mixed 2^0 %.6f mixed 2^8 %.6f\n", $2, a[$2], b[$2], $3 }' "$work/f32-long/losses" "$work/mixed-long/losses" "$work/mixed8-long/losses" >> "$log"
say "# ended $(date -u +%Y-%m-%dT%H:%M:%SZ), $([ $fail -eq 0 ] && echo 'all checks passed' || echo 'FAILURES')"
cat "$log"
exit $fail
