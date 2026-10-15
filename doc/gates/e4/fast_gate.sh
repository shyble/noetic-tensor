#!/bin/sh
# E4 gate 2: the fast mode (`--fast`: per-rank sums, buckets, the ring all-reduce, overlap with the
# backward pass) on examples/ddp_gate.rs, against the deterministic mode on the same machine and
# device. The fast mode is not reproducible bit for bit: its results are compared with a stated
# tolerance, never byte for byte, and are never pooled with deterministic results.
#
# 1. The gate's model (12 steps) at M 2 and M 4 global micro-steps, W 2 (and W 4 when 4 processes
#    may run): per-step losses against the deterministic run's, final weights, the ranks in sync
#    (the per-step weight checks), the key and the record; buckets of 25 MiB (one bucket) and of
#    16 KiB (several), overlap on and off.
# 2. The same model for 200 steps at M 4, W 2: the loss curves of the two modes.
# 3. A repeat of one fast run (an observation, not a claim).
# 4. A mixed-mode job (two launchers as nodes, one with --fast) and a fast job whose ranks ask for
#    other buckets: refused before step 0.
# 5. A fast run checkpointed and resumed; a deterministic resume of a fast checkpoint is refused.
#
#   cargo build --release --examples [--features metal]
#   doc/gates/e4/fast_gate.sh mac-cpu cpu 2       # label, device, most processes at once
#   doc/gates/e4/fast_gate.sh mac-metal metal 2
set -eu
label=$1
device=$2
max=${3:-4}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
E="$root/target/release/examples"
work="$root/runs/e4-fast-$label"
log="$here/logs/$label-fast.log"
rm -rf "$work"
mkdir -p "$work" "$here/logs"
# Background priority where the system has it.
if command -v taskpolicy > /dev/null; then prio="taskpolicy -b"; else prio="nice -n 10"; fi
fail=0
say() { echo "$*" >> "$log"; }
{
  echo "# E4 gate 2 (the fast mode), $label: device $device, ranks sharing the device, at most $max processes at once"
  echo "# $(rustc --version | sed 's/ (.*//'); $(uname -sm); $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# fast-mode results are compared with a tolerance, never byte for byte; they are not reproducible bit for bit"
} > "$log"
final() { sed -n 's/^final //p' "$1/record.txt"; }
record() { head -1 "$1/record.txt" | sed 's/libm=[0-9a-f]*/libm=<sha256>/; s/kernels-[0-9a-f]*/kernels-<hex>/; s/metal-Apple_[A-Za-z0-9_]*-mathmode/metal-<the GPU name>-mathmode/'; }
# ref NAME ARGS...: one process (the deterministic reference); its losses in NAME/losses.
ref() {
  name=$1; shift
  # shellcheck disable=SC2086
  $prio "$E/ddp_gate" --device "$device" --losses --out "$work/$name" "$@" > "$work/$name.out" 2>&1 || { say "$name: FAILED"; fail=1; return 1; }
  grep '^loss ' "$work/$name.out" > "$work/$name/losses"
}
# ranks NAME W ARGS...: W processes under the launcher sharing the device; losses in NAME/losses.
ranks() {
  name=$1; w=$2; shift 2
  $prio "$E/launch" --nproc-per-node "$w" --log-dir "$work/$name/logs" -- "$E/ddp_gate" --device "$device" --ranks-per-device "$w" --losses --out "$work/$name" "$@" > "$work/$name.out" 2>&1 || { say "$name: FAILED"; fail=1; return 1; }
  grep '^loss ' "$work/$name/logs/rank-0.log" > "$work/$name/losses"
}
# lossdiff A B: the largest |loss_A - loss_B| over the steps (and its step), and the largest relative difference.
lossdiff() {
  awk 'NR == FNR { a[$2] = $3; next } ($2 in a) { d = $3 - a[$2]; if (d < 0) d = -d; r = d / (a[$2] < 0 ? -a[$2] : a[$2]); if (d >= m) { m = d; s = $2 }; if (r > rm) rm = r; n++ }
       END { printf "largest |loss difference| %.3e at step %d, largest relative %.3e, over %d steps", m, s, rm, n }' "$1" "$2"
}
synced() { grep -q "weight_checks=[0-9]* ok" "$1/record.txt" && echo "every per-step weight check passed on every rank" || { fail=1; echo "WEIGHT CHECKS FAILED"; }; }

say "## 1. The gate's model, 12 steps"
for micro in 2 4; do
  ref "det-m$micro-ref" --micro "$micro" || continue
  say "deterministic M $micro, 1 process: final weights sha256 $(final "$work/det-m$micro-ref")"
  ranks "det-m$micro-w2" 2 --micro "$micro" || continue
  say "deterministic M $micro W 2: final weights sha256 $(final "$work/det-m$micro-w2") ($([ "$(final "$work/det-m$micro-w2")" = "$(final "$work/det-m$micro-ref")" ] && echo "equal to 1 process, as the deterministic mode promises" || { fail=1; echo "NOT EQUAL TO 1 PROCESS"; }))"
  for w in 2 4; do
    [ $((micro % w)) -eq 0 ] || continue
    if [ "$w" -gt "$max" ]; then say "fast M $micro W $w: SKIPPED ($w processes at once, at most $max)"; continue; fi
    for b in "25600:on" "16:on" "16:off"; do
      kb=${b%%:*}; ov=${b#*:}
      name="fast-m$micro-w$w-b$kb-$ov"
      extra=""; [ "$ov" = off ] && extra="--no-overlap"
      # shellcheck disable=SC2086
      ranks "$name" "$w" --micro "$micro" --fast --bucket-kb "$kb" $extra || continue
      same=$([ "$(final "$work/$name")" = "$(final "$work/det-m$micro-ref")" ] && echo "the same bits as the deterministic run" || echo "other bits than the deterministic run")
      say "fast M $micro W $w buckets $kb KiB overlap $ov: final weights sha256 $(final "$work/$name") ($same); against the deterministic losses: $(lossdiff "$work/det-m$micro-ref/losses" "$work/$name/losses"); $(synced "$work/$name")"
    done
  done
done
say "fast-mode record (M 4 W 2, 16 KiB buckets): $(record "$work/fast-m4-w2-b16-on")"
say "deterministic record (M 4 W 2): $(record "$work/det-m4-w2")"
say "note: at M 2, W 2 each rank has one micro-step, so the ring adds the two micro-steps' gradients a and b once, and f32 addition commutes: the fast mode can give the deterministic bits there; at M 4 W 2 (micro-steps a, b, c, d) it adds (a + b) + (c + d) where the deterministic mode adds ((a + b) + c) + d"

say "## 2. The gate's model, 200 steps, M 4"
ref det-long --micro 4 --steps 200 && ranks fast-long 2 --micro 4 --steps 200 --fast && ranks fast-long-b16 2 --micro 4 --steps 200 --fast --bucket-kb 16 && {
  tail10() { tail -10 "$1" | awk '{ s += $3 } END { printf "%.6f", s / NR }'; }
  say "deterministic, 1 process: final weights sha256 $(final "$work/det-long"); mean loss over steps 1-10 $(head -10 "$work/det-long/losses" | awk '{ s += $3 } END { printf "%.6f", s / NR }'), over steps 191-200 $(tail10 "$work/det-long/losses")"
  for name in fast-long fast-long-b16; do
    say "$name (W 2): final weights sha256 $(final "$work/$name"); mean loss over steps 191-200 $(tail10 "$work/$name/losses"); against the deterministic losses: $(lossdiff "$work/det-long/losses" "$work/$name/losses"); $(synced "$work/$name")"
  done
  awk 'NR == FNR { a[$2] = $3; next } ($2 in a) && ($2 % 20 == 0 || $2 == 1) { printf "  step %3d: deterministic %.6f fast %.6f difference %+.3e\n", $2, a[$2], $3, $3 - a[$2] }' "$work/det-long/losses" "$work/fast-long/losses" >> "$log"
}

say "## 3. A repeat of one fast run (an observation; the mode makes no such promise)"
ranks fast-repeat 2 --micro 4 --fast --bucket-kb 16 && {
  if cmp -s "$work/fast-repeat/rank-0.steps" "$work/fast-m4-w2-b16-on/rank-0.steps"; then r="the same per-step hashes as the first run"; else r="other per-step hashes than the first run"; fi
  say "fast M 4 W 2 buckets 16 KiB, run again: $r"
}

say "## 4. Refusals before step 0"
port=$((20000 + $$ % 20000))
for case in "mixed:--fast:" "buckets:--fast --bucket-kb 16:--fast"; do
  cname=${case%%:*}; rest=${case#*:}; o0=${rest%%:*}; o1=${rest#*:}
  for n in 0 1; do
    if [ $n -eq 0 ]; then opts=$o0; else opts=$o1; fi
    # shellcheck disable=SC2086
    $prio "$E/launch" --nproc-per-node 1 --nnodes 2 --node-rank $n --master-addr 127.0.0.1 --master-port $port --no-auth --log-dir "$work/refuse-$cname/logs-$n" -- "$E/ddp_gate" --device "$device" --micro 2 $opts > "$work/refuse-$cname-$n.out" 2>&1 &
  done
  st=0
  wait || st=$?
  port=$((port + 1))
  refused=$(cat "$work/refuse-$cname"/logs-*/rank-*.log | grep -c "mixed-mode job" || true)
  stepped=$(cat "$work/refuse-$cname"/logs-*/rank-*.log | grep -c "^step " || true)
  if [ "$refused" -ge 2 ] && [ "$stepped" -eq 0 ]; then r="refused on both ranks before step 0"; else r="NOT REFUSED ($refused refusals, $stepped steps)"; fail=1; fi
  say "$cname (node 0: ddp_gate $o0; node 1: ddp_gate $o1): $r; rank 0 said: $(grep -h "mixed-mode" "$work/refuse-$cname"/logs-0/rank-0.log | head -1 | sed 's/.*the rendezvous refused the job: //' | cut -c1-200)"
done

say "## 5. A fast checkpoint and its resume"
run="$work/fast-resume"
if ranks fast-resume 2 --micro 4 --fast --bucket-kb 16 --steps 7 --ckpt-every 3; then
  $prio "$E/launch" --nproc-per-node 2 --log-dir "$run/logs2" -- "$E/ddp_gate" --device "$device" --ranks-per-device 2 --micro 4 --fast --bucket-kb 16 --resume --out "$run" > "$run-2.out" 2>&1 || { say "fast resume: FAILED"; fail=1; }
  if [ "$(final "$run")" = "$(final "$work/fast-m4-w2-b16-on")" ]; then r="the uninterrupted fast run's final weights"; else r="other final weights than the uninterrupted fast run"; fi
  say "fast M 4 W 2, 7 steps with checkpoints every 3, resumed from step 6 to 12: $r (an observation)"
  if $prio "$E/launch" --nproc-per-node 2 --log-dir "$run/logs3" -- "$E/ddp_gate" --device "$device" --ranks-per-device 2 --micro 4 --resume --out "$run" > "$run-3.out" 2>&1; then
    say "deterministic resume of the fast checkpoint: NOT REFUSED"; fail=1
  else
    say "deterministic resume of the fast checkpoint: refused ($(grep -h "resume refused" "$run"/logs3/rank-0.log | head -1 | sed 's/.*(\(another[^)]*\)).*/\1/'))"
  fi
fi
say "# ended $(date -u +%Y-%m-%dT%H:%M:%SZ), $([ $fail -eq 0 ] && echo 'all checks passed' || echo 'FAILURES')"
cat "$log"
exit $fail
