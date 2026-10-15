#!/bin/sh
# C1 gate 1 (the reduced-precision product off): E2's script (doc/gates/e2/ddp_gate_refs.sh), run on this tree, and
# every final-weights hash it prints compared with the 0.5.0 log of the same label
# (doc/gates/e2/logs/<label>-ddp-gate.log); they must be equal, line for line.
#
# References of examples/ddp_gate.rs on one machine: for each configuration, the reference (one
# process, the global micro-steps accumulated on one device) and the same training under the
# launcher with the ranks sharing the machine's device; the final weights and every per-step
# hash must equal the reference's. Then a resume from a checkpoint (7 steps, checkpoints every 3,
# a relaunch to step 12 from step 6) and two launchers acting as nodes with node copies of the
# checkpoints. The log goes to doc/gates/c1/logs/<label>-ddp-gate.log.
#
#   cargo build --release --examples [--features metal|cuda]
#   doc/gates/c1/ddp_gate_refs.sh mac-cpu cpu 2       # label, device, most processes at once
#   doc/gates/c1/ddp_gate_refs.sh mac-metal metal 2
set -eu
label=$1
device=$2
max=${3:-4}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
E="$root/target/release/examples"
work="$root/runs/c1-ddp-gate-$label"
log="$here/logs/$label-ddp-gate.log"
rm -rf "$work"
mkdir -p "$work" "$here/logs"
fail=0
say() { echo "$*" >> "$log"; }
{
  echo "# C1 gate 1 (half precision off): examples/ddp_gate.rs references, $label: device $device, ranks sharing the device, at most $max processes at once"
  echo "# $(rustc --version | sed 's/ (.*//'); $(uname -sm); $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$log"
final() { sed -n 's/^final //p' "$1/record.txt"; }
for cfg in "gate:" "absent:--absent" "idle-after:--idle after" "idle-torch:--idle torch" "clip:--clip 1.0" "sgd:--opt sgd"; do
  name=${cfg%%:*}
  opts=${cfg#*:}
  for micro in 2 4; do
    ref="$work/$name-m$micro-ref"
    # shellcheck disable=SC2086
    "$E/ddp_gate" --device "$device" --micro "$micro" $opts --out "$ref" > "$ref.out" 2>&1 || { say "$name M $micro reference: FAILED"; fail=1; continue; }
    say "$name M $micro reference (1 process, $micro accumulation micro-steps): final weights sha256 $(final "$ref")"
    for w in 2 4; do
      [ $((micro % w)) -eq 0 ] || continue
      if [ "$w" -gt "$max" ]; then say "$name M $micro W $w: SKIPPED ($w processes at once, at most $max)"; continue; fi
      run="$work/$name-m$micro-w$w"
      # shellcheck disable=SC2086
      "$E/launch" --nproc-per-node "$w" --log-dir "$run/logs" -- "$E/ddp_gate" --device "$device" --ranks-per-device "$w" --micro "$micro" $opts --out "$run" > "$run.out" 2>&1 || { say "$name M $micro W $w: FAILED"; fail=1; continue; }
      same=yes
      for r in $(seq 0 $((w - 1))); do cmp -s "$ref/rank-0.steps" "$run/rank-$r.steps" || same=NO; done
      [ "$(final "$run")" = "$(final "$ref")" ] || same=NO
      [ $same = yes ] || fail=1
      say "$name M $micro W $w ($w ranks sharing the device): final weights sha256 $(final "$run"); every per-step hash and the final weights equal the reference: $same"
    done
  done
done
# Resume: 7 steps with checkpoints every 3, then a relaunch to step 12 from the checkpoint of step 6.
if [ "$max" -ge 2 ]; then
  run="$work/resume"
  "$E/launch" --nproc-per-node 2 --log-dir "$run/logs1" -- "$E/ddp_gate" --device "$device" --ranks-per-device 2 --steps 7 --ckpt-every 3 --out "$run" > "$run.out" 2>&1 || fail=1
  "$E/launch" --nproc-per-node 2 --log-dir "$run/logs2" -- "$E/ddp_gate" --device "$device" --ranks-per-device 2 --resume --out "$run" >> "$run.out" 2>&1 || fail=1
  grep -q "resumed at step 6" "$run/logs2/rank-0.log" || fail=1
  same=yes
  [ "$(final "$run")" = "$(final "$work/gate-m2-ref")" ] || same=NO
  # Step 7 ran twice (before and after the resume): one line per step means the same hash both times.
  sort -u "$run/rank-1.steps" | sort -n | cmp -s - "$work/gate-m2-ref/rank-0.steps" || same=NO
  [ $same = yes ] || fail=1
  say "resume W 2 M 2 (checkpoint of step 6, relaunched to step 12): final weights sha256 $(final "$run"); equals the reference and every per-step hash: $same"
  # Two launchers acting as nodes (one process each, localhost), checkpoints with node copies.
  run="$work/two-nodes"
  port=$((20000 + $$ % 20000))
  secret="$work/secret"
  head -c 48 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$secret"
  for n in 0 1; do
    "$E/launch" --nproc-per-node 1 --nnodes 2 --node-rank $n --master-addr 127.0.0.1 --master-port $port --secret-file "$secret" --log-dir "$run/logs-$n" -- "$E/ddp_gate" --device "$device" --micro 2 --ckpt-every 6 --ckpt-dir "$run/node-$n" --node-copies --out "$run" > "$run-$n.out" 2>&1 &
  done
  wait || fail=1
  rm -f "$secret"
  same=yes
  [ "$(final "$run")" = "$(final "$work/gate-m2-ref")" ] || same=NO
  for f in vars.json adam_m.json adam_v.json meta.json; do cmp -s "$run/node-0/step-0000000012/$f" "$run/node-1/step-0000000012/$f" || same=NO; done
  [ $same = yes ] || fail=1
  say "2 nodes x 1 process on localhost, node copies: final weights sha256 $(final "$run"); equals the reference, and node 1's checkpoint copy is node 0's byte for byte: $same"
  grep -h "^record" "$work/gate-m2-w2/logs/rank-0.log" | head -1 | sed 's/libm=[0-9a-f]*/libm=<sha256>/; s/kernels-[0-9a-f]*/kernels-<hex>/; s/metal-Apple_[A-Za-z0-9_]*-mathmode/metal-<the GPU name>-mathmode/; s/^/run record: /' >> "$log"
fi
# Against 0.5.0: every configuration's final weights, in order, as the 0.5.0 log printed them.
hashes() { sed -n 's/^\([^#].*\): final weights sha256 \([0-9a-f]\{64\}\).*/\1 \2/p' "$1" | sed 's/ ([^)]*)//'; }
old="$here/../e2/logs/$label-ddp-gate.log"
if [ -f "$old" ]; then
  hashes "$old" > "$work/hashes-050"
  hashes "$log" > "$work/hashes-new"
  if cmp -s "$work/hashes-050" "$work/hashes-new"; then
    say "against 0.5.0 (doc/gates/e2/logs/$label-ddp-gate.log): all $(wc -l < "$work/hashes-new" | tr -d ' ') final-weights lines identical"
  else
    say "against 0.5.0 (doc/gates/e2/logs/$label-ddp-gate.log): DIFFERENT"
    diff "$work/hashes-050" "$work/hashes-new" >> "$log" || true
    fail=1
  fi
else
  say "against 0.5.0: no 0.5.0 log for $label"
fi
say "# ended $(date -u +%Y-%m-%dT%H:%M:%SZ), $([ $fail -eq 0 ] && echo 'all identical' || echo 'FAILURES')"
cat "$log"
exit $fail
