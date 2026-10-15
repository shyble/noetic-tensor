#!/bin/sh
# E4 gate 2, timing: the wall time of examples/ddp_gate.rs's training loop (`wall <seconds>`,
# rank 0, after the setup), deterministic against fast mode, on one machine and device, ranks
# sharing it. A measurement on this machine only: no claim about other machines, networks,
# world sizes, or any other engine.
#
# Model: a decoder of width D, L blocks, MLP F (vocab 16, 4 heads, context 8, 2 models on the
# seed axis), M global micro-steps of 2 items, N steps, AdamW. Each configuration runs R times, in
# rotation (so a change in the machine's load spreads over all of them).
#
#   doc/gates/e4/timing.sh mac-cpu cpu [D L F M N R]     # defaults 256 4 1024 4 10 3
set -eu
label=$1
device=$2
D=${3:-256}; L=${4:-4}; F=${5:-1024}; M=${6:-4}; N=${7:-10}; R=${8:-3}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
E="$root/target/release/examples"
work="$root/runs/e4-timing-$label"
log="$here/logs/$label-timing.log"
rm -rf "$work"
mkdir -p "$work" "$here/logs"
if command -v taskpolicy > /dev/null; then prio="taskpolicy -b"; else prio="nice -n 10"; fi
hw() {
  if [ "$(uname -s)" = Darwin ]; then
    # The model identifier names the machine; macOS runs background-priority work on the
    # efficiency cores.
    echo "Apple silicon, model $(sysctl -n hw.model), $(sysctl -n hw.perflevel0.logicalcpu) performance and $(sysctl -n hw.perflevel1.logicalcpu) efficiency cores, $(($(sysctl -n hw.memsize) / 1073741824)) GiB; $(sw_vers -productName) $(sw_vers -productVersion)"
  else
    uname -a
  fi
}
load() { uptime | sed 's/.*load average[s]*: //'; }
{
  echo "# E4 gate 2, timing, $label: device $device; $(hw)"
  echo "# $(rustc --version | sed 's/ (.*//'); $(uname -sm); $(date -u +%Y-%m-%dT%H:%M:%SZ); load average at the start: $(load)"
  echo "# model: width $D, $L blocks, MLP $F; $M global micro-steps; $N steps; $R runs per configuration, in rotation; every process at background priority ($prio), sharing the device"
  echo "# a measurement on this machine, with its key; not a statement about other hardware, networks or engines"
} > "$log"
size="--width $D --blocks $L --mlp $F --micro $M --steps $N"
# name:W:options
list="ref:1:
det-w2:2:
fast-w2:2:--fast
fast-w2-noovl:2:--fast --no-overlap
fast-w2-b1m:2:--fast --bucket-kb 1024"
for rep in $(seq 1 "$R"); do
  echo "$list" | while IFS= read -r c; do
    name=${c%%:*}; rest=${c#*:}; w=${rest%%:*}; opts=${rest#*:}
    out="$work/$name-$rep"
    if [ "$w" = 1 ]; then
      # shellcheck disable=SC2086
      $prio "$E/ddp_gate" --device "$device" $size $opts --out "$out" < /dev/null > "$out.log" 2>&1 || { echo "$name run $rep: FAILED" >> "$log"; continue; }
      rlog="$out.log"
    else
      # shellcheck disable=SC2086
      $prio "$E/launch" --nproc-per-node "$w" --log-dir "$out/logs" -- "$E/ddp_gate" --device "$device" --ranks-per-device "$w" $size $opts --out "$out" < /dev/null > "$out.log" 2>&1 || { echo "$name run $rep: FAILED" >> "$log"; continue; }
      rlog="$out/logs/rank-0.log"
    fi
    echo "$name run $rep: $(sed -n 's/^wall \([0-9.]*\) seconds.*/\1/p' "$rlog") s (load average after: $(load))" >> "$log"
  done
done
{
  echo "## median wall time per configuration (seconds for $N steps)"
  for name in ref det-w2 fast-w2 fast-w2-noovl fast-w2-b1m; do
    med=$(grep "^$name run " "$log" | sed 's/.*: \([0-9.]*\) s .*/\1/' | sort -n | awk '{ a[NR] = $1 } END { if (NR) print a[int((NR + 1) / 2)] }')
    all=$(grep "^$name run " "$log" | sed 's/.*: \([0-9.]*\) s .*/\1/' | tr '\n' ' ')
    echo "$name: median $med (runs: $all)"
  done
  echo "## keys"
  for name in det-w2 fast-w2 fast-w2-noovl fast-w2-b1m; do
    echo "$name: $(head -1 "$work/$name-1/record.txt" | sed 's/libm=[0-9a-f]*/libm=<sha256>/; s/kernels-[0-9a-f]*/kernels-<hex>/; s/metal-Apple_[A-Za-z0-9_]*-mathmode/metal-<the GPU name>-mathmode/; s/ weight_checks=.*//')"
  done
  echo "# ended $(date -u +%Y-%m-%dT%H:%M:%SZ); load average at the end: $(load)"
} >> "$log"
cat "$log"
