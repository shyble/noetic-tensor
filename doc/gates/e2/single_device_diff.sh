#!/bin/sh
# The E2 differential gate for single-device training: the probe (single_device_probe.rs: six
# configurations, 60 steps, 3 models on the seed axis) and the copy-task example, built and run
# in a clean copy of the reference release (<tag>: its release tag, as `git tag` lists it) and of
# HEAD, must print the same bytes.
#
#   doc/gates/e2/single_device_diff.sh <tag> mac-cpu                 # the CPU reference backend
#   doc/gates/e2/single_device_diff.sh <tag> mac-metal metal metal   # Metal
#   doc/gates/e2/single_device_diff.sh <tag> win-cuda cuda cuda      # CUDA
#
# The copies and their build folders go to runs/diff/ (git-ignored); the log to
# doc/gates/e2/logs/<label>-single-device.log.
set -eu
ref=$1
label=$2
features=${3:-}
device=${4:-cpu}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
work="$root/runs/diff"
log="$here/logs/$label-single-device.log"
mkdir -p "$here/logs" "$work"
for t in ref new; do
  rm -rf "${work:?}/$t"
  mkdir -p "$work/$t"
done
git -C "$root" archive "$ref" | tar -x -C "$work/ref"
git -C "$root" archive HEAD | tar -x -C "$work/new"
for t in ref new; do
  cp "$here/single_device_probe.rs" "$work/$t/examples/"
  (
    cd "$work/$t"
    export CARGO_TARGET_DIR="$work/target-$t"
    cargo run -q --release -j "${CARGO_BUILD_JOBS:-1}" ${features:+--features "$features"} --example single_device_probe -- "$device"
    echo "copy_task:"
    if [ "$device" = cpu ]; then
      cargo run -q --release -j "${CARGO_BUILD_JOBS:-1}" ${features:+--features "$features"} --example copy_task
    else
      cargo run -q --release -j "${CARGO_BUILD_JOBS:-1}" ${features:+--features "$features"} --example copy_task -- "$device"
    fi
  ) > "$work/$t.out" 2>&1
done
{
  echo "# E2 single-device differential, $label: the reference ${ref#v} (its release tag) against this tree (HEAD), device $device"
  echo "# $(rustc --version | sed 's/ (.*//'); $(uname -sm); $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "## reference"
  cat "$work/ref.out"
  echo "## this tree"
  cat "$work/new.out"
  if cmp -s "$work/ref.out" "$work/new.out"; then echo "## identical: yes"; else echo "## identical: NO"; fi
} > "$log"
tail -1 "$log"
cmp -s "$work/ref.out" "$work/new.out"
