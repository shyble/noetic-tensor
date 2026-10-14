#!/bin/sh
# The E2 gates (sharded data parallelism against data parallelism; doc/distributed.md,
# "Sharded data parallelism"), with the 0.3.0/0.4.0 data-parallel gates beside them, in release
# mode, one test at a time. The log goes to doc/gates/e2/logs/<label>.log.
#
#   doc/gates/e2/run_gates.sh mac-cpu              # the CPU reference backend
#   doc/gates/e2/run_gates.sh mac-metal metal      # Apple GPU ranks (the gpu_ tests)
#   doc/gates/e2/run_gates.sh win-cuda cuda        # CUDA ranks (the gpu_ tests)
#
# DIST_TEST_MAX_PROCS=N (a development switch) skips the runs needing more than N processes at
# once; each skip is printed in the log. CARGO_BUILD_JOBS limits the build.
#
# The log keeps the tests' output; Cargo's build lines are dropped, and local paths and short
# hexadecimal ids (build ids, kernel-source digests) are replaced by <path> and <hex>. The sha256
# values the gates print are kept whole.
set -eu
label=$1
features=${2:-}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/logs"
log="$here/logs/$label.log"
cd "$here/../../.."
root=$(pwd)
sanitize() {
  perl -pe 'BEGIN { $r = shift } s/\Q$r\E/./g; s{(/U[s]ers/|/h[o]me/|[A-Za-z]:\\U[s]ers\\)\S*}{<path>}g; s/\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}\b/<hex>/g' "$root" |
    grep -vE '^ *(Compiling|Finished|Running|Blocking|Fresh|Doc-tests)' || true
}
{
  echo "# E2 gates, $label"
  echo "# tree: the commit that records this log; working tree $(git diff --quiet HEAD && echo clean || echo modified)"
  echo "# $(rustc --version | sed 's/ (.*//'); $(uname -sm); DIST_TEST_MAX_PROCS=${DIST_TEST_MAX_PROCS:-unset}"
  echo "# started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$log"
if [ -n "$features" ]; then
  set -- --features "$features" --lib nn::dist::tests -- gpu_ --test-threads 1 --nocapture
else
  set -- --lib nn::dist -- --test-threads 1 --nocapture
fi
status=0
cargo test --release -j "${CARGO_BUILD_JOBS:-1}" "$@" > "$log.raw" 2>&1 || status=$?
sanitize < "$log.raw" >> "$log"
rm -f "$log.raw"
echo "# ended $(date -u +%Y-%m-%dT%H:%M:%SZ), exit $status" >> "$log"
grep -E "^test result|FAILED|panicked" "$log" || true
exit $status
