#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fixture_dir=$(mktemp -d)
trap 'rm -rf "$fixture_dir"' 0
repo_dir="$fixture_dir/fixture checkout"
mkdir -p "$repo_dir/deploy/phoenix" "$fixture_dir/bin"
cp "$script_dir/run.sh" "$repo_dir/deploy/phoenix/run.sh"
export CONTRACTOR_PHOENIX_TEST_CAPTURE="$fixture_dir"
export PATH="$fixture_dir/bin:$PATH"

cat > "$fixture_dir/bin/podman" <<'SH'
#!/bin/sh
set -eu
capture=$CONTRACTOR_PHOENIX_TEST_CAPTURE
printf '%s\n' "$1" >> "$capture/calls"
case "$1" in
  container)
    [ "${CONTRACTOR_PHOENIX_TEST_ENGINE_ERROR:-false}" != true ] || exit 125
    [ -f "$capture/exists" ]
    ;;
  run)
    printf '%s\n' "$@" > "$capture/args"
    touch "$capture/exists"
    ;;
  start|stop) : ;;
  healthcheck)
    echo probe >> "$capture/probes"
    [ "${CONTRACTOR_PHOENIX_TEST_UNHEALTHY:-false}" != true ]
    ;;
  port) echo '127.0.0.1:6006' ;;
  inspect) echo running ;;
  rm) rm "$capture/exists" ;;
  *) exit 2 ;;
esac
SH
cat > "$fixture_dir/bin/sleep" <<'SH'
#!/bin/sh
exit 0
SH
chmod 755 "$fixture_dir/bin/podman" "$fixture_dir/bin/sleep"

run() { sh "$repo_dir/deploy/phoenix/run.sh" "$@"; }
run up >/dev/null
test -d "$repo_dir/.local/phoenix"
printf 'retained\n' > "$repo_dir/.local/phoenix/sentinel"
args=$(cat "$fixture_dir/args")
case "$args" in *"127.0.0.1:6006:6006"*) ;; *) exit 1 ;; esac
case "$args" in *"$repo_dir/.local/phoenix:/data:Z"*) ;; *) exit 1 ;; esac
case "$args" in *"@sha256:"*) ;; *) exit 1 ;; esac
run up >/dev/null
test "$(rg -c '^run$' "$fixture_dir/calls")" -eq 1
test "$(run status)" = running
run down >/dev/null
test "$(cat "$repo_dir/.local/phoenix/sentinel")" = retained
test "$(run status)" = 'Phoenix is stopped'
run down >/dev/null

expect_failure() {
  if "$@" >/dev/null 2>&1; then
    echo 'expected launch rejection' >&2
    exit 1
  fi
}
for port in invalid 65536 999999999999999999999999; do
  export CONTRACTOR_PHOENIX_PORT="$port"
  expect_failure run up
done
unset CONTRACTOR_PHOENIX_PORT
expect_failure run unknown
expect_failure run up unexpected
export CONTRACTOR_PHOENIX_TEST_ENGINE_ERROR=true
expect_failure run up
unset CONTRACTOR_PHOENIX_TEST_ENGINE_ERROR
test ! -f "$fixture_dir/exists"
export CONTRACTOR_PHOENIX_TEST_UNHEALTHY=true
: > "$fixture_dir/probes"
expect_failure run up
test "$(wc -l < "$fixture_dir/probes" | tr -d ' ')" -eq 30
run down >/dev/null
unset CONTRACTOR_PHOENIX_TEST_UNHEALTHY
mv "$repo_dir/.local/phoenix" "$repo_dir/.local/retained"
ln -s "$repo_dir/.local/retained" "$repo_dir/.local/phoenix"
expect_failure run up
test "$(cat "$repo_dir/.local/retained/sentinel")" = retained
echo 'Phoenix launch, reuse, loopback binding, retained data and bounded failure checks passed'
