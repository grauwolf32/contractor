#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fixture_dir=$(mktemp -d)
trap 'rm -r "$fixture_dir"' 0
umask 077

master_key=sk-test-master-argv-secret
salt_key=sk-test-salt-argv-secret
upstream_token=sk-or-v1-test-upstream-argv-secret
printf '%s\n' "$master_key" > "$fixture_dir/master-key"
printf '%s\n' "$salt_key" > "$fixture_dir/salt-key"
printf '%s\n' "$upstream_token" > "$fixture_dir/upstream-key"

cat > "$fixture_dir/podman" <<'SH'
#!/bin/sh
set -eu
printf '%s\n' "$1" >> "$CONTRACTOR_TEST_CAPTURE_DIR/podman-calls"
case "$1" in
  network|container|start|exec) exit 0 ;;
  volume)
    [ "$2" = create ] && [ "$3" = --ignore ]
    touch "$CONTRACTOR_TEST_CAPTURE_DIR/retained-volume"
    ;;
  run)
    [ "$2" = --rm ] || exit 1
    printf '%s\n' "$@" > "$CONTRACTOR_TEST_CAPTURE_DIR/args"
    printf '%s' "$LITELLM_MASTER_KEY" > "$CONTRACTOR_TEST_CAPTURE_DIR/master-env"
    printf '%s' "$LITELLM_SALT_KEY" > "$CONTRACTOR_TEST_CAPTURE_DIR/salt-env"
    printf '%s' "$OPENROUTER_API_KEY" > "$CONTRACTOR_TEST_CAPTURE_DIR/upstream-env"
    ;;
  *) exit 1 ;;
esac
SH
chmod 700 "$fixture_dir/podman"

run_fixture() {
  PATH="$fixture_dir:$PATH" \
  CONTRACTOR_TEST_CAPTURE_DIR="$fixture_dir" \
  CONTRACTOR_LITELLM_MASTER_KEY_FILE="$1" \
  CONTRACTOR_LITELLM_SALT_KEY_FILE="$fixture_dir/salt-key" \
  CONTRACTOR_OPENROUTER_API_KEY_FILE="${2:-$fixture_dir/upstream-key}" \
    sh "$script_dir/run.sh"
}

run_fixture "$fixture_dir/master-key"
[ -f "$fixture_dir/retained-volume" ]
run_fixture "$fixture_dir/master-key"
[ -f "$fixture_dir/retained-volume" ]
args=$(cat "$fixture_dir/args")
for secret in "$master_key" "$salt_key" "$upstream_token"; do
  case "$args" in
    *"$secret"*) echo 'secret value appeared in podman arguments' >&2; exit 1 ;;
  esac
done
[ "$(cat "$fixture_dir/master-env")" = "$master_key" ]
[ "$(cat "$fixture_dir/salt-env")" = "$salt_key" ]
[ "$(cat "$fixture_dir/upstream-env")" = "$upstream_token" ]

previous=
master_name=false
salt_name=false
upstream_name=false
while IFS= read -r argument; do
  if [ "$previous" = -e ]; then
    case "$argument" in
      LITELLM_MASTER_KEY) master_name=true ;;
      LITELLM_SALT_KEY) salt_name=true ;;
      OPENROUTER_API_KEY) upstream_name=true ;;
    esac
  fi
  previous=$argument
done < "$fixture_dir/args"
[ "$master_name" = true ] && [ "$salt_name" = true ] && [ "$upstream_name" = true ]

expect_rejected() {
  rm -f "$fixture_dir/podman-calls"
  if run_fixture "$1" "$2" >/dev/null 2>&1; then
    echo 'invalid credential file was accepted' >&2
    exit 1
  fi
  if [ -e "$fixture_dir/podman-calls" ]; then
    echo 'invalid credential file reached Podman' >&2
    exit 1
  fi
}

ln -s "$fixture_dir/master-key" "$fixture_dir/master-link"
ln -s "$fixture_dir/upstream-key" "$fixture_dir/upstream-link"
printf '%s\n' invalid-key > "$fixture_dir/invalid-key"
: > "$fixture_dir/empty-key"
expect_rejected "$fixture_dir/master-link" "$fixture_dir/upstream-key"
expect_rejected "$fixture_dir/invalid-key" "$fixture_dir/upstream-key"
expect_rejected "$fixture_dir/master-key" "$fixture_dir/upstream-link"
expect_rejected "$fixture_dir/master-key" "$fixture_dir/missing-key"
expect_rejected "$fixture_dir/master-key" "$fixture_dir/invalid-key"
expect_rejected "$fixture_dir/master-key" "$fixture_dir/empty-key"

echo 'LiteLLM launch arguments omit credential values; invalid key files fail before Podman'
