#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fixture_dir=$(mktemp -d)
trap 'rm -r "$fixture_dir"' 0
umask 077

master_key=sk-test-master-argv-secret
salt_key=sk-test-salt-argv-secret
upstream_token=test-upstream-argv-secret
printf '%s\n' "$master_key" > "$fixture_dir/master-key"
printf '%s\n' "$salt_key" > "$fixture_dir/salt-key"

cat > "$fixture_dir/podman" <<'SH'
#!/bin/sh
set -eu
case "$1" in
  network|volume|container|start|exec) exit 0 ;;
  run)
    [ "$2" = --rm ] || exit 1
    printf '%s\n' "$@" > "$CONTRACTOR_TEST_CAPTURE_DIR/args"
    printf '%s' "$LITELLM_MASTER_KEY" > "$CONTRACTOR_TEST_CAPTURE_DIR/master-env"
    printf '%s' "$LITELLM_SALT_KEY" > "$CONTRACTOR_TEST_CAPTURE_DIR/salt-env"
    printf '%s' "$CONTRACTOR_LM_STUDIO_TOKEN" > "$CONTRACTOR_TEST_CAPTURE_DIR/upstream-env"
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
  CONTRACTOR_LM_STUDIO_URL=http://lm-studio.example:1234/v1 \
  CONTRACTOR_LM_STUDIO_TOKEN="$upstream_token" \
    sh "$script_dir/run.sh"
}

run_fixture "$fixture_dir/master-key"
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
      CONTRACTOR_LM_STUDIO_TOKEN) upstream_name=true ;;
    esac
  fi
  previous=$argument
done < "$fixture_dir/args"
[ "$master_name" = true ] && [ "$salt_name" = true ] && [ "$upstream_name" = true ]

ln -s "$fixture_dir/master-key" "$fixture_dir/master-link"
if run_fixture "$fixture_dir/master-link" >/dev/null 2>&1; then
  echo 'symlinked master key was accepted' >&2
  exit 1
fi
printf '%s\n' invalid-key > "$fixture_dir/invalid-key"
if run_fixture "$fixture_dir/invalid-key" >/dev/null 2>&1; then
  echo 'invalid master key prefix was accepted' >&2
  exit 1
fi

echo 'LiteLLM launch arguments omit credential values'
