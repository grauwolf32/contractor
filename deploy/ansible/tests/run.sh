#!/bin/sh
# Gate for deploy/ansible (V402-001): syntax check, ansible-lint, the
# controller-only resolution tests, and the tool probe against the locked
# Runtime environment. Needs uv.
set -eu
ansible_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$ansible_dir/../.." && pwd)
ansible_core='ansible-core==2.21.5'
ansible_lint='ansible-lint==26.10.1'
export ANSIBLE_CONFIG="$ansible_dir/ansible.cfg" ANSIBLE_NOCOLOR=1
cd "$ansible_dir"

uvx --from "$ansible_core" ansible-playbook --syntax-check -i inventory.example.yml runtime.yml
uvx --from "$ansible_lint" --with "$ansible_core" ansible-lint --offline --strict
uvx --from "$ansible_core" ansible-playbook -i tests/inventory.yml tests/resolve.yml

# With no executables on PATH the Runtime's own probes must advertise neither
# validator nor any scanner; instance.yml relies on this output shape.
python=$(cd "$repo_dir/runtime" && uv run --locked python -c 'import sys; print(sys.executable)')
"$python" -I tests/test_scope_policy.py
probe=$(env -i PATH=/nonexistent LANG=C.UTF-8 HOME=/nonexistent \
  "$python" -I roles/contractor_runtime/files/probe_tools.py)
printf '%s' "$probe" | "$python" -I -c '
import json, sys
found = json.load(sys.stdin)
assert sorted(found) == ["likec4@1", "openapi@1", "scan@1"], found
assert found["scan@1"] == [], found
assert "validate_openapi" not in found["openapi@1"] and found["openapi@1"], found
assert "validate_likec4" not in found["likec4@1"] and found["likec4@1"], found
'
echo "deploy/ansible: all checks passed"
