# Packaging and updating Agent Skills

[Documentation index](../README.md) · [Local stack](local-stack.md)

Run commands from the repository root. API examples require the local Server
and `CONTRACTOR_API_TOKEN` from the local-stack guide. The owning contract is
[Agent Skills](../spec/09-agent-skills.md).

In the UI, open a package from **Skills** and choose **Browse files** to inspect
its folders. `SKILL.md` opens automatically after loading the directory. Select
a reference or asset to view UTF-8 text up to 256 KiB; Markdown also has a source
tab. Preview reads the selected exact revision without extracting or executing
the package. Binary and oversized files are available in the original ZIP
download. See the [UI archive limits](../../ui/README.md) for supported archives.

## Bundled Agent Skills

Reviewable built-in Agent Skill sources live only below
`configs/skills/<name>`. Validate a source tree and create its deterministic,
script-free package with:

```shell
contractor-skill validate configs/skills/likec4
contractor-skill package configs/skills/likec4 /tmp/likec4.zip
```

`AgentTemplate` is a Server-side configuration term, not Worker guidance.
Never refer to it from bundled `SKILL.md` or reference content. Conditional
guidance says that an operation must be visible in the current Worker
invocation and states what to do when it is absent. The aggregate package gate
enforces this boundary for all bundled packages.

At startup the Server validates the complete bundled set before writing any
artifact, then creates only missing `skills/<name>` bindings in the configured
local user's UserScope. A restart never treats the filesystem as desired state:
an existing artifact with different bytes or media type is reported as
`seed_drift` and remains current. To adopt an edited bundled source, package it
explicitly and upload the resulting ZIP through the ordinary user Artifact PUT
with media type `application/vnd.contractor.agent-skill+zip` and the current
revision precondition. Future Runs use that new binding; existing Runs keep
their pinned revision.

Inspect the current binding and immutable version history, then use the current
revision as the compare-and-swap precondition for an operator-authored update:

```shell
SKILL_NAME=likec4
SKILL_BASE="http://127.0.0.1:8080/v1/artifacts/skills/$SKILL_NAME"

curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "$SKILL_BASE/metadata" | jq .
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "$SKILL_BASE/versions?limit=100" | jq .

CURRENT_REVISION="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "$SKILL_BASE/metadata" | jq -r .artifact.revision)"
contractor-skill validate "configs/skills/$SKILL_NAME"
contractor-skill package \
  "configs/skills/$SKILL_NAME" "/tmp/$SKILL_NAME.zip"
curl --fail --silent --show-error -X PUT \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: application/vnd.contractor.agent-skill+zip' \
  -H "If-Match: \"$CURRENT_REVISION\"" \
  --data-binary "@/tmp/$SKILL_NAME.zip" "$SKILL_BASE" | jq .
```

Create a Run through the ordinary Workflow API after uploading exact `source`
and optional `existing_likec4` inputs as described in the
[project-workflow guide](project-workflows.md#likec4-from-a-standalone-run-workspace). The current workspace
Workflow assigns the bundled LikeC4 Skill to its builder and validator Workers:

```shell
jq -n --argjson source "$SOURCE_REF" --arg objective 'Model the architecture' \
  '{workflow:"likec4-from-workspace@7",parameters:{objective:$objective},artifacts:{source:$source}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H "Idempotency-Key: likec4-skilled-$(date +%s)" \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/runs | jq .
```

The bounded real-process proof covers startup seeding, restart idempotency,
native ADK disclosure tools, an Artifact CAS update between source selection
and Run initialization, retry pinning, release cleanup, and empty-skill slot
reuse:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-agent-skills-mvp
```

Before publishing any bundled-package or Runtime Skill change, run the complete
bounded gate. It validates the shared adversarial corpus in independent Go and
Python implementations, packages all bundled sources twice, checks the exact
server-side selection topology, exercises PostgreSQL/CAS and write-fence races,
and finishes with the real-process proof:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-agent-skills-hardening
```

The executable ownership map is
[`tests/e2e/agent_skills_matrix.yml`](../../tests/e2e/agent_skills_matrix.yml).
Keep a new fault or invariant in that matrix and give it one concrete test
owner; a prose-only hardening claim is not a release gate.

Operational recovery is deliberately Artifact-first:

- `seed_drift` means the database current binding won. Inspect its metadata and
  immutable versions; never restart repeatedly expecting filesystem content to
  overwrite it. Publish the intended package with an ordinary `If-Match` CAS.
- A Run in `initializing` with reason `skill_initialization_pending` is not
  schedulable. Retryable Artifact failures are recovered by Scheduler using the
  already selected exact source. Invalid media, digest, archive or missing
  source terminates only that Run with a bounded `skill_*` code and logical
  `skills/<name>`; package content and parser text must not appear in logs.
- A CAS conflict during operator update means current changed independently.
  Re-read metadata and decide from the new exact revision; do not replay an
  update without a fresh precondition.
- `worker_stop_unconfirmed`, `allocation_cleanup_failed`, or a fenced Runtime
  after Skill extraction cleanup is a process-isolation event. Do not return
  that slot to service. Replace the Runtime process and inspect/remove its
  allocation work directory before reusing the same work root.

`resolvedSkills` is a mandatory private allocation field. Even an
AgentTemplate without skills is sent as `"resolvedSkills": []`; a non-empty
value contains only exact RunScope `skills/<name>` revisions and package
digests. It never contains the owner's source ref, package bytes or catalog
authority. This private-wire change is deliberately fail-closed: mixed Server
and Runtime Agent versions are unsupported. Before deploying a version that
adds or changes the allocation shape, stop new Run admission, let active
allocations finish (or cancel them), confirm every Runtime slot is released,
then replace the Server and all Runtime Agents together. Do not attempt a
rolling upgrade with live allocations.
