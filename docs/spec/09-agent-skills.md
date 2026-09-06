# 09 — Agent Skills

Status: **Working agreement**

Depends on: [01](01-agent-template.md), [02](02-runtime-and-a2a.md),
[03](03-artifact-plane.md) and [04](04-execution-lifecycle-and-metrics.md)

## Purpose and terminology

Contractor uses the open Agent Skills package format and Google ADK's native
`SkillToolset` for reusable Worker guidance. A skill is progressively disclosed
model context, not a Worker implementation, Toolset implementation, executable
plugin or placement capability.

This document uses **Agent Skill** or **skill package** for the directory with
`SKILL.md`. That concept is unrelated to an A2A Agent Card `AgentSkill`. Runtime
Agent continues to advertise only Contractor's stage-content operation in its
A2A Agent Card; selected Agent Skills are private allocation configuration and
never create additional A2A operations.

The initial contract preserves the useful content shape from `contractor-old`
without preserving its implementation. The old system copied Markdown into
reserved Memory notes and exposed `skills_list` and `skills_read`. The new
system stores immutable ZIP versions in the existing ArtifactStore and uses
ADK's native `list_skills`, `load_skill` and `load_skill_resource`
implementations. Contractor deliberately filters `run_skill_script` and
replaces ADK's stock script-bearing system instruction with an equivalent
script-free progressive-disclosure instruction. No Contractor Memory note,
MemoryTools tag, parallel model-facing Skill API or separate Skill persistence
is created.

Skills are global to the authenticated owner rather than owned by a Project.
The Projects UI may display the exact Skills required by a candidate Workflow,
but ProjectScope never contains Skill copies. Run initialization continues to
resolve the AgentTemplate-selected current UserScope binding and fork its exact
revision into the reserved RunScope namespace, including for a Project Run.

## AgentTemplate ownership

Skills belong to an `AgentTemplate` because they describe reusable Worker
specialization. Workflow, Stage, Run input and Planner output cannot add,
remove or replace them.

An AgentTemplate source excerpt is:

```yaml
spec:
  skills:
    - namespace: skills
      name: likec4
```

`spec.skills` is optional; omission and `skills: []` normalize to the same empty
set. Each entry is an existing versionless `ArtifactRef` with exact Namespace
`skills`; `revision` is forbidden in AgentTemplate source. Entries are unique by
`(namespace, name)`, normalized in name order and contribute to the
AgentTemplate digest as those logical refs. The canonical manifest omits an
empty set, preserving the identity of pre-Agent-Skills templates; adding or
removing refs, or otherwise changing the normalized set to a distinct value, is
a behavioral change that requires a new AgentTemplate version. The name follows
the Agent Skills grammar: 1 through 64
lowercase ASCII letters, digits or hyphen-separated components; it cannot start
or end with a hyphen or contain consecutive hyphens. It must equal `name` in
the package's `SKILL.md` frontmatter.

The source scope is structural rather than part of `ArtifactRef`: the Run-create
transaction resolves every logical ref in that Run owner's UserScope. A write
through the ordinary User Artifact API advances the binding and returns a new
exact ArtifactRef. Future Runs resolve that new exact ref; an existing Run
continues to use its selected exact revision and resulting RunScope fork.

One AgentTemplate may declare at most 32 skills. The union retained by one
WorkflowRun snapshot may contain at most 128 distinct refs. Both limits are
validated before resolving or forking any package.

With a non-empty set, the three model-visible function names `list_skills`,
`load_skill` and `load_skill_resource` are reserved. AgentTemplate
configuration is invalid if a selected Contractor Toolset operation has any of
those names. Runtime's `adk@1` probe also requires the pinned native
SkillToolset plus Contractor filter/instruction adapter to expose exactly that
initial set, with neither `run_skill_script` nor `search_skills`.

The distinction between instructions and skills is strict:

- `AgentTemplate.instructions` is always-on mandatory Worker behavior;
- `AgentTemplate.skills` is progressively disclosed supplementary guidance;
- a requirement the Worker must observe without first choosing `load_skill`
  belongs in `instructions`, not in a skill;
- there is no `requiredSkills`, activation mode, Workflow/Stage/Run override or
  Planner-selected skill in this version.

`AgentTemplate` is a Server-side selection and provenance abstraction. Its
name, ref and catalog mechanics are not part of Worker-facing Skill guidance.
Contractor-provided `SKILL.md` files and resources therefore never tell the
model that a Skill or operation was selected by an AgentTemplate. When guidance
depends on another capability, it refers only to an operation visible in the
current Worker invocation and defines the behavior when that operation is
absent. Server configuration, migration documents and selection tests may use
the AgentTemplate term because they describe the control-plane side of this
boundary.

For a checked-in package that depends on named operations, the repository may
lock one reviewed AgentTemplate to that package and assert that its explicit
Toolset allowlist contains every operation named by the guidance. This is a
release-time compatibility test only. Runtime still does not interpret Skill
metadata or prose as dependencies, install tools automatically, or expand an
allocation's authority.

AgentTemplate configuration loading validates ref syntax, uniqueness and
limits but does not read ArtifactStore. Run initialization fails before
scheduling if the owner has no valid current artifact for any declared ref.

## Package profile

One package is one ZIP artifact whose stored media type is canonicalized to
`application/vnd.contractor.agent-skill+zip`; media-type parameters are not
accepted. The archive root represents the skill directory and contains
`SKILL.md` directly:

```text
SKILL.md
references/
  syntax.md
  examples.md
assets/
  template.c4
```

Reviewable built-in sources live at `configs/skills/<name>/`. Runtime never
scans that tree; `SkillCatalog` uses it only for the bounded owner-artifact
initialization described below. Only an exact artifact resolved and forked into
a Run can affect a Worker.

The initial profile accepts exactly:

- one root `SKILL.md` encoded as UTF-8;
- zero or more regular UTF-8 files below `references/`;
- zero or more regular UTF-8 or binary files below `assets/`;
- directory entries needed for those paths.

It rejects `scripts/`, every other root entry, symlinks, hard links, device or
special files, duplicate normalized paths, backslash paths, absolute paths,
empty/dot/dot-dot path components, encrypted ZIP members and malformed ZIP
metadata. It also rejects lowercase `skill.md`; the portable contract uses the
standard exact spelling even when one ADK release accepts more.

Resource paths are deliberately portable in the initial profile. They contain
at most eight components and 512 ASCII bytes including `/`; every component
matches `[a-z0-9][a-z0-9._-]{0,127}`. ZIP member names are forward-slash paths
whose raw/decoded characters are ASCII; non-ASCII names are rejected regardless
of ZIP encoding flag. Accepted compression methods are Store and Deflate.
These rules eliminate Unicode/case/platform normalization ambiguity rather than
trying to repair it after publication.

`SKILL.md` starts with a YAML frontmatter mapping and closing `---`, followed by
a non-empty Markdown body:

```yaml
---
name: likec4
description: Build and validate LikeC4 models; use for .c4 and .likec4 output.
license: Apache-2.0                  # optional
compatibility: Contractor adk@1     # optional
metadata:                           # optional bounded string map
  author: contractor
---
```

`name` and non-empty `description` are mandatory. Description is at most 1,024
UTF-8 bytes; optional `license` is at most 512 and `compatibility` at most 500.
`metadata` has at most 32 entries. Keys are 1 through 64 ASCII characters
matching `[A-Za-z0-9_.-]+`; values are UTF-8 strings of at most 1,024 bytes.
Nested values, lists, non-string scalars and duplicate YAML keys are rejected.
The frontmatter mapping contains exactly `name`, `description` and the optional
`license`, `compatibility`, `metadata` keys. Unknown keys, non-string keys, YAML
aliases/anchors and custom tags are rejected.

The bytes between the two delimiters are at most 32 KiB. Bounded YAML
tokenization accepts at most 128 nodes and nesting depth 3 before constructing
the typed mapping; a deeply nested or node-heavy document cannot consume the
whole `SKILL.md` allowance in parser state. Byte/node overflow is
`skill_limit_exceeded`; a shape, type, tag or key violation is
`skill_manifest_invalid`.

Frontmatter `allowed-tools`/`allowed_tools` is rejected because it is not an
authorization grant. Metadata keys with prefix `adk_` are rejected, including
`adk_additional_tools` and `adk_inject_state`. Metadata has no executable or
authorization semantics. `license`, `compatibility` and `metadata` are
descriptive model-visible package data; Runtime does not interpret them as a
capability, placement predicate or dependency resolver. They contain no secret.

The complete `SKILL.md` is at most 256 KiB; one regular reference or asset is at
most 1 MiB. Text members are valid UTF-8 and have no NUL. In addition:

- stored Skill ZIP payload is at most 16 MiB, a package-specific limit below
  the generic ArtifactStore limit of 64 MiB;
- the ZIP contains at most 2,000 entries including directories;
- total declared and actually streamed decompressed bytes are at most 32 MiB.

The decompressed budget is enforced while streaming, not only from the
attacker-controlled central directory. Server and Runtime never silently drop
a forbidden member. Every package digest is lowercase `sha256:<64 hex>` over
the exact ZIP payload bytes, not a normalized directory tree.

The repository authoring helper emits a single canonical representation:
lexicographically ordered regular files using ZIP Store, fixed DOS-epoch
timestamps, portable mode `0644`, ASCII names and no directory entries, extra
fields or archive/member comments. Validation accepts ordinary safe Store or
Deflate archives, but bundled initialization uses this canonical writer so
unchanged source has stable bytes and digest.

Package validators use one bounded classification vocabulary across Go, Python
and the authoring command:

- `skill_archive_invalid` for malformed/encrypted/unsupported ZIP structure;
- `skill_path_invalid` for a non-portable, duplicate or escaping member path;
- `skill_member_forbidden` for a script, unknown root, link or special member;
- `skill_manifest_invalid` for encoding, frontmatter or body violations;
- `skill_name_mismatch` when directory/artifact/frontmatter names differ;
- `skill_limit_exceeded` for any per-member, count or aggregate bound.

Diagnostics may add a bounded field/member name only after it passed portable
path validation. They never include content, raw parser text or an absolute
filesystem path.

Skill packages are user/operator-managed behavioral configuration and contain
no secret. Validation cannot make instruction text benign: a skill may tell the
model to call any tool already selected by AgentTemplate. Hard authority remains
the explicit Toolset operation allowlist, allocation Artifact grant, write
fence, SandboxProfile and resolved RuntimeSettings. A skill widens none of them.

## Artifact-backed SkillCatalog and bundled initialization

Server has one internal `SkillCatalog` application boundary. It is a typed view
and validator over the ordinary ArtifactStore, not a new store: it owns bundled
source discovery, package validation, owner-UserScope resolution and exact
UserScope-to-RunScope skill forks. ArtifactStore remains authoritative for
bindings, revisions, bytes, CAS, lineage and retention. SkillCatalog has no
public/private HTTP endpoint or independent database schema.

After opening ArtifactStore and identifying the configured local owner, but
before Server reports ready, `SkillCatalog` scans immediate directories below
the `skills` subtree of the operator configuration root (the repository path is
`configs/skills`). It never scans the managed configuration root. Missing or
empty `skills` is valid. Each directory name is the expected skill name.
`SkillCatalog` rejects source symlinks and special files, validates every source
tree, deterministically packages it, then validates the ZIP again. The whole
discovered set is checked before the first write; an invalid bundled source is
a startup configuration error. At most 128 seed directories and 256 MiB total
canonical ZIP bytes are accepted. The validated discovery plan freezes those
exact canonical bytes; Artifact writes never re-read a source tree after the
all-or-nothing check.

Each package is initialized through the same ArtifactStore write semantics as
the public User Artifact API at:

```text
UserScope(<configured-local-owner>)
  namespace = skills
  name      = <directory and SKILL.md name>
```

Bundled catalog initialization is create-only:

- an absent binding is written with `If-None-Match: *` semantics and media type
  `application/vnd.contractor.agent-skill+zip`;
- an existing binding with identical exact ZIP digest and canonical Skill media
  type is `in_sync` and is not advanced;
- an existing binding with another digest or media type is reported as bounded
  `seed_drift` and remains current; Server still becomes ready.

This makes startup crash-safe and prevents restart from rolling back an update
made through the ordinary Artifact API. A changed bundled source is applied
explicitly by packaging it and performing an ordinary CAS Artifact PUT. There
is no synthetic user, special seed table, desired-state reconciler, Skill API or
SystemScope.

The initial deployment has one owner, matching the current public-auth model.
Provisioning shared/operator skills for future multiple owners is deferred and
must not be approximated by giving Runtime global Artifact authority.

If two Server processes accidentally initialize the catalog concurrently,
ordinary Artifact create CAS selects one winner. The loser re-reads current and
reports `in_sync` or `seed_drift`; it never overwrites or invents another
initialization mechanism.

## Ordinary Artifact update lifecycle

Skill upload/update uses the existing authenticated routes:

```text
PUT /v1/artifacts/skills/<name>
GET /v1/artifacts/skills/<name>
GET /v1/artifacts/skills/<name>/metadata
GET /v1/artifacts/skills/<name>/versions
```

Create uses `If-None-Match: *`; update uses the strong current revision ETag.
The response contains the new exact `ArtifactRef`. The generic Artifact API
enforces bytes, CAS, owner isolation and immutable history but deliberately does
not infer that an artifact is a Skill. The `contractor-skill` helper validates
before upload; Run initialization is the authoritative Server validation and
Runtime validates again before extraction.

Consequently an invalid current package may be stored as an ordinary artifact,
but no Run using it becomes schedulable. Updating it through CAS fixes future
Runs without mutating already initialized Runs. Existing artifact list/version
UI is sufficient for this increment; no `/operations/skills` endpoint, active
flag, skill-specific database table or skill-specific UI is introduced.

## Run resolution and fork

Run initialization resolves skills for every AgentTemplate retained by the
WorkflowRun snapshot, including templates used only by later Stages or pinned
escalation variants. Duplicate logical refs are resolved once. Before reading
package bodies, SkillCatalog selects the complete owner-current source set under
one short PostgreSQL snapshot. That selection outcome commits in the same
transaction as the WorkflowRun and its idempotency binding. Every logical ref
records either its exact current source or a durable missing-source result; no
committed Run exists with an unresolved/partial source list. A transient failure
aborts Run creation, while a committed missing-source result deterministically
fails that Run's initialization with `skill_artifact_not_found`. Every later
recovery step uses only the committed outcome even if owner current changes.

For each selected ref, Scheduler:

1. reads the already recorded exact artifact from the Run owner's UserScope;
2. requires the Skill media type, validates the complete package and checks its
   frontmatter name against the ArtifactRef name;
3. records the validated exact package digest beside the already pinned source
   ref in the immutable Run configuration snapshot;
4. forks that exact version, reusing the immutable blob, to reserved RunScope
   binding `skills/<name>`;
5. records the resulting exact RunScope ArtifactRef used by allocations.

Conceptually:

```text
UserScope(owner): skills/likec4@revision-3
  -> RunScope(run-42): skills/likec4@revision-17
```

All validation and forks complete before the Run becomes `running`; a partially
initialized Run is never schedulable. Run-creation idempotency replay returns
the already selected revisions or the same missing-source failure even if UserScope
current advances. Every retry, later Stage and escalation in the Run uses the
same exact Run bindings.

`skills` is a reserved RunScope Namespace beside `inputs` and `outputs`.
Scheduler alone creates its bindings. The private allocation Artifact API
permits exact reads required by trusted Runtime loading and rejects every
write. `run-artifacts@1` filters this Namespace from model-visible
list/read/write operations in Runtime; Agent Skills access content only through
ADK functions.
The authenticated Run owner may inspect ordinary exact metadata, lineage and
bytes through existing public Artifact routes, just as it may inspect the owner
source package. That user-facing authority is not exposed to Worker model tools;
package bodies never enter lifecycle events or telemetry.

Retention pins both exact owner source and Run fork while a retained Run
snapshot refers to them. Ordinary current updates therefore never invalidate a
Run. Garbage collection of unreferenced historical packages is deferred.

In addition to count limits, each AgentTemplate's resolved set is bounded to 64
MiB stored package bytes and 128 MiB decompressed bytes. The complete Run union
is bounded to 256 MiB stored and 512 MiB decompressed. Server checks both before
scheduling; Runtime independently enforces the per-allocation bounds.

## Allocation contract

`AllocationSpec` carries a sorted `resolvedSkills` list outside the
digest-bearing AgentTemplate body because exact revisions are selected at Run
initialization:

```python
class ResolvedSkill(BaseModel):
    name: SkillName
    artifact: ArtifactRef       # exact RunScope skills/<name> revision
    package_digest: str         # "sha256:" + 64 lowercase hex of exact ZIP bytes


class AllocationSpec(BaseModel):
    # existing identity, template, policy and RuntimeSettings fields
    resolved_skills: list[ResolvedSkill]
```

The list contains exactly the refs declared by the selected AgentTemplate, each
once and with a non-null Run revision. Empty skills encode `resolvedSkills: []`.
Runtime rejects missing, extra, duplicate, versionless, wrong-Namespace,
name-mismatched or digest-malformed entries before creating Worker. Server
provenance retains owner source and Run refs; Runtime receives only exact
current-Run refs.

Making `resolvedSkills` mandatory is a lockstep private-wire increment for the
single-VM deployment. Upgrade first lets every non-terminal Run terminate and
releases its allocations, then replaces Server and Runtime Agent processes
together. Missing/new fields fail closed during prepare; rolling mixed-version
operation is not promised.

Skill bytes never appear in AllocationSpec, A2A content, registration,
heartbeat or Agent Card. Runtime fetches through the existing allocation-bound
private Artifact API.

## Runtime and ADK mapping

After slot, artifact grant and allocation workspace exist but before Worker is
ready, Runtime Agent:

1. reads every exact `resolvedSkills[].artifact` through its private client;
2. requires the media type and recomputes exact package SHA-256;
3. applies complete archive/profile validation again;
4. extracts below an allocation-owned temporary directory named by validated
   skill name without following links or overwriting paths;
5. loads through the pinned Google ADK Agent Skills loader;
6. attaches one native ADK SkillToolset through Contractor's script-free
   filter/instruction adapter to Worker.

Runtime passes `registry=None`, no `environment`, `code_executor`,
`skills_folder`, `additional_tools` or `tool_name_prefix`, and an exact native
tool filter containing only `list_skills`, `load_skill` and
`load_skill_resource`. AgentTemplate's ordinary Toolsets are constructed
independently; Skill activation can never make one of them appear dynamically.

`registry=None` is mandatory. Runtime does not expose ADK `search_skills`,
contact Google Cloud Skill Registry, scan host/global directories or resolve a
model-selected name. The only discoverable skills are exact Run-selected ones.

The stock ADK toolset also implements `run_skill_script` and injects generic
script guidance even without an executor. Contractor exposes neither. Its
adapter reuses the native list/load/resource tools and native Skill models, but
injects a bounded instruction that describes only `SKILL.md`, `references/`
and `assets/` plus those three available functions. Runtime creates no skill
executor. Startup compatibility probes fail if filtering no longer hides the
function or the generated instruction contains `run_skill_script`,
`search_skills` or `scripts/` guidance.

Runtime callbacks validate a tool-supplied skill name against the exact
selected set before native dispatch. Resource paths must be portable validated
`references/...` or `assets/...` paths. Oversize, absolute, backslash or
dot-segment arguments return bounded `INVALID_ARGUMENTS` and are never retained
as a safe metric path.

ADK injects generic skill-use instructions and function declarations while
forming an LLM request. `list_skills` discloses names/descriptions; `load_skill`
returns selected `SKILL.md` instructions as a tool response;
`load_skill_resource` returns one selected reference or asset. Contractor does
not copy every body into system instructions, synthesize activation calls or
automatically activate a skill.

`list_skills`, `load_skill` and `load_skill_resource` share a 16 MiB
allocation-lifetime disclosure budget. Before native dispatch, a Runtime-owned
ADK callback computes and atomically reserves a deterministic conservative
charge no smaller than the complete model-visible result: UTF-8 structured
envelopes include descriptions/frontmatter, and binary resources include their
encoded representation. Repeated calls reserve again; reservations are not
refunded when native dispatch later returns an error. A call that would exceed
the budget returns bounded `SKILL_DISCLOSURE_LIMIT` before activation or content
lookup, reveals no partial content and still counts against the same
`maxToolCalls` and ModelPolicy limits as a Contractor Toolset call. A non-empty
skills set therefore requires a Worker ModelPolicy with `maxToolCalls` even
when `toolsets: []`. Per-member bounds keep estimation and one native result
bounded; release destroys the counter with other allocation State.

Runtime's `adk@1` startup probe proves the pinned ADK build contains package
models, loader and native SkillToolset before advertising `adk@1`. Runtime does
not advertise installed skill names because packages are allocation data, not
immutable environment capabilities.

An immutable cache keyed only by verified package digest is allowed. It is
bounded, never resolves a logical/current binding, and contains no User/Run
authority, secret, ADK State or conversation data. Cache hit still verifies
manifest name and exact digest.

Release removes extracted allocation directories, loaded Skill objects,
SkillToolset activation State and model context with Worker. Cleanup failure
keeps Runtime fenced under the existing release contract. A bounded immutable
package cache may remain.

## Planner and routing boundary

AgentTemplate skills apply only to Worker created from that template.
Passthrough, Streamline and Router Planners do not inherit Worker skills and do
not receive their descriptions or bodies. Router roster still uses logical
Agent name and `AgentTemplate.description`; it cannot inspect skills to route.

Planner-specific skills, Stage-selected skills, global search and dynamic
changes inside an allocation are deferred. Planner rules remain in Stage
`instructions`; mandatory Worker rules remain in AgentTemplate `instructions`.

## Failures, metrics and observability

Run initialization cannot become schedulable when a referenced owner artifact
is missing, has wrong media type, is invalid or cannot be forked. It records a
stable code and logical ref but no body, host path or parser exception. Invalid
content is a configuration error; transient ArtifactStore failures use ordinary
initialization recovery without resolving a different current revision after
an exact source has been recorded.

Terminal WorkflowRun initialization codes are
`skill_artifact_not_found`, `skill_media_type_invalid`, any exact package
validator code defined above, or `skill_fork_conflict`. Transient exact
reads/forks use `skill_artifact_unavailable` or `skill_fork_failed`, leave the Run
`initializing` and follow the same Scheduler recovery as an input fork; they do
not select current again after Run creation committed. A transient failure of
the source-selection transaction itself commits no Run/idempotency row and is a
retryable `POST /v1/runs` failure rather than a partially initialized Run.

Allocation preparation distinguishes at least:

- `skill_artifact_unavailable`, retryable for transient exact private reads;
- `skill_digest_mismatch`, non-retryable for the recorded Run snapshot;
- an exact package validator code, non-retryable for the recorded Run snapshot;
- `skill_runtime_unsupported`, non-retryable placement/configuration failure.

Preparation failure creates no ready Worker, Planner or A2A Task and follows
existing Stage preparation termination/retry. A model-issued load/resource
error is an ordinary bounded Worker tool error and cannot bypass policy.

Existing Worker metrics record these ADK functions like every tool. Retained
projection contains only validated skill name, bounded normalized
`references/...` or `assets/...` path, outcome, duration, result byte count and
stable ADK error code. It never retains instructions, resource/archive bytes,
owner refs, revisions, extracted paths or generated context. Aggregate counts
flow through ordinary `ExecutionReport`; no Skill telemetry service exists.

## Initial increment acceptance

1. valid `configs/skills` sources initialize missing configured-owner artifacts,
   survive crash replay and never overwrite an Artifact API update;
2. ordinary Artifact PUT creates a new exact skill ArtifactRef, a future Run
   uses it and an existing Run retains its prior source/fork;
3. two AgentTemplates share one logical owner artifact while one Run forks it
   once and each allocation receives only its declared subset;
4. Runtime fetches exact RunScope packages, verifies them, constructs native
   SkillToolset and Worker can list, load and read a reference;
5. dual validation rejects traversal, links, duplicates, oversize, scripts and
   behavioral ADK metadata;
6. generic Run Artifact tools cannot observe or mutate reserved `skills`, and a
   stale/fenced allocation cannot advance it;
7. `run_skill_script`, `search_skills` and script-bearing ADK guidance are
   absent from the model-visible request;
8. retry, later Stage and escalation use Run-pinned package after owner current
   advances;
9. release removes extraction and ADK activation State;
10. repeated list/load/resource disclosure calls stop at the allocation budget;
11. metrics expose bounded names/counts/outcomes but no content or refs.

## Deliberately deferred

- scripts and `run_skill_script` execution;
- Skill-owned executor, shell, Environment or SandboxProfile extension;
- executable dependency probing, script limits or script-specific placement;
- `allowed-tools`, `adk_additional_tools`, `adk_inject_state` or Skill-expanded
  Toolset authority;
- remote/dynamic registry search or model-chosen unpinned package;
- Workflow-, Stage-, Run- or Planner-selected skills;
- hot replacement inside an active Run/allocation;
- multi-owner shared/system package provisioning;
- special Skill API/UI, editor, marketplace, signing/trust federation;
- delete/retire and garbage collection of retained historical packages.

Enabling scripts later requires an explicit working agreement covering executor
and sandbox authority, Runtime capability matching, limits, metrics and
migration. Upgrading ADK or accepting one ZIP member never enables execution.

## Invariants

1. AgentTemplate selects versionless owner UserScope ArtifactRefs; Run pins
   exact immutable source and RunScope refs before scheduling.
2. Ordinary Artifact CAS update affects future Runs only and requires no
   Runtime Agent file synchronization or restart.
3. Runtime receives exact Run refs, never UserScope/global authority or inline
   package bytes.
4. Skills use native ADK progressive disclosure and have no MemoryTools adapter.
5. Skill content cannot add tools, broaden grants, change labels or escape the
   selected sandbox/tool authority.
6. Initial packages contain no scripts and no executable ADK metadata.
7. Planner and A2A Agent Card contracts remain independent of Agent Skills.
8. `configs/skills` seeds `SkillCatalog` create-only; database current wins on
   drift.
9. SkillCatalog is an internal ArtifactStore-backed boundary; no Skill-specific
   API, SystemScope or persistence model is introduced.
