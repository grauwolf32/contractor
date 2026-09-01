# Live-testing Agent Skill migration

The `auth`, `caido`, `code-exec`, and `exploit` packages were migrated from the
exact contractor-old revision
`9c76b56cf7b83377fb1dd5e4a17440fa27b723f3`. Only the 19 Markdown knowledge
files listed below were copied. No legacy agent, workflow, prompt, Python
implementation, tool declaration, container image, sandbox, credential, or
network configuration was migrated.

The packages are deliberately unassigned. They are ordinary Skill artifacts,
not Runtime capabilities. Their model-visible text does not mention
AgentTemplate; server-side AgentTemplate selection remains the place where a
future configuration must combine a package with every compatible Toolset.

The canonical package digests are:

| Package | Digest |
| --- | --- |
| `auth` | `sha256:c7165c518840bf65cb2f139d9b06ed1aa240597356c20f081ab1ae2894a3f62f` |
| `caido` | `sha256:ab88f0a1f411c67b5060bb338928d2b7bd13b5096ed076d6d9b7ac5ae2448499` |
| `code-exec` | `sha256:ae482885e234465206e603a258463508998478d9845f1c323cfd2d2b5a7bd4d0` |
| `exploit` | `sha256:e44969fa40e36907273490e1f7e58743003d46612f1d5ac31bc9156c16276e3f` |

## Complete source inventory

Every source Markdown file is mapped exactly once.

| Pinned contractor-old path | New package path |
| --- | --- |
| `contractor/skills/auth/index.md` | `configs/skills/auth/SKILL.md` |
| `contractor/skills/caido/index.md` | `configs/skills/caido/SKILL.md` |
| `contractor/skills/code-exec/index.md` | `configs/skills/code-exec/SKILL.md` |
| `contractor/skills/exploit/index.md` | `configs/skills/exploit/SKILL.md` |
| `contractor/skills/exploit/references/auth-bypass.md` | `configs/skills/exploit/references/auth-bypass.md` |
| `contractor/skills/exploit/references/auth-discovery.md` | `configs/skills/exploit/references/auth-discovery.md` |
| `contractor/skills/exploit/references/broken-auth.md` | `configs/skills/exploit/references/broken-auth.md` |
| `contractor/skills/exploit/references/cmdi.md` | `configs/skills/exploit/references/cmdi.md` |
| `contractor/skills/exploit/references/idor.md` | `configs/skills/exploit/references/idor.md` |
| `contractor/skills/exploit/references/info-disclosure.md` | `configs/skills/exploit/references/info-disclosure.md` |
| `contractor/skills/exploit/references/mass-assignment.md` | `configs/skills/exploit/references/mass-assignment.md` |
| `contractor/skills/exploit/references/nosqli.md` | `configs/skills/exploit/references/nosqli.md` |
| `contractor/skills/exploit/references/path-traversal.md` | `configs/skills/exploit/references/path-traversal.md` |
| `contractor/skills/exploit/references/rate-limiting.md` | `configs/skills/exploit/references/rate-limiting.md` |
| `contractor/skills/exploit/references/sqli.md` | `configs/skills/exploit/references/sqli.md` |
| `contractor/skills/exploit/references/ssrf.md` | `configs/skills/exploit/references/ssrf.md` |
| `contractor/skills/exploit/references/ssti.md` | `configs/skills/exploit/references/ssti.md` |
| `contractor/skills/exploit/references/xss.md` | `configs/skills/exploit/references/xss.md` |
| `contractor/skills/exploit/references/xxe.md` | `configs/skills/exploit/references/xxe.md` |

## Server-side operation compatibility

The packages name an operation only so the model can use it when that exact
operation is present in its invocation. They neither register an operation nor
choose the factory that implements it. A future descriptor may use a different
Toolset ref, but it must export the exact model-visible name below or the package
must be revised and republished; this migration adds no alias.

| Model-visible operations | Used by | Current v2 status | Required contract before assignment |
| --- | --- | --- | --- |
| `load_skill`, `load_skill_resource` | `exploit` | Native bounded ADK SkillToolset operations | Already supplied only when the resolved package list is non-empty; they grant no domain capability. |
| `read_memory`, `write_memory` | `auth`, `exploit` | Implemented by `memory-tools@1` | Select the exact operations explicitly; notes use valid names such as `auth_creds`, not reserved paths. |
| `http_request` | all four packages | No current Toolset exports it | A future authorized HTTP Toolset must bind an allocation-local client to `runtime-http-client`, enforce its own bounds, and define evidence fields. |
| `http_session_set` | `auth`, `exploit` | No current Toolset exports it | A future HTTP Toolset must define allocation-local cookie/header/auth state and how later requests consume it. |
| `get_vulnerability`, `submit_verdict` | `code-exec`, `exploit` | No current Toolset exports them | A future finding/verdict Toolset must define exact input, durable output, idempotency, and optional `request_ids` semantics. |
| `run_python`, `execute_bash` | `code-exec`, `exploit` references | No current Toolset exports them | A future code-execution Toolset must use `runtime-subprocess-launcher` and an exact stronger execution contract; `local-workdir@1` promises no container, dependency, network isolation, or persistence. |
| `caido_replay`, `caido_automate_run`, `caido_history`, `caido_request_detail` | `caido`, `exploit` | No current Toolset exports them | A future Caido API Toolset must use an allocation-owned HTTP client and typed instance settings; applying the `caido` Runtime label or HTTP proxy alone must not add model-visible tools. |
| `caido_workflow_list`, `caido_workflow_run`, `caido_workflow_findings` | `caido` | No current Toolset exports them | The same future Caido Toolset must define instance-specific IDs, bounded polling/results, and enabled-workflow semantics. |

`request_id`, `request_tag`, and `request_ids` are conditional data fields, not
operations. Package text now tells the model to use them only when the visible
operation schemas actually provide them.

## Intentional platform-integration edits

- Every `index.md` became a valid `SKILL.md` with a portable name, bounded
  description and compatibility text, plus the pinned source revision.
- `exploit` now uses native `load_skill_resource` calls and exact
  `references/<name>.md` paths. Optional resources are never described as
  automatically injected.
- The old reserved Memory convention (`auth/creds`, `auth/user1`, and related
  paths) became optional `memory-tools@1` calls with valid logical names such as
  `auth_creds`, `auth_user1`, and `exploit_probe_log`. Absence of MemoryTools no
  longer implies hidden persistence.
- `code-exec` no longer promises an ephemeral Kali container, a `/project`
  mount, host networking, preinstalled packages, a persistent interpreter or
  working directory, background-process survival, or automatic artifact
  export. Every recipe is conditional on the visible operation's own schema and
  environment description.
- Caido, HTTP, finding/verdict, and code-execution names remain recognizable for
  a future compatible Toolset, but all statements are conditional. No current
  descriptor, Runtime capability, configuration selector, or alias was added.
- Authorization, exact target scope, controlled identities, non-destructive
  proof, stop conditions, and cleanup now appear in the root instructions before
  live-testing procedure. The reviewed reference payload corpus and technical
  techniques were not expanded.

All fifteen `exploit` references otherwise remain byte-for-byte copies of the
pinned revision. This preserves the reviewed class knowledge while keeping the
new Runtime and authority boundary explicit in the always-loaded root.
