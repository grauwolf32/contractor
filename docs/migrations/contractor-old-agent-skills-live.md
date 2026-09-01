# Live-testing Agent Skill migration

The `auth`, `caido`, `code-exec`, and `exploit` packages were migrated from the
exact contractor-old revision
`9c76b56cf7b83377fb1dd5e4a17440fa27b723f3`. Only the 19 Markdown knowledge
files listed below were copied. No legacy agent, workflow, prompt, Python
implementation, tool declaration, container image, sandbox, credential, or
network configuration was migrated.

The `auth`, `code-exec`, and `exploit` packages remain deliberately unassigned.
The `caido` package is assigned only by the checked-in `caido_analyst@1`
AgentTemplate, which explicitly selects every HTTP and Caido operation named by
that package. All four remain ordinary Skill artifacts, not Runtime
capabilities. Their model-visible text does not mention AgentTemplate;
server-side configuration and repository compatibility tests own the package
to Toolset association.

The canonical package digests are:

| Package | Digest |
| --- | --- |
| `auth` | `sha256:c7165c518840bf65cb2f139d9b06ed1aa240597356c20f081ab1ae2894a3f62f` |
| `caido` | `sha256:676d2d4736054dad6556a5a9f8fac49e7ffd89858bf9761fbe2517634b3459c1` |
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
| `http_request`, `http_read_body`, `http_history` | `caido` (`http_request` also appears in all four packages) | Implemented by `http-tools@1`; all three are selected by `caido_analyst@1` | The allocation-local client enforces bounded requests, artifact bodies and history. Other packages remain unassigned until a template selects their complete compatible surface. |
| `http_session_set` | `auth`, `exploit` | Implemented by `http-tools@1` but not assigned to either package | A future assigning template must select it explicitly together with the HTTP operations required by that package; session secrets remain allocation-local. |
| `get_vulnerability`, `submit_verdict` | `code-exec`, `exploit` | No current Toolset exports them | A future finding/verdict Toolset must define exact input, durable output, idempotency, and optional `request_ids` semantics. |
| `run_python`, `execute_bash` | `code-exec`, `exploit` references | No current Toolset exports them | A future code-execution Toolset must use `runtime-subprocess-launcher` and an exact stronger execution contract; `local-workdir@1` promises no container, dependency, network isolation, or persistence. |
| `caido_scope`, `caido_history`, `caido_request_detail`, `caido_replay`, `caido_automate_run`, `caido_automate_results`, `caido_sitemap` | `caido`, with a subset also named by `exploit` | Implemented by `caido@1`; the complete set is selected only by `caido_analyst@1` | The Toolset uses an allocation-owned static GraphQL adapter and bounded operation contracts. Applying a Runtime label alone still adds no model-visible operation. |
| `caido_workflow_list`, `caido_workflow_run`, `caido_workflow_findings` | `caido` | Implemented by `caido@1` and selected by `caido_analyst@1` | IDs remain instance-specific; calls use bounded inputs, polling/results and exact artifact output where required. |

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
- Caido guidance was revised against the exact bounded `caido@1` and
  `http-tools@1` schemas and is selected only by `caido_analyst@1`. This
  explicit configuration adds no dynamic dependency interpretation: removing a
  required operation fails the repository release gate. Finding/verdict and
  code-execution names remain conditional future compatibility points.
- Authorization, exact target scope, controlled identities, non-destructive
  proof, stop conditions, and cleanup now appear in the root instructions before
  live-testing procedure. The reviewed reference payload corpus and technical
  techniques were not expanded.

All fifteen `exploit` references otherwise remain byte-for-byte copies of the
pinned revision. This preserves the reviewed class knowledge while keeping the
new Runtime and authority boundary explicit in the always-loaded root.
