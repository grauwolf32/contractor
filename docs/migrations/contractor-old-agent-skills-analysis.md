# Analysis and security-review Agent Skill migration

The STRIDE, trace, vulnerability-scan, and vulnerability-playbook packages were
migrated from the exact contractor-old revision
`9c76b56cf7b83377fb1dd5e4a17440fa27b723f3`. The migration copies Markdown
knowledge only. It does not copy legacy agents, prompts, workflows, Python
implementations, Memory adapters, or tool declarations.

The packages are deliberately unassigned. No current AgentTemplate provides an
exact replacement for the legacy threat-analysis, trace, source-annotation,
finding-reporting, or authorized HTTP-testing surfaces. Publishing knowledge is
not treated as proof that a compatible Worker exists.

The canonical package digests for the migrated source trees are:

| Package | Digest |
| --- | --- |
| `stride` | `sha256:92cb91b0952fb419021e89ec5d977ae36b1ab6439d9f36f2b5240412ea530043` |
| `trace` | `sha256:245b3799afc85ab27cb55fdeb196f461e85e8a5c4ab5542e2a959b61fd5fec98` |
| `vuln-scan` | `sha256:504c68f2c72545ab190d9b79140ee74fcab7abee6d02a4039cdc41caada20b16` |
| `vulns` | `sha256:92dc4640426c8aa5f6374eed1b53775552fe14daf1becd89d7f456886212d274` |

The digest identifies exact package bytes. Server-side AgentTemplate selection
uses the logical versionless `skills/<name>` binding and is intentionally
outside the model-visible package contract.

## Complete source inventory

Every one of the 23 source Markdown files is mapped exactly once.
`vuln_scan` becomes `vuln-scan` because portable Agent Skill names cannot
contain underscores.

| Pinned contractor-old path | New package path |
| --- | --- |
| `contractor/skills/stride/index.md` | `configs/skills/stride/SKILL.md` |
| `contractor/skills/trace/index.md` | `configs/skills/trace/SKILL.md` |
| `contractor/skills/trace/references/annotations.md` | `configs/skills/trace/references/annotations.md` |
| `contractor/skills/trace/references/controls.md` | `configs/skills/trace/references/controls.md` |
| `contractor/skills/trace/references/cwe-mapping.md` | `configs/skills/trace/references/cwe-mapping.md` |
| `contractor/skills/trace/references/finding-shapes.md` | `configs/skills/trace/references/finding-shapes.md` |
| `contractor/skills/trace/references/frameworks.md` | `configs/skills/trace/references/frameworks.md` |
| `contractor/skills/trace/references/sinks.md` | `configs/skills/trace/references/sinks.md` |
| `contractor/skills/trace/references/sources.md` | `configs/skills/trace/references/sources.md` |
| `contractor/skills/vuln_scan/index.md` | `configs/skills/vuln-scan/SKILL.md` |
| `contractor/skills/vuln_scan/references/absence-detection.md` | `configs/skills/vuln-scan/references/absence-detection.md` |
| `contractor/skills/vuln_scan/references/business-logic.md` | `configs/skills/vuln-scan/references/business-logic.md` |
| `contractor/skills/vuln_scan/references/checklist.md` | `configs/skills/vuln-scan/references/checklist.md` |
| `contractor/skills/vuln_scan/references/grep-patterns.md` | `configs/skills/vuln-scan/references/grep-patterns.md` |
| `contractor/skills/vuln_scan/references/miss-patterns.md` | `configs/skills/vuln-scan/references/miss-patterns.md` |
| `contractor/skills/vuln_scan/references/php-wordpress.md` | `configs/skills/vuln-scan/references/php-wordpress.md` |
| `contractor/skills/vuln_scan/references/secrets.md` | `configs/skills/vuln-scan/references/secrets.md` |
| `contractor/skills/vuln_scan/references/sink-patterns.md` | `configs/skills/vuln-scan/references/sink-patterns.md` |
| `contractor/skills/vulns/index.md` | `configs/skills/vulns/SKILL.md` |
| `contractor/skills/vulns/references/idor.md` | `configs/skills/vulns/references/idor.md` |
| `contractor/skills/vulns/references/ssrf.md` | `configs/skills/vulns/references/ssrf.md` |
| `contractor/skills/vulns/references/ssti.md` | `configs/skills/vulns/references/ssti.md` |
| `contractor/skills/vulns/references/xxe.md` | `configs/skills/vulns/references/xxe.md` |

## Intentional platform-integration edits

- Each legacy `index.md` became `SKILL.md` with an exact portable name, bounded
  description, compatibility statement, and pinned source revision.
- Legacy `skills_read`/`skills_list` wording and extensionless package paths
  became native `load_skill` followed by
  `load_skill_resource(skill_name="...", file_path="references/<file>.md")`.
  References are explicitly on-demand, never assumed to be in model context.
- `vuln_scan` identifiers became `vuln-scan` in the package body and native
  disclosure examples. Reference filenames and the security subject matter did
  not change.
- Old exact tool names such as `list_symbols`, `read_file`, `changed_paths`,
  and `report_vulnerability` became capability-neutral source-navigation,
  editing, and reporting operations. The text now requires an operation to be
  present in the current Worker invocation before it can be used; it does not
  expose the server-side AgentTemplate abstraction to the model.
- Trace annotation is conditional on a future Worker receiving edit tools. A
  read-only Worker must return proposed locations and may not claim mutations.
- The STRIDE package still models rather than exploits. The `vulns` package
  retains explicit authorization, non-destructive proof, evidence, stop, and
  scope-escalation boundaries. No payload, bypass, exploit chain, or offensive
  technique was added or broadened relative to the pinned source.

The reusable taxonomies, checklists, framework notes, vulnerability patterns,
playbooks, examples, and remediation guidance otherwise remain source-derived.
