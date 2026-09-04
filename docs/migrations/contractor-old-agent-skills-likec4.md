# LikeC4 Agent Skill migration

The initial LikeC4 package was migrated from the exact contractor-old revision
`9c76b56cf7b83377fb1dd5e4a17440fa27b723f3`. No code, Memory adapter, tool
implementation, generated file, or other Skill directory was copied.

The canonical package digest for this migrated source tree is
`sha256:84bc32ac3f6ca32d13785280090701e1d54a3e4f0cd373572236f0a2cd22b95e`.
`contractor-skill validate` derives that digest from a deterministic canonical
archive; the logical AgentTemplate reference remains the versionless
`skills/likec4` binding so a Run can pin the currently published exact revision.

## Complete source inventory

Every file below is accounted for exactly once.

| Pinned contractor-old path | New package path |
| --- | --- |
| `contractor/skills/likec4/index.md` | `configs/skills/likec4/SKILL.md` |
| `contractor/skills/likec4/references/cli.md` | `configs/skills/likec4/references/cli.md` |
| `contractor/skills/likec4/references/configuration.md` | `configs/skills/likec4/references/configuration.md` |
| `contractor/skills/likec4/references/deployment.md` | `configs/skills/likec4/references/deployment.md` |
| `contractor/skills/likec4/references/dynamic-views.md` | `configs/skills/likec4/references/dynamic-views.md` |
| `contractor/skills/likec4/references/examples.md` | `configs/skills/likec4/references/examples.md` |
| `contractor/skills/likec4/references/identifier-validity.md` | `configs/skills/likec4/references/identifier-validity.md` |
| `contractor/skills/likec4/references/include-predicates-wildcards.md` | `configs/skills/likec4/references/include-predicates-wildcards.md` |
| `contractor/skills/likec4/references/model.md` | `configs/skills/likec4/references/model.md` |
| `contractor/skills/likec4/references/predicates.md` | `configs/skills/likec4/references/predicates.md` |
| `contractor/skills/likec4/references/relationships-bidirectional.md` | `configs/skills/likec4/references/relationships-bidirectional.md` |
| `contractor/skills/likec4/references/specification.md` | `configs/skills/likec4/references/specification.md` |
| `contractor/skills/likec4/references/style-tokens-colors.md` | `configs/skills/likec4/references/style-tokens-colors.md` |
| `contractor/skills/likec4/references/troubleshooting.md` | `configs/skills/likec4/references/troubleshooting.md` |
| `contractor/skills/likec4/references/views.md` | `configs/skills/likec4/references/views.md` |

## Intentional text changes

- `index.md` became the required `SKILL.md` and gained bounded `name`,
  `description`, `compatibility`, and pinned source-revision frontmatter.
- Old `skills_read` calls, extensionless paths, and the assumption that an index
  is always in model context became native progressive-disclosure calls:
  `load_skill` followed by
  `load_skill_resource(skill_name="likec4", file_path="references/<topic>.md")`.
- General LikeC4 DSL, configuration, and multi-file knowledge was retained, but
  Contractor-specific procedure now states the actual `likec4@1` contract: one
  durable self-contained artifact, a no-argument `validate_likec4()` call, an
  installed `likec4` executable, a fixed isolated invocation, and bounded
  validation results. Stale path, overlay, package-runner fallback, and
  multi-file result claims were removed.
- The new v2 Worker instructions retain mandatory source/artifact handling,
  phase ordering, evidence rules, repair bounds, validation, result media types,
  and finish conditions. Detailed DSL examples and troubleshooting move to
  optional Skill disclosure.

## Current configuration after catalog consolidation

The migration was initially introduced through additive versions. Once durable
Runs stored complete resolved snapshots, those rollout-only predecessors no
longer served a runtime compatibility purpose and were deleted rather than
archived. Deleted identities are never reused.

The current archive-backed entry is `likec4-from-analysis@3`, using
`likec4_builder@2` and `likec4_validator@2`. The current source-to-document
entries are `likec4-from-workspace@4` and
`likec4-from-workspace-streamline@2`; their workspace-specific builder and
validator templates also select `skills/likec4`. Dependency/project discovery
uses `workspace_source_graph_analyst@1` and remains unskilled.
