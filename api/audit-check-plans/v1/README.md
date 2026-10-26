# Typed accepted Audit plans

The registry maps `contractor.audit.finding-check-plan.v1` methods `verify`,
`static-trace` and `http-probe` to phase `verify`; `replay`, `applicability` and
`alternative` each have their same-named phase. A trusted receipt adapter reads
the real proposed-check method. A role name or arbitrary Workflow reference
never registers semantics.

`accepted-plan.schema.json` describes the immutable task/execution provenance.
The full dependency manifest remains in the existing `dependencies` field;
`dependencies_digest` pins its RFC 8785 bytes or is explicitly null. Plan and
manifest headers, exact proposal source, receipt/ordinal, consumer pins, route,
policy and required review agree before any child intent. Shape validation is
not source authorization. Review subjects and decisions remain outside task
bytes because their identity includes the final task digest.

`finding-source.schema.json` describes a bounded verified receipt fragment and
context. Each typed task pins the actual exact source Artifact bytes. A paged
selection root seals source ordering; it does not replace per-task source refs.

Plan, dependency and accepted-plan digests use RFC 8785/JCS UTF-8 bytes. The
existing Workflow and Skill consumer pins remain Server-generated opaque
digests with their existing encoding. Go and Python shared cases include
Cyrillic, HTML characters, escaped quotes, newline, U+2028/U+2029, ordinal zero,
size zero and the optional fields of each closed dependency origin.

Limits are 64KiB accepted-plan metadata, 32 descriptors/256KiB dependencies,
128 profile mappings, 32 required Worker toolsets and 128 tools per requirement.
A fixed budget checkpoint is at most 4KiB. Paged admission retains actual
per-source and task codec limits and a 64MiB aggregate metadata budget.
