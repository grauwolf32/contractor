---
description: Trace-and-annotate skill — sink/source taxonomy, finding shapes (A/B/C), per-handler control checklist, annotation discipline. Detail files live under `trace/references/<topic>` — read on demand via skills_read.
---

# Trace Annotation Skill

Body of knowledge for tracing handler/request execution paths through a project, annotating relevant functions, and reporting vulnerabilities supported by visible code.

## Workflow at a glance

1. Identify the entrypoint for the assigned target.
2. Trace tainted/derived values through calls until a sink or terminal operation.
3. Annotate entrypoint, validation points, sinks, and key transformations.
4. Walk the per-handler control checklist (`trace/references/controls`).
5. Classify each finding into Shape A / B / C (`trace/references/finding-shapes`).
6. Report only what evidence supports.

## Key invariants

- Reaching a sink alone is NOT a vulnerability. Shape A requires the input
  to control structure (path, command, query shape) AND a missing/weak
  blocking control.
- Shape B and C are first-class. Do not force a taint→sink narrative on a
  missing-control or at-rest defect.
- Annotate only on visible code evidence. Never invent calls or files.
- Defects on opened files outside the traced flow (hardcoded secrets,
  plaintext password storage, cookie flag gaps, weak crypto algorithms)
  are Shape C — report once with a deterministic slug.

## Sink categories (high-level)

Names only — full catalogue and per-sink checklists in
`trace/references/sinks`:

  DATABASE  •  FILESYSTEM  •  PROCESS  •  NETWORK  •  RENDERING
  SERIALIZATION  •  CACHE / QUEUE  •  CRYPTO / SECRETS
  AUTH / AUTHZ  •  REFLECTION  •  OBSERVABILITY  •  IPC / INTER-SERVICE

## Argument states

One-liner: `tainted | validated | clean | derived`. Mark `tainted` /
`validated` only with explicit code evidence; default values transformed
from a tainted input to `derived` (still tainted) until validation is
visible.

Full taxonomy, decision table, and propagation cheatsheet →
`trace/references/sources`.

## Finding shapes (one-liners)

  Shape A — tainted/derived input controls structure at a sink, AND a
            blocking control is missing or weak.
  Shape B — sensitive operation reachable without a required control
            (auth, authz, signature_verify, expiry_check, role_check,
            ownership_check, csrf).
  Shape C — sensitive value at rest / in transit / in response without
            protection (plaintext credentials, hardcoded secrets, cookie
            flag gaps, unfiltered response, secret logged, weak randomness).

Full per-shape mechanics, required fields, and reporting template →
`trace/references/finding-shapes`.

## Reference index

| File                          | Load when working on...                                              |
| ----------------------------- | -------------------------------------------------------------------- |
| `references/sinks`            | Identifying a sink; need its per-sink vulnerability checklist        |
| `references/sources`          | Deciding argument states; need taint-source taxonomy                 |
| `references/finding-shapes`   | Reporting a finding; need shape mechanics and required fields        |
| `references/controls`         | Walking the per-handler control checklist before reporting           |
| `references/annotations`      | Placing annotations; resolving format ambiguity                      |
