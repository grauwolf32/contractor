# Decisions and planning history

[Documentation index](../README.md)

| File | Contents | Status |
| --- | --- | --- |
| [fixed-decisions.yml](fixed-decisions.yml) | Settled architecture decisions: languages and stacks, the Worker model boundary, Planner modes, persistence, deployment and the boundaries of each subsystem | Normative. Change one only with the user, together with the specification |
| [planning-history.yml](planning-history.yml) | Planning and implementation reviews of each epic, ports of contractor-old, the 2026-09-06 delivery order and checkpoint summaries | Historical record. It explains why tasks look the way they do; it does not set current priorities |
| [open-questions.md](open-questions.md) | Unresolved design questions and contradictions between the legacy specifications found during the 2026-10 migration to the domain layout | Working list. Resolve each item in the owning specification and delete it here |
| [spec-history.md](spec-history.md) | Delivery plans, increments, acceptance narratives, implementation status and task references that the feature-ordered specifications carried until the migration | Historical record, verbatim |
| [spec-migration.md](spec-migration.md) | Where each legacy `docs/spec/NN-*.md` document went, with the share of its content per new document | Lookup table for older task records |

The two YAML files were moved verbatim from `tasks/index.yml` on 2026-10-11. Older task
records and reviews refer to their sections as `tasks/index.yml <section>`; the
section names are unchanged.

Current work lives elsewhere: the generated [backlog](../../tasks/BACKLOG.md),
open tasks in [tasks/](../../tasks/), finished tasks in
[tasks/done/](../../tasks/done/), accepted contracts in
[the specification](../spec/README.md).
