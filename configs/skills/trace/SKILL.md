---
name: trace
description: "Trace request and data-flow paths, classify sinks and controls, annotate evidence, and shape defensible security findings."
compatibility: "Contractor adk@1 native Agent Skill disclosure; requires source browsing; editing and reporting steps apply only when those tools are available"
metadata:
  source-revision: 9c76b56cf7b83377fb1dd5e4a17440fa27b723f3
---

# Trace Annotation Skill

Use this skill to trace handler/request execution paths through a project,
annotate relevant functions with the dedicated annotation operations when they
are available, and report only vulnerabilities supported by visible code
evidence.

The goal is not to find every suspicious pattern. The goal is to produce
a defensible trace: where input enters, how it changes, where it reaches
sensitive operations, which controls are present or missing, and which
finding shape (if any) is supported.

This `SKILL.md` is loaded only after `load_skill(skill_name="trace")`.
References are disclosed separately and only when needed with
`load_skill_resource(skill_name="trace", file_path="references/<topic>.md")`;
they are not injected automatically. Use only source, edit, and reporting
operations exposed in the current Worker invocation.

## Operating principles

1. **Trace before judging.** Do not classify a vulnerability until the
   relevant path, sink, and controls have been inspected.
2. **Evidence beats naming.** Function names, decorators, middleware
   names, and helper names are hints only. Open the implementation
   before relying on them.
3. **A sink is not a bug.** Reaching a sink is expected in real
   applications. A finding requires a visible exploit mechanic or
   missing protection.
4. **Controls are per-handler.** A control on a sibling route does not
   protect the assigned target unless the target reaches the same
   control.
5. **Annotations describe visible code.** Never invent files, calls,
   arguments, middleware, or framework behavior.
6. **Uncertainty lowers confidence.** Do not upgrade a finding because
   it "probably" works a certain way.
7. **Empty findings are valid.** If the code does not support a
   vulnerability, report none.

## Workflow

### Phase 0 — Pin the target

Identify the assigned handler, route, operation, job, consumer, or
entrypoint. Record target id, route/trigger, file path and line range,
request arguments, and middleware/decorators/router wrappers.

If the target is ambiguous, use `target=unknown` in annotations rather
than inventing an id.

### Phase 1 — Trace values

For each request-derived or externally influenced value, assign an
argument state (`tainted | validated | clean | derived`) and follow
calls that transform, validate, persist, emit, render, authorize, or
otherwise use the value. Stop at a sink, a terminal response, a
blocking control, or visibly unreachable code.

Call `load_skill_resource(skill_name="trace", file_path="references/sources.md")`
the first time argument-state decisions become uncertain.

### Phase 2 — Identify sinks precisely

When a call performs a side effect or sensitive operation, decide
whether it is a sink and label it precisely (e.g. `db.query` vs
`db.query.raw`, `shell.exec.args` vs `shell.exec`).

Call `load_skill_resource(skill_name="trace", file_path="references/sinks.md")`
before labeling an unfamiliar sink.

### Phase 3 — Annotate relevant functions

When annotation tools are available, annotate only the entrypoint,
validation/sanitization point, sink wrapper, key transformation, or
authorization decision. Use `annotate_trace`, `annotate_validate`, and
`annotate_sink`; do not emulate them with a generic source edit. Before ending
annotation work, use `changed_paths` and `diff` and confirm only intended
files/lines changed. Use `rollback_changes` if the resulting change set is not
safe. If annotation tools are unavailable, report proposed locations without
claiming that source was modified.

Call `load_skill_resource(skill_name="trace", file_path="references/annotations.md")`
for forms, syntax, and placement.

### Phase 4 — Walk the per-handler control checklist

Mark each row as `present (file:line)`, `absent`, `weak`, or `N/A`.
Inspect middleware, decorators, router wrappers, policy helpers, and
schema validators before declaring a control absent.

Call `load_skill_resource(skill_name="trace", file_path="references/controls.md")`
before composing the checklist.

### Phase 5 — Classify findings by shape

Each finding must match exactly one shape.

| Shape | Use when                                                                                                               |
| ----- | ---------------------------------------------------------------------------------------------------------------------- |
| A     | tainted/derived input controls structure/path/command/query shape at a sink, and a blocking control is missing or weak |
| B     | a sensitive operation is reachable without a required control                                                          |
| C     | a sensitive value is stored, transmitted, logged, or returned without required protection                              |
| D     | a money/quantity/state/workflow operation violates an application invariant (no atomicity/idempotency, client-trusted value, step bypass) — controls present, logic broken |

Call `load_skill_resource(skill_name="trace", file_path="references/finding-shapes.md")`
before reporting. Shapes C and D
are the most-missed: run the **response/at-rest sweep** (Shape C) and the
**invariants** check (Shape D) on every handler that returns/persists data
or moves money/quantity/state — they need no taint flow.

### Phase 6 — Verify evidence

Before reporting, ask:

- Is every cited line on the traced path or at the defect site?
- Is the sink label exact?
- Is the missing/weak control visible in code?
- Did I inspect decorators/middleware/wrappers that may supply the control?
- Is this finding duplicated elsewhere?
- Would the finding still be defensible using only the cited evidence?

If not, downgrade confidence or do not report.

## Key invariants

- **Shape A requires structural control.** Input must influence path,
  command, query structure, template selection, URL/host, field set, or
  equivalent structure. Merely passing user data as a bound value is
  not enough.
- **Parameterized ORM/driver calls are not raw SQL.** Mislabeling these
  as `db.query.raw` / `db.exec.raw` is the #1 false-positive source.
- **Shape B, C, and D are first-class.** Do not force a taint→sink story
  onto access-control gaps, plaintext secrets, cookie flag defects,
  logging leaks, unfiltered responses, or business-logic invariant breaks
  (race/idempotency/client-trusted value/workflow bypass).
- **Validation is not authorization.** A schema may prove a field is
  well-formed; it does not prove the caller may access the resource.
- **Authentication is not ownership.** A valid user identity must still
  be compared to the target resource when ownership matters.
- **Sibling handlers do not count.** Re-check the assigned handler even
  when nearby handlers implement the control correctly.
- **Hardcoded secrets and at-rest defects are Shape C.** If discovered
  in opened files outside the traced flow, report once using a
  deterministic slug.
- **Do not rely on absence of search results.** Open relevant routing,
  middleware, decorators, and helper implementations before claiming a
  control is absent.

## Argument states

`tainted | validated | clean | derived` — full taxonomy, decision
table, and propagation cheatsheet → `references/sources.md`.

- `tainted` — value comes from an external or user-controlled source.
- `validated` — visible code enforces a schema, type, format, range,
  allowlist, or sanitizer.
- `clean` — value is a trusted constant, server-generated value, or
  internal-only value.
- `derived` — value is computed from other values; if any input is
  tainted, the derived value remains attacker-influenced unless
  validation is applied.

## Sink categories (high-level)

Names only — full catalogue and per-sink checklists in
`references/sinks.md`:

  DATABASE  •  FILESYSTEM  •  PROCESS  •  NETWORK  •  RENDERING
  SERIALIZATION  •  CACHE / QUEUE  •  CRYPTO / SECRETS
  AUTH / AUTHZ  •  REFLECTION  •  OBSERVABILITY  •  IPC / INTER-SERVICE

## Finding shapes (one-liners)

  Shape A — tainted/derived input controls structure at a sink, AND a
            blocking control is missing or weak.
  Shape B — sensitive operation reachable without a required control
            (auth, authz, signature_verify, expiry_check, role_check,
            ownership_check, csrf).
  Shape C — sensitive value at rest / in transit / in response without
            protection (plaintext credentials, hardcoded secrets, cookie
            flag gaps, unfiltered response, secret logged, weak randomness).
  Shape D — money/quantity/state/workflow operation violates an application
            invariant despite controls being present (missing atomicity →
            balance/quota race, missing idempotency → double-spend, negative/
            unbounded amount, client-trusted price/role, workflow-step bypass).

Full per-shape mechanics, required fields, severity/confidence
heuristics, and generic finding-record field mapping →
`references/finding-shapes.md`.

## Quick triage — which reference to load

Pick by the signal you observe on the traced path. Load each reference
ONCE, the first time its topic becomes the current step.

| Signal observed                                                | Load                          |
| -------------------------------------------------------------- | ----------------------------- |
| Need to assign argument state, or unsure if a value is tainted | `references/sources.md`          |
| A reached call may be a sink; need to label it precisely       | `references/sinks.md`            |
| Comment form unclear, or mixed-language source on the path     | `references/annotations.md`      |
| About to walk the per-handler control checklist                | `references/controls.md`         |
| About to report — need shape, fields, severity, slug rules     | `references/finding-shapes.md`   |
| Writing the `details` field — need the class → CWE ID + sink   | `references/cwe-mapping.md`      |
| Tracing a Spring/Django/Go/Express/Laravel/WordPress app; need its routing, DI, or **authz-control primitive** | `references/frameworks.md`       |

## Common false-positive traps

| Trap                                                            | Right call                                                                  |
| --------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Driver-bound placeholders treated as `db.query.raw`             | `db.query` / `db.exec` — parameterized; not Shape A                         |
| ORM filters treated as string-built SQL                         | Verify whether the ORM parameterizes values before labeling raw             |
| `subprocess.run([...], shell=False)` treated as `shell.exec`    | `shell.exec.args` — no shell; injection mechanic absent                     |
| Schema validation read as authorization                         | input_validation `present`; authorization still evaluated separately        |
| Auth in middleware satisfies ownership check                    | authentication `present`; ownership/scoping evaluated separately            |
| Type cast with default (`int(x or 0)`) treated as validated     | still tainted/derived — no rejection branch                                 |
| Decorator name treated as evidence                              | open the decorator; name alone is not a control                             |
| Sibling handler's projection assumed for this handler           | re-verify; otherwise output_filter `absent` here                            |
| Sensitive field leak forced into Shape A                        | use Shape C with `output_filter` when the defect is unfiltered response data |
| Missing search hit treated as missing control                   | inspect routing/middleware/wrappers first; confidence depends on visibility  |

## Reference index

| File                          | Load when working on...                                              |
| ----------------------------- | -------------------------------------------------------------------- |
| `references/sinks.md`            | Identifying a sink; need its per-sink vulnerability checklist        |
| `references/sources.md`          | Deciding argument states; need taint-source taxonomy                 |
| `references/finding-shapes.md`   | Reporting a finding; need shape mechanics, fields, tool-field mapping |
| `references/controls.md`         | Walking the per-handler control checklist before reporting           |
| `references/annotations.md`      | Placing annotations; resolving comment syntax / format ambiguity     |
| `references/cwe-mapping.md`      | Writing the `details` field; need the class → primary CWE + sink label |
| `references/frameworks.md`       | Routing + authz-primitive per framework (Spring/Django/Go/Express/Laravel/WordPress) |
