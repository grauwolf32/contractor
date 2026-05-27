---
description: Trace-and-annotate skill — sink/source taxonomy, finding shapes (A/B/C), per-handler control checklist, annotation discipline. Detail files live under `trace/references/<topic>` — read on demand via skills_read.
---

# Trace Annotation Skill

Use this skill to trace handler/request execution paths through a project,
annotate relevant functions, and report only vulnerabilities supported by
visible code evidence.

The goal is not to find every suspicious pattern. The goal is to produce
a defensible trace: where input enters, how it changes, where it reaches
sensitive operations, which controls are present or missing, and which
finding shape (if any) is supported.

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

Load `trace/references/sources` the first time argument-state decisions
become uncertain.

### Phase 2 — Identify sinks precisely

When a call performs a side effect or sensitive operation, decide
whether it is a sink and label it precisely (e.g. `db.query` vs
`db.query.raw`, `shell.exec.args` vs `shell.exec`).

Load `trace/references/sinks` before labeling an unfamiliar sink.

### Phase 3 — Annotate relevant functions

Annotate only the entrypoint, validation/sanitization point, sink
wrapper, key transformation, or authorization decision. Use only the
recognized annotation forms. Before ending annotation work, run
`changed_paths` + `diff` and confirm only intended files/lines changed.

Load `trace/references/annotations` for forms, syntax, and placement.

### Phase 4 — Walk the per-handler control checklist

Mark each row as `present (file:line)`, `absent`, `weak`, or `N/A`.
Inspect middleware, decorators, router wrappers, policy helpers, and
schema validators before declaring a control absent.

Load `trace/references/controls` before composing the checklist.

### Phase 5 — Classify findings by shape

Each finding must match exactly one shape.

| Shape | Use when                                                                                                               |
| ----- | ---------------------------------------------------------------------------------------------------------------------- |
| A     | tainted/derived input controls structure/path/command/query shape at a sink, and a blocking control is missing or weak |
| B     | a sensitive operation is reachable without a required control                                                          |
| C     | a sensitive value is stored, transmitted, logged, or returned without required protection                              |

Load `trace/references/finding-shapes` before reporting.

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
- **Shape B and C are first-class.** Do not force a taint→sink story
  onto access-control gaps, plaintext secrets, cookie flag defects,
  logging leaks, or unfiltered responses.
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
table, and propagation cheatsheet → `trace/references/sources`.

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
`trace/references/sinks`:

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

Full per-shape mechanics, required fields, severity/confidence
heuristics, and `report_vulnerability` field mapping →
`trace/references/finding-shapes`.

## Quick triage — which reference to load

Pick by the signal you observe on the traced path. Load each reference
ONCE, the first time its topic becomes the current step.

| Signal observed                                                | Load                          |
| -------------------------------------------------------------- | ----------------------------- |
| Need to assign argument state, or unsure if a value is tainted | `references/sources`          |
| A reached call may be a sink; need to label it precisely       | `references/sinks`            |
| Comment form unclear, or mixed-language source on the path     | `references/annotations`      |
| About to compose §7 — need control checklist rows              | `references/controls`         |
| About to report — need shape, fields, severity, slug rules     | `references/finding-shapes`   |
| Tracing a Spring/Django/Go app; need routing or DI patterns    | `references/frameworks`       |

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
| `references/sinks`            | Identifying a sink; need its per-sink vulnerability checklist        |
| `references/sources`          | Deciding argument states; need taint-source taxonomy                 |
| `references/finding-shapes`   | Reporting a finding; need shape mechanics, fields, tool-field mapping |
| `references/controls`         | Walking the per-handler control checklist before reporting           |
| `references/annotations`      | Placing annotations; resolving comment syntax / format ambiguity     |
| `references/frameworks`       | Spring/Django/Go routing, DI, and service-layer patterns             |
