# v2 research hypotheses — 2026-10-23

[Research index](README.md)

- **Date:** 2026-10-23, checked against `76470fcd`.
- **Source:** the v2-native hypothesis catalog and the v2 mechanism map of
  2026-09-04, removed in `85b142fc`. The full records (claim, mechanism,
  guardrails, related legacy IDs) stay in git:
  `git show 85b142fc^:docs/research/hypotheses/v2-native.yaml`. The legacy
  catalog from the same commit (228 hypotheses in 51 directions plus 11
  Planner variants) is `git show 85b142fc^:docs/research/hypotheses/legacy.yaml`.
- **Kind:** non-normative experiment backlog. No experiment has been run for
  any of these hypotheses.
- **Outcome:** of 28 v2 hypotheses in eight directions, 15 can run on the
  current platform, 5 are blocked by a specification non-goal and 8 need a new
  capability. From the legacy catalog: six validity checks to run before any
  experiment, eleven ideas worth porting and the recorded legacy results that
  bear on v2 hypotheses.

## Status on 2026-10-23

Most of the evaluation foundation the catalog waited for now exists: Run
metadata labels (V20-001–006), Worker session modes (V19-001), typed findings
with fingerprints and correlation ([findings](../spec/findings/findings.md)),
and managed Evals with pinned plans, paired members and pass@N (V38-001–010,
[managed Evals](../spec/evals/managed-evals.md),
[evaluation format](../spec/evals/format.md)). The mechanism map is not
restored: the specification now describes those mechanisms.

How to run one:

- Each arm is an exact Workflow or Audit binding with its `executionConfig`
  (EVM-R53); an AgentTemplate change needs its own Workflow version (EVM-R54).
- Managed Evals run no arbitrary scorer (EVM-R55); their native evaluators
  check structure. Semantic scoring against hidden ground truth comes from an
  external evaluator, and the ground truth stays outside the project the agent
  scans.
- Freeze the effect threshold, sample count and decision rule before the first
  model call, and compare repeated samples: one sample is noise.
- Check that every metric a record names is actually reported before freezing
  its design.
- Change one mechanism per arm pair. The legacy catalog flagged an arm that
  changed several at once (PV2) because it cannot attribute an effect.

## Before any experiment

These checks come from the legacy catalog's evaluation-validity directions
(AO, AP, AR, AS, AY). Without them a measured difference between arms cannot
be told apart from noise, scorer artefacts or memorisation.

| Legacy ID | Check | Why | State on 2026-10-23 |
| --- | --- | --- | --- |
| AR1 | A/A Eval: both arms bound to the same Workflow version | Measures the noise floor of the serving stack (non-determinism at temperature 0 from batching); every effect must clear it | Nothing in the managed Evals specification or `internal/evaldomain` rejects identical arms; not tried |
| AY2 | Clean-target controls: fixtures without the vulnerability | Shows over-reporting that a vulnerable-only suite cannot measure | No such control in `docs/spec/evals` |
| AS1 | Semantics-preserving perturbations of public fixtures (renames, moves) | A recall drop means the model remembers a public benchmark rather than analysing the code | Not present |
| AR2, AP1 | Sweep scorer tolerances (sibling CWE, sink line offset) and adjudicate false positives that may be unlabelled real vulnerabilities | Matcher strictness and incomplete ground truth bias precision independently of the arm | Belongs in the external ground-truth evaluator that still has to be written |
| AO1, AO2 | Deterministic floor: semgrep, CodeQL or osv-scanner against the same fixtures and ground truth | Credits the model only above what deterministic tools already find | Not present |
| AR3 | Preregistration plus false-discovery-rate control across a batch | With 15 tests at α = 0.05, the chance of at least one false positive is about 54% | Preregistration is already required above; FDR control is new |

## Prerequisites then and now

| Prerequisite | 2026-09-04 | 2026-10-23 |
| --- | --- | --- |
| `eval_foundation` | Missing: labels, evidence export, frozen digests, one Run per sample | Done: V20-001–006, V38-001–010 |
| `worker_session_modes` | Specified, V19-001 pending | Done: `isolated` and `shared` (WRK-R65 onwards) |
| `security_eval_contract` | Missing: typed findings, verdicts, ground-truth matching | Platform part done (findings, assessments, FND-R163, FND-R174–178); a finding ground-truth scorer must be written as an external evaluator |
| `observation_profiles` | Missing | Blocked: only `lean@1` exists, and user-defined observation reducers and YAML A/B profiles are a Worker non-goal |
| `tool_evidence_outcomes` | Missing | Missing: no per-tool semantic outcome separate from transport success |
| `stateflow_prototype` | Research draft `stateflow-1.md` | No design left: the draft was removed and nothing implements it |

## Ready to run (15)

| ID | Hypothesis | Arms: control → treatment | Primary metric | Still needed |
| --- | --- | --- | --- | --- |
| AZ1 | Isolated Worker sessions flatten context growth | Stage session `shared` → `isolated` | `worker_input_token_slope_per_invocation` ↓ | Stages with ≥3 Worker invocations; per-invocation Worker tokens in reports |
| AZ2 | Shared sessions keep useful tacit context in tight refinement loops | `isolated` → `shared` | `refinement_case_completion_rate` ↑ | Inspect-edit-validate fixtures |
| BB1 | Terminal summarization rescues work before hard exhaustion | AgentTemplate without summarizer → with `terminal@1` | `valid_completion_rate_after_soft_boundary` ↑ | Cases that reach the trigger |
| BB2 | A cheaper dedicated summarizer is sufficient | Worker-family summary policy → smaller `summarizerModelPolicy` (TPL-R112) | `summarizer_cost_per_valid_result` ↓ | As BB1; non-inferiority margin |
| BB3 | Summarizer trigger timing has an interior optimum | Default trigger → grid of `contextWindowRatio` × `cumulativeBudget` | `quality_adjusted_completion_per_token` max | Preregistered grid |
| BC1 | Typed findings improve verification over a Markdown handoff | Markdown report → typed findings | `verifier_precision` ↑ | Finding ground truth and scorer; verification Stage |
| BC2 | Deterministic reconciliation beats LLM-only deduplication | LLM dedup Workflow → automatic correlation (FND-R174–178) | `reconciliation_error_rate` ↓ | Finding ground truth; the control Workflow. Correlation only marks findings (FND-R178) |
| BD1 | Run metadata labels are semantically inert | No labels → full eval labels | `model_visible_request_digest_mismatch_rate` = 0 | V20-006 already proves isolation end to end; only the request-digest probe is left |
| BD3 | Complete-case telemetry analysis biases estimates | Complete reports only → all attempts with missingness strata | `complete_case_quality_bias` ≠ 0 | Offline analysis of existing eval usage |
| BE1 | Worker capacity buys more quality than Planner capacity | Cheap/strong Planner × Worker policies at matched cost | `severity_weighted_recall_at_fixed_cost` ↑ | Finding ground truth; matched-cost policy pairs |
| BE2 | A Stage-local strong model approaches all-strong quality | All small / all strong → strong at one Stage | `quality_per_model_cost` ↑ | As BE1; one preregistered Stage |
| BF3 | Workspace and parser coverage explain different false negatives | One coverage family → both | `fn_cause_prediction_log_loss` ↓ | Finding ground truth joined with `lean@1` and code-analysis coverage |
| BG1 | Workflow Stage granularity has an interior optimum | One-Stage and maximal split → 2-, 3- and 4-Stage variants | `end_to_end_f1_per_cost` max | Workflow variants; finding ground truth |
| BG2 | Serial passthrough Stages can beat one Router | `router@1` → serial `passthrough@1` Stages | `planner_tokens_per_completed_case` ↓ | Two equivalent Workflows |
| BG3 | Explicit Memory notes beat tacit shared-session handoff | `shared` without notes → `isolated` with a Memory note handoff | `downstream_output_variance` ↓ | Handoff instructions |

## Blocked by a specification non-goal (5)

All five need observation profiles that can be switched per arm. The Worker
specification has only `lean@1` and lists user-defined observation reducers and
YAML A/B profiles as a non-goal. They become runnable only if that non-goal is
reopened. Legacy results AC3, AC5 and AC6 below are the existing evidence for
reopening it.

| ID | Hypothesis | Arms: control → treatment |
| --- | --- | --- |
| BA3 | Pull-based Planner state beats eager observation injection | Eager detailed projection → lean result plus pull tools |
| BA4 | Completeness flags prevent a premature Planner finish | Counts without flags → counts with completeness and truncation flags |
| BF1 | Lean observations improve the next Planner dispatch | Worker result only → result plus `lean@1` |
| BF2 | Adaptive observation detail beats a fixed projection | Result-only and always-detailed → lean with pull on incomplete |
| BF4 | Path detail can bias breadth on large repositories | Counts and flags → plus first-observed paths |

## Need a new capability (8)

| ID | Hypothesis | Missing piece |
| --- | --- | --- |
| AZ3 | Bounded State carry keeps the useful part of session continuity | `isolated` always carries eligible State (WRK-R73); the stateless control arm needs a switch |
| AZ4 | Isolated sessions pay a Skill reload tax | Trusted re-disclosure of an already pinned Skill |
| BA1 | Explicit Planner state bounds prompt growth | A stateflow Planner; no design remains |
| BA2 | Explicit current state reduces duplicate dispatches | Same as BA1 |
| BC3 | Exact evidence references beat copied excerpts | An excerpt-only finding variant for the control arm |
| BC4 | Handle and preview beat inline large results | `http_request` already returns a preview and request ID; the inline control arm does not exist |
| BD2 | Infrastructure equivalence predicts semantic equivalence | Effective served-model and tool-capability fingerprints; profile digests identify configuration, not provider weights (TPL-R113) |
| BE3 | Evidence-aware escalation beats same-policy retry | Per-tool semantic evidence outcomes; escalation is chosen by fixed Workflow outcome branches |

## Recorded legacy results

24 of the 228 legacy hypotheses have a recorded result; the other 204 are
proposals only. The catalog keeps one sentence per result and no numbers, and
the source report (`contractor-old`, `docs/research/reports/research.html` at
`9c76b56c`) is not in this repository. These results describe the legacy
system and prove nothing about v2; the rows below bear on v2 hypotheses.

| Legacy ID | Legacy result | Bears on |
| --- | --- | --- |
| AC3, AC5, AC6 | Supported or shipped: deterministic observations raised recall without adding run-to-run variance, coverage-gap observations paid off, and lean versus lean-plus-paths moved precision and recall in opposite directions | The observation-profile non-goal that blocks BA3, BA4, BF1, BF2 and BF4 |
| AD6, M1 | Contradicted and negative so far (pass@1): a staged pipeline lost to a flat one because findings were lost at handoffs | BG1, BG2, BC1: measure per-Stage finding retention before comparing Stage layouts |
| I1 | Partially supported: graph tools helped decisively on large fixtures and were near-neutral, at extra cost, on small ones | BF3, BF4: stratify by repository size |
| F3, G1, G2 | Negative: a held-out fixture did not confirm the overfitting explanation; no under-covered language or vulnerability class matched the premise | Do not repeat as stated |
| PV2 | Design warning: the arm changed several mechanisms at once | Every arm pair (see *How to run*) |

The other recorded results (A2, AC2, AE1, AE4, AF1, D3, F1, F2, F4, K1, PV5,
X1, X2) concern legacy components and were not mapped to v2.

## Legacy ideas worth porting

Checked on 2026-10-23 by searching `docs/spec`, `configs/` and the Runtime
toolset list, not by reading the implementation.

### Deterministic tools instead of model reasoning

| Legacy ID | Idea | State on 2026-10-23 |
| --- | --- | --- |
| J1, J3 | Match dependency manifests against known CVEs (osv): true positives at almost no model cost | No dependency or CVE lookup anywhere; `scan@1` exports nuclei, sqlmap, naabu and ffuf (SCN-R52). Fits a Tool Worker without a model |
| H4 | Dominator check: a guard counts only if it dominates the sink on every path | The finding-proposal example in the [findings](../spec/findings/findings.md) specification makes this a `static-trace` check for the model |
| N4 | Git provenance tools (`blame`, `log`, `show`) for N-day scoping and churn-based prioritisation | No git tool among the Runtime toolsets |
| N1 | Variant hunting: from a confirmed finding and its fix pattern, find unpatched siblings in the same repository | Not present; correlation (FND-R174–178) only links findings that were already reported |
| Q1–Q3 | The model writes a semgrep rule from a confirmed finding; the rule then runs without a model and accumulates in a library | Not present; the largest of these ideas |

### Agent-loop control

| Legacy ID | Idea | State on 2026-10-23 |
| --- | --- | --- |
| C6 | Accept a negative exploitability verdict only after N distinct failed bypass attempts | Not present |
| AW2 | Fail closed on an ambiguous empty tool result, with one bounded retry or alternate query | The same gap as `tool_evidence_outcomes` (BE3) |
| AG1 | Cap characters per line when reading files, so one minified or lockfile line cannot fill a read | `read_file` pages by `max_lines=200` (LIM-WSP-12) under the output limit; the specification has no per-line cap |
| AL6 | Record refuted paths in Memory, not only explored ones, so the model does not return to them | Not present in the [Memory](../spec/tools/memory.md) specification |
| PV8 | Re-run an incomplete subtask once in place before decomposing it | Not checked against the Planner retry and escalation policy |
| AM3 | Stop after K iterations without a new finding instead of after a fixed count | Not checked against the Audit Round stop rules |

### Already in v2 or not worth porting

Already present in some form: cached-input token usage
(`gen_ai.usage.cache_read.input_tokens`, A1), deterministic Planners and scan
planning (`passthrough@1`, W2), typed findings and correlation (B2, K1), HTTP
request IDs as evidence (C1, close to BC3) and per-role model escalation (AI,
close to BE). SARIF is not: it is a non-goal of `code-analysis@1`.

Not worth porting now: retrieval and experience memory (AJ, O), automatic
prompt refinement in the style of DSPy or GEPA (AK), adversarial self-play on
synthetic vulnerabilities (P) and concolic fuzzing (T) are research projects of
their own. Most remaining ideas are tied to legacy fixtures (XBOW, the
vuln-detection set) and components.

## Directions

AZ: Worker session isolation and explicit State carry · BA: bounded Planner
state and selective projections · BB: terminal Worker summarization policy ·
BC: typed artifact and evidence authority · BD: evaluation non-interference and
infrastructure confounding · BE: per-role model allocation and evidence-aware
escalation · BF: observation composition and coverage signals · BG: Stage
granularity and explicit handoff channels.

## Not rechecked

Whether reports expose per-invocation Worker tokens, a seed or a served-model
fingerprint outside the specification; whether findings are retained per Stage.
Legacy ideas not listed above were read by claim only.
