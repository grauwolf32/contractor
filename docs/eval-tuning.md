# Eval Tuning Configs

A design for parameter-sweep configurations runnable through the eval suite
(`tests/eval/`), so tuning decisions (sampling, tool params, budgets) are backed
by precision/recall/verdict numbers instead of guesses.

Pairs with [tuning.md](tuning.md) (the full knob inventory). This doc is narrower:
**which knobs are worth sweeping in evals, where to inject them, and the concrete
experiments to run** — prioritizing sampling (temperature/top_p) and tool params.

---

## 1. What the eval harness already gives us

| Capability | Where | Note |
|------------|-------|------|
| Model override | `conftest.py:248` `eval_model` fixture | `CONTRACTOR_EVAL_MODEL` → `LiteLlm(model=..., timeout=600)`. **Single chokepoint — every eval agent gets this object.** |
| Prompt-version override | trace/exploit/vuln tests | `CONTRACTOR_EVAL_*_PROMPT_VERSION`. |
| Toolset toggle | `trace_harness.py:231`, `vuln_scan_harness.py:60` | `with_graph_tools: bool` (graph/trailmark tools on/off). |
| Pass@N | `test_vuln_detection_eval.py:160-207` | `CONTRACTOR_EVAL_VULN_PASS_AT` (default 3); passes if any attempt clears thresholds. |
| Score thresholds | vuln test | `CONTRACTOR_EVAL_VULN_MIN_RECALL` / `_MIN_PRECISION`. |
| N-run aggregation | `scripts/probe_variance.py` | mean/stdev/min/max per metric across samples. |

**Scoring is deterministic** (set-matching precision/recall/f1; verdict equality) —
no LLM judge — so any sweep produces directly comparable numbers. Variance comes
only from the model's own non-determinism, which is exactly what the sampling sweep
should quantify.

**Gaps:** sampling params are unset (every agent runs at backend-default temperature),
`max_tokens` is hardcoded `80_000` in all three harnesses, and tool caps (graph result
limits, body-preview chars, fs listing breadth) are baked into factories with no eval hook.

---

## 2. The eval-config surface (prioritized)

Each config is a bundle of overrides. Group by tuning ROI:

### Tier 1 — Sampling (highest ROI, smallest plumbing)

| Knob | Proposed env var | Sweep values | Hypothesis |
|------|------------------|--------------|------------|
| `temperature` | `CONTRACTOR_EVAL_TEMPERATURE` | `0.0, 0.2, 0.4, 0.7` | Low temp ↑ schema adherence on structured tasks (OAS/trace/verdict) → fewer `malformed`/retries; high temp ↑ recall on open-ended vuln scan. |
| `top_p` | `CONTRACTOR_EVAL_TOP_P` | `1.0, 0.9, 0.8` | Nucleus trim stabilizes output without flattening exploration as hard as temperature. |
| (optional) `reasoning_effort` / thinking budget | `CONTRACTOR_EVAL_REASONING` | backend-specific | Trade latency for depth on vuln_scan / exploitability. |

**Injection — one place.** Extend `eval_model` (`conftest.py:240-253`) so the
`LiteLlm` carries sampling kwargs. ADK's `LiteLlm` forwards extra kwargs to
`litellm.completion`:

```python
@pytest.fixture(scope="session")
def eval_model() -> LiteLlm:
    name = os.environ.get("CONTRACTOR_EVAL_MODEL")
    extra = {}
    if (t := os.environ.get("CONTRACTOR_EVAL_TEMPERATURE")) is not None:
        extra["temperature"] = float(t)
    if (p := os.environ.get("CONTRACTOR_EVAL_TOP_P")) is not None:
        extra["top_p"] = float(p)
    if name:
        return LiteLlm(model=name, timeout=600, **extra)
    if not extra:
        return DEFAULT_MODEL
    return LiteLlm(model=DEFAULT_MODEL.model, timeout=600, **extra)  # clone w/ sampling
```

Because every harness pulls its model from this fixture, this single change makes
sampling sweepable across **all** eval types. (Verify `LiteLlm` forwards the kwargs
in the installed ADK version before trusting the numbers.)

### Tier 2 — Tool params (prioritized per request)

These shape what the agent can *see* and how much it costs. Sweep targets, by the
eval that exercises them:

| Knob | Source (prod) | Proposed env var | Sweep values | Exercised by | Hypothesis |
|------|---------------|------------------|--------------|--------------|------------|
| `with_graph_tools` | `trace_harness.py:231` | `CONTRACTOR_EVAL_GRAPH_TOOLS` | `0 / 1` | trace, vuln (trace mode) | Graph/trailmark tools ↑ cross-file recall on large fixtures (confirms the v7+graph memory result). |
| graph `DEFAULT_MAX_RESULTS` | `tools/code/graph.py:44` | `CONTRACTOR_EVAL_GRAPH_MAX_RESULTS` | `100, 200, 400` | trace, vuln | More symbol hits ↑ recall but ↑ context/token cost. |
| graph `DEFAULT_MAX_PATHS` / `_MAX_PATH_DEPTH` | `graph.py:45,47` | `CONTRACTOR_EVAL_GRAPH_MAX_PATHS` / `_DEPTH` | paths `15/25/40`, depth `20/30` | trace | Deeper call-path enumeration ↑ taint-flow recall; risk of blow-up/noise. |
| HTTP `body_preview_chars` | `tools/http.py` (512 in exploit agents) | `CONTRACTOR_EVAL_HTTP_BODY_PREVIEW` | `256, 512, 2048` | exploitability, web | Bigger preview ↑ evidence quality for verdicts vs. token burn. |
| fs `max_items` / `max_output` | `tools/fs/read_tools.py:58-59` | `CONTRACTOR_EVAL_FS_MAX_ITEMS` / `_OUTPUT` | items `100/250`, output `80k/160k` | all code evals | Wider listings ↑ discovery on big repos vs. context cost. |

**Injection.** Two patterns:
1. **Already-a-param** (`with_graph_tools`): thread an env read into each test's call,
   mirroring `_resolve_prompt_version`. Trivial.
2. **Baked-into-factory** (graph limits, body-preview, fs caps): cleanest is to make the
   tool/graph constructors read an optional override (default = today's constant), then
   set it in a `conftest` autouse fixture from the env var. Keep the production default
   unchanged so non-eval runs are untouched. This is the bulk of the Tier-2 work.

### Tier 3 — Agentic budgets (already partly wired)

| Knob | Source | Proposed env var | Sweep values | Note |
|------|--------|------------------|--------------|------|
| `max_tokens` (summarization trigger) | hardcoded `80_000` in 3 harnesses | `CONTRACTOR_EVAL_MAX_TOKENS` | `60k, 80k, 100k, 120k` | Replace the literals; bigger budget ↑ cross-file reasoning on large fixtures, ↑ cost. |
| `elide_keep_last_n` | `worker_factory.py:56` (15) | `CONTRACTOR_EVAL_ELIDE_KEEP` | `8, 15, 25` | Recall of earlier tool output vs. token cost. |
| planner `max_steps` | per-pipeline | `CONTRACTOR_EVAL_MAX_STEPS` | `15, 30, 75` | Only relevant for task-runner evals (oas/vuln_assess), not single-agent harnesses. |
| `pass@N` | exists (vuln) | `CONTRACTOR_EVAL_VULN_PASS_AT` | `1, 3, 5` | Already wired; extend the pattern to trace/exploit if you want pass@N there. |

---

## 3. Named configs to run (the experiment matrix)

Each row is a config worth committing as a reproducible experiment. Run with
`CONTRACTOR_RUN_EVAL=1` and the listed env vars; aggregate ≥3 samples with
`probe_variance.py` (or pass@N for vuln) since sampling adds variance.

| Config name | Overrides | Target fixtures | Metric to watch | Question it answers |
|-------------|-----------|-----------------|-----------------|---------------------|
| `baseline` | none (backend-default sampling) | all | p/r/f1, success rate, malformed count | Current production behavior + its variance floor. |
| `det-t0` | `TEMPERATURE=0` | trace, oas, exploit | malformed/retry count, verdict accuracy | Does greedy decoding cut format churn without hurting recall? |
| `det-t0-p09` | `TEMPERATURE=0`, `TOP_P=0.9` | trace, oas | f1, variance (stdev across N) | Best stability/quality point for structured output. |
| `explore-t07` | `TEMPERATURE=0.7` | vuln_scan (pass@5) | recall@5, unique findings union | Does higher temp + more attempts ↑ vuln coverage? |
| `graph-on` | `GRAPH_TOOLS=1` | trace, vuln-trace (large: cloud-core, crapi) | recall, annotation count | Re-confirm graph-tools win; quantify token tax. |
| `graph-rich` | `GRAPH_TOOLS=1`, `GRAPH_MAX_PATHS=40`, `GRAPH_MAX_RESULTS=400` | large trace fixtures | recall vs. precision (noise) | Diminishing returns / precision loss from deeper graph. |
| `lean-context` | `MAX_TOKENS=60000`, `ELIDE_KEEP=8` | all | f1 delta vs. baseline, tokens/run | How cheap can we go before quality drops? |
| `rich-context` | `MAX_TOKENS=120000`, `ELIDE_KEEP=25` | large fixtures | recall delta, tokens/run | Is the cost of more context justified on big repos? |
| `evidence-rich` | `HTTP_BODY_PREVIEW=2048` | exploitability, web | verdict accuracy, evidence-present rate | Does more response body ↑ correct verdicts? |

Run sampling × context as a small grid on a fixed fixture subset first (cheap, fast
fixtures like `vampi`, `dsvw`, `vulnyapi`), then promote the winning config to the
expensive large fixtures.

---

## 4. Suggested workflow

1. **Land the plumbing** Tier 1 → Tier 2 → Tier 3 in that order (Tier 1 is ~3 lines
   and unblocks the highest-ROI sweep immediately).
2. **Pin a fixture subset** for fast iteration (small/medium fixtures) via
   `-k` or `CONTRACTOR_EVAL_CASE_IDS`.
3. **Sample ≥3× per config** — sampling sweeps are meaningless single-shot; reuse
   `probe_variance.py` aggregation and report mean ± stdev, not one number.
4. **Compare on deterministic metrics** the harness already emits (precision/recall/f1,
   success rate, verdict accuracy, annotation count, tokens/run).
5. **Promote the winner** to large fixtures, then update the production defaults
   (`worker_factory.py` / pipeline constants / litellm config) — and the
   [Don't switch models] rule still stands: tune sampling/tools/budgets, keep the
   project's default model.

---

## 5. Plumbing checklist (smallest viable change set)

- [ ] `conftest.py` `eval_model`: read `CONTRACTOR_EVAL_TEMPERATURE` / `TOP_P`, pass to `LiteLlm`. **(Tier 1, unblocks sampling everywhere.)**
- [ ] Replace hardcoded `max_tokens=80_000` in `trace_harness.py:266`, `vuln_scan_harness.py:80,97`, `exploitability_harness.py:87,102,117` with an env-driven default. **(Tier 3.)**
- [ ] Env-drive `with_graph_tools` in the trace/vuln tests (mirror `_resolve_prompt_version`). **(Tier 2, trivial.)**
- [ ] Add optional override args to `tools/code/graph.py` constants + HTTP `body_preview_chars` + fs `max_items/max_output`, set from an autouse `conftest` fixture; keep production defaults unchanged. **(Tier 2, the real work.)**
- [ ] Document the new env vars in each test module's header docstring (matches existing convention).

> Guardrails to keep: production defaults must not change from these eval hooks (env-gated,
> default = current constant), and no `assert` in the new override code — use explicit
> `if ...: raise` per project rule.

---

*Injection points verified against `tests/eval/conftest.py`, `trace_harness.py`,
`vuln_scan_harness.py`, `exploitability_harness.py`, and `tests/eval/scorers.py` /
`scoring.py`. Confirm `LiteLlm` kwarg forwarding in the installed ADK before trusting
sampling numbers.*
