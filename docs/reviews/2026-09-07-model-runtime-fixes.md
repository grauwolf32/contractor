# Model policy runtime fixes

Implemented on 2026-09-07 after the configuration review. The original three tasks are
registered in `tasks/index.yml`; each records its implementation commit.

## V47-001 — Planner temperature

Streamline and Router now forward the selected policy temperature. An explicit
zero remains zero; an omitted configured temperature is omitted from the wire
request. Legacy factories without ModelAccess keep their deterministic zero.
The regression exercises the shared Planner engine for both profiles and checks
serialized gateway requests at temperatures 0, 0.1 and 1, plus omitted settings.

Implementation: `5696778d`.

## V47-002 — Gateway retryability

The Python adapter projects HTTP status/known error-code metadata into a safe
retryable flag without retaining provider bodies, headers or exception context.
400/401/403/404/413/422 are permanent; 408/409/429 and 5xx are retryable, except
known insufficient-quota/budget/context-limit codes. Transport failures and
timeouts remain retryable. Cancellation still propagates.

Ordinary Worker execution, the result finalizer and terminal summarizer preserve
that flag. Public failure messages remain content-free. This controls Contractor
retries after the adapter reports failure; existing OpenAI SDK transport retries
are unchanged. The Go Planner gateway classifier is outside this Worker task.

Implementation: `b76f1de4`.

## V47-003 — Summarizer context admission (reverted by V47-005)

Implementation `4d81f18c` added a byte-based context estimate, a 4096-token
framing reserve, oldest-history removal and a local `input_context_exceeded`
failure. The user subsequently rejected that implicit behavior. V47-005 removes
all four pieces; the gateway again decides whether the summary request fits.
V47-002 permanent gateway error classification remains in place.

The pre-existing 512 KiB transcript/document projection still bounds data passed
to the summary agent and can omit older groups. It is separate from the removed
context admission. No history-compacting continuation strategy is implemented;
future compactification should be an explicit strategy.

## Verification

- `go test ./internal/planner/streamline ./internal/planner/router` — passed.
- `cd runtime && .venv/bin/python -m pytest tests/test_worker_summarizer.py tests/test_openai_gateway_llm.py tests/test_adk_runtime.py` — 98 passed.
- Ruff checks and formatting checks for the changed Python files — passed.
- `git diff --check` — passed.

The fixes do not change e2e manifests, Audit collector code, production model
budgets or running services. Live quality and throughput were not benchmarked.
