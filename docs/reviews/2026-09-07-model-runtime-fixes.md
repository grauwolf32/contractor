# Model policy runtime fixes

Implemented on 2026-09-07 after the configuration review. The three tasks are
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

## V47-003 — Summarizer context admission

Before generation, the final ADK request is converted using the same
OpenAI-compatible serializer as the model adapter. It includes system text,
messages and the strict result schema. Admission reserves the policy output
limit plus 4096 tokens for backend framing, and conservatively charges one token
per UTF-8 byte of the complete serialized request.

If necessary, binary search removes the oldest complete transcript groups while
preserving the newest suffix, required task identity, observations and schema.
An irreducible oversized request fails with `input_context_exceeded`,
`retryable=false`, and zero model calls. The pre-existing 512 KiB projection
limit remains a separate memory bound.

For the current 118000-token context and 8192-token output, the serialized
request allowance is 105712 UTF-8 bytes. This is intentionally conservative,
not an exact token count; it can discard more history than a model-aware count.
The byte-level-tokenizer assumption and framing reserve do not establish a
universal guarantee for arbitrary upstream normalization or custom templates.
The provider remains authoritative.

The current route does not declare an exact tokenizer/template identity.
LiteLLM documents that its token counter can fall back to a different tokenizer
when no model-specific tokenizer is available. [LiteLLM token counting](https://docs.litellm.ai/docs/completion/token_usage).
LM Studio exposes exact model tokenization via its SDK and recommends counting
after applying the chat template; that is a separate capability from the current
OpenAI-compatible inference route. [LM Studio tokenization](https://lmstudio.ai/docs/python/tokenization).
No tokenizer dependencies, new network routes or live-service changes were added.

Implementation: `4d81f18c`.

## Verification

- `go test ./internal/planner/streamline ./internal/planner/router` — passed.
- `cd runtime && .venv/bin/python -m pytest tests/test_worker_summarizer.py tests/test_openai_gateway_llm.py tests/test_adk_runtime.py` — 98 passed.
- Ruff checks and formatting checks for the changed Python files — passed.
- `git diff --check` — passed.

The fixes do not change e2e manifests, Audit collector code, production model
budgets or running services. Live quality and throughput were not benchmarked.
