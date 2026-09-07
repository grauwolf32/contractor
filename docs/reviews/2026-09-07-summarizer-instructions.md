# Configurable terminal summarizer instructions

V47-004 adds an optional `spec.summarizer.instructions.ref` to AgentTemplate.
It uses the existing instruction-file loader, so the exact ref, SHA-256 digest
and text are resolved into the immutable template/Run snapshot. The public
configuration resource publishes ref and digest; Runtime receives and verifies
the text. Go and Python share a golden fixture for the template digest.

The working catalog selects `instructions/terminal-summarizer.md` from all
30 ordinary Worker templates, including Memory variants. Audit templates remain
without a terminal summarizer. The text preserves the previous instruction.
Legacy templates that omit the new field retain the existing built-in default,
which also preserves old template digests and e2e manifests.

Configuration and resolved-wire validation reject blank text and text longer
than 8000 Unicode characters, counting whitespace and newlines. Roughly 2000
tokens is only an estimate: no exact tokenizer is configured, and this change
does not introduce a 2000-token admission rule. ADK receives a literal instruction
provider so braces in Markdown or JSON are not expanded as session variables.

V47-005 separately reverts all V47-003 byte-based context admission. Gateway
context errors retain V47-002 retryability classification. The original 512 KiB
projection remains, including its existing truncation behavior. A future
compactification strategy is not implemented here.

Verification:

- Config, contracts and control-plane Go package tests passed.
- Python summarizer, gateway, runtime and contract tests passed, including literal
  instructions, exact/oversized Unicode limits, digest tampering and a large
  summary request delegated to the gateway without local context trimming.
- Both ordinary and e2e configuration catalogs validated.
- All 57 configs/e2e files remained byte-identical.
