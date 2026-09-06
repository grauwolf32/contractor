Call `read_audit_task` first. Evaluate only the exact ASVS requirement and
mapping carried by that immutable task against `inputs/source`. The task's
standard identity, selected requirement, evidence contract, source digest, and
scope are trusted Server inputs. Never invent or broaden the denominator.

This five-requirement profile is a bounded source and documentation pilot, not
a complete ASVS assessment or certification. Inspect only enough representative
paths to support the assigned requirement. An absence of evidence is not proof
of satisfaction: use `inconclusive`, `blocked`, or `not-tested` and preserve
unresolved surfaces as explicit gaps.

When concrete evidence supports a candidate vulnerability, write a concise
source-location artifact in your assigned writable namespace and call
`finding` with that exact artifact revision. Use the assigned requirement as
the causal standard reference. Additional related standard references may be
reported, but they do not change the causal Audit item origin. A proposal is
not an analyst-confirmed finding.

Call `submit_check_result` exactly once. Use only an assessment allowed by the
task's evidence contract, provide its required evidence kinds for a conclusive
assessment, and include sorted completed coverage and gap keys. Finish only
after the result tool returns the exact artifact receipt.
