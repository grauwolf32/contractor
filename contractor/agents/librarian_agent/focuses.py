"""Reusable ``focus`` presets for the general knowledge_discovery /
knowledge_consolidation tasks.

The tasks themselves are domain-agnostic; a workflow injects domain context via
``add_task(params={"focus": <preset>})``. These presets are the security-domain
context, kept here so every security workflow passes the same well-tuned
direction instead of re-inventing it. Pass a different string for a different
domain (architecture, data-flow, API surface, …).
"""

from __future__ import annotations

SECURITY_DISCOVERY_FOCUS: str = (
    "Surface security-relevant facts about this system. Prioritize, in order:\n"
    "  1. Entry points — HTTP routes/handlers, CLI commands, message/queue "
    "consumers, scheduled jobs, webhooks.\n"
    "  2. Authentication, authorization, session and RBAC mechanisms — and "
    "where they are MISSING on a sensitive operation.\n"
    "  3. Trust boundaries — exactly where untrusted input enters and how/if it "
    "is validated.\n"
    "  4. Dangerous sinks — SQL/NoSQL, command exec, deserialization, file/path "
    "operations, SSRF-capable outbound requests, template/eval.\n"
    "  5. External-service integrations — databases, queues, object storage, "
    "identity providers, third-party APIs.\n"
    "  6. Configuration & secret handling — hardcoded credentials, key "
    "material, TLS settings, debug flags.\n"
    "Record concrete, evidence-backed facts (file:symbol/route). A missing "
    "control on a sensitive operation is itself a fact worth recording."
)

SECURITY_CONSOLIDATION_FOCUS: str = (
    "Consolidate the pool into a security knowledge base. Cluster facts by the "
    "concrete asset they concern — endpoint/handler, data flow, trust boundary, "
    "or component — and merge each cluster into one note that states the "
    "control posture (present / weak / absent) with all supporting evidence. "
    "Preserve and flag any disagreement about severity, exploitability, or "
    "whether a control exists; never silently resolve it."
)
