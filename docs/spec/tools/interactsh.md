# Interactsh tools

Requirement prefix: `INT` · [Specification index](../README.md)

## Purpose and scope

This document owns `interactsh@1`, the private `interactsh-http@1` adapter,
allocation-owned callback registrations and their ordinary Artifact evidence.
RuntimeConfig layering and encrypted credentials belong to
[Runtime configuration](../configuration/runtime-config.md); tool authority
belongs to [Toolsets](toolsets.md).

## Requirements

**INT-R1.** AgentTemplate selects the exact `interactsh_register` and/or
`interactsh_poll` tools. Both require the `interactsh-client` private channel
from `interactsh-http@1`. Missing configuration or advertised capability fails
placement; prepare repeats the check. Labels and Skills grant no tools.

**INT-R2.** The operator supplies one atomic `worker.interactsh` block:
`adapter: interactsh-http@1`, `endpoint`, optional `credential`, `caBundlePem`
and `requestTimeoutSeconds`. Endpoints use the existing Runtime endpoint
contract and a DNS hostname whose served callback domain is that hostname;
IP-literal endpoints are rejected; the DNS hostname is bounded by LIM-INT-9. A fixed path prefix is supported. Credentials
are `interactsh-token@1`, canonical `{"token":"..."}`, with the Runtime
credential token bound (LIM-RCF-24). The private settings contain `token`, never
an inline token in RuntimeConfig. Omission inherits, null clears, and same-layer
unequal complete blocks conflict. Operations authors and displays the safe
configuration and creates write-only credentials.

**INT-R3.** Only the adapter constructs the HTTP client, with system TLS trust
plus the optional validated CA bundle, `trust_env=false`, no redirects, no
proxy or alternate collector fallback. Authorization carries the exact token,
without a Bearer prefix. Generic target tools deny the configured control
origin as infrastructure. The adapter never sends model-supplied URLs.

**INT-R4.** Each registration owns a fresh RSA-2048 key, unpredictable 20-character
correlation ID, 13-character nonce, independent 256-bit registration secret and
opaque safe handle. Only the callback payload, correlation ID, handle, state
and remaining local lifetime are model-readable. Session secrets and private
keys never enter telemetry, artifacts, ADK State or model context.

**INT-R5.** `interactsh_register(action)` records a bounded, nonblank intended
trigger description or request ID. This is an unverified model-supplied relation,
not proof that a target caused the callback. `interactsh_poll(handle)` accepts
only a handle created by this active allocation and performs one finite poll.
Unknown/released and expired handles are distinct permanent errors.

**INT-R6.** The supported protocol is ProjectDiscovery Interactsh **v1.4.1**
(commit `a5b78b2f1d990cd18c85cc422f0e651cf204c464`), with `/register`, `/poll`
and `/deregister`, RSA-OAEP/SHA256 key exchange and AES-CTR event encryption.
The test server uses this pinned upstream implementation. Unsupported or
malformed protocol data fails closed; no alternate encryption reader is used.
Key generation, JSON parsing and decryption run off the event loop. A protocol
upgrade requires running the real-server contract suite before changing the pin.

**INT-R7.** Poll responses have bounded bytes and at most the configured event
prefix is decrypted. Only exact correlation-plus-nonce matches with supported
DNS, HTTP(S) or SMTP(S), a timezone-aware observation time and source IP are
accepted. Root-domain and token-wide unencrypted events are discarded. Excess,
unowned or oversized events are explicitly counted as discarded/truncated;
malformed encrypted content is a classified response failure. No event is
silently attributed to a different registration.

**INT-R8.** Before returning event previews, publish one immutable ordinary
Artifact containing the exact decrypted event bytes (base64), correlation ID,
payload, unverified trigger relation, discarded count and coverage-gap flag.
Media type is `application/vnd.contractor.interactsh-evidence+json`; binding
prefix is `interactsh.evidence.`. Returned entries include protocol, observation
time, source IP, exact artifact ref and zero-based event index. Previews never
include raw callback request/response text. Retained raw content is untrusted.
Deployment secrets reflected into callback content cause rejection, not storage.

**INT-R9.** One pending bounded batch per registration survives an unconfirmed
Artifact write. Reconcile through that same Artifact binding and exact payload;
never append another binding after a lost response. A subsequent poll publishes
the pending batch before contacting the collector again. There is no new
receipt table, secret store or durable callback controller.

**INT-R10.** A successful empty poll reports `no_interaction`; it does not prove
a vulnerability absent. Infrastructure errors, registration uncertainty,
expiration and truncation remain distinct. Every result's attribution is
unverified. Generic HTTP/Caido evidence is not upgraded to machine proof.

**INT-R11.** Allocation teardown stops and joins active tool and adapter calls
before cleanup, attempts deregistration with the owned identity under a separate
finite cleanup budget, and dereferences all keys, secrets, pending events and
clients. Remote deregistration failures are safe adapter metrics and do not
prevent local erasure; operator-owned upstream TTL bounds remote retention.
Published evidence follows ordinary Run/Project retention and purge. Ephemeral
registrations are never restored from artifacts or reused by another allocation.

## Limits, failures and errors

- LIM-INT-1: at most 16 registrations over an allocation's lifetime.
- LIM-INT-2: at most 100 accepted/decrypted event candidates per poll.
- LIM-INT-3: at most 1 MiB decoded HTTP response and 64 KiB per decrypted event;
  exact serialized evidence is at most 2 MiB. At most one batch per registration
  is pending, so retained pending buffers are bounded by 48 MiB in total.
- LIM-INT-4: local registration lifetime is 3600 seconds; no background renewal.
- LIM-INT-5: at least two seconds between network polls for one handle.
- LIM-INT-6: registration and poll independently use 1..120 seconds, defaulting
  to the allocation request timeout capped at 120; evidence publication has its
  own equal budget. Cleanup has one independent total budget of 10 seconds.
- LIM-INT-7: intended action is 1..1024 UTF-8 bytes.
- LIM-INT-9: callback domain at most 219 ASCII bytes.
- LIM-INT-8: public poll results contain at most 100 metadata entries and are
  capped at 64 KiB serialized JSON.

**FAIL-INT-1.** Registration sends one identity once. A lost, malformed or
retryable response is reconciled by an authenticated poll of that same
identity, retaining any resulting events. If reconciliation cannot confirm it,
return `outcome_unknown` plus the safe handle. Poll can reconcile it later.
Auth and invalid-request failures are permanent; never blindly register a new
identity. A timeout before identity generation is a retryable request failure.
Cancellation retains remote ownership for allocation cleanup and joins work.

**FAIL-INT-2.** Poll drains upstream data, so transport, malformed-response and
timeout failures set a sticky coverage gap and are never automatically retried.
A later successful empty poll cannot clear that gap. Explicit subsequent polls
are rate bounded. HTTP 429 supplies a bounded Retry-After hint; other retryable
failures carry no provider text. Auth, invalid/expired handles, malformed
protocol and oversized responses are classified separately. Cancellation joins
in-flight work; cleanup uses its own budget. A lost deregistration response
may be reconciled only by the same identity or left to upstream TTL; it never
causes a new session. Failure is recorded without exposing provider bodies.

**FAIL-INT-3.** Evidence publication uses the ordinary binding as its receipt.
Confirmed matching content yields an exact ref. An unavailable or mismatched
receipt returns `interactsh_evidence_unconfirmed`, keeps the pending batch and
never reports evidence saved. Cancellation joins work and release erases only
the ephemeral batch; already published artifacts remain ordinary evidence.

**LIFE-INT-1.** The allocation owns ephemeral registrations, keys and pending
buffers; cancel, lease loss and release stop and join calls, deregister where
possible and erase local private material. Published evidence is deleted only
through ordinary Artifact/Run/Project lifecycle. Cleanup failure is detected by
adapter metrics; an operator must configure upstream retention. Reopen remote
cleanup guarantees if callbacks need to survive allocation replacement.

Stable tool errors: `interactsh_not_configured`, `interactsh_handle_invalid`,
`interactsh_registration_expired`, `interactsh_auth_failed`,
`interactsh_request_invalid`, `interactsh_request_failed`,
`interactsh_rate_limited`, `interactsh_limit_exceeded`,
`interactsh_response_invalid`, `interactsh_response_too_large`,
`interactsh_evidence_unconfirmed`. Only request failures, rate limits and
unconfirmed evidence are retryable, subject to the outcome rules above.

## Invariants

`runtime/tests/test_interactsh.py` checks ownership, encryption, bounds, lost
responses, exact evidence and joined teardown. The real local upstream server
suite is `make test-interactsh-protocol`. Shared Toolset descriptors and
RuntimeSettings fixtures enforce Go/Python/schema agreement.
