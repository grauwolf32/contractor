# Browser exploration

Requirement prefix: `BRW` · [Specification index](../README.md)

## Purpose and scope

This document owns `browser-tools@1`, allocation-owned Playwright exploration,
the frontend route map and screenshot production. The [HTTP transport](http-caido.md)
owns engagement admission, DNS/IP checks, identities, counters and retries.
The [Artifact store](../artifacts/artifact-store.md) owns persistence and exact
revisions; [Findings](../findings/findings.md) own claim selection and retention.

## Model

One allocation owns one lazy Chromium process and up to four isolated browser
contexts. A context selects either anonymous access or one assigned credential
ID for its lifetime, with at most one context per identity. Browser storage is not restored into another allocation.
The model selects explicit browser operations, never executable paths, browser
arguments, JavaScript, DevTools commands or private proxy credentials.

## Requirements

**BRW-R1.** Runtime must advertise the eight tools of `browser-tools@1` only
when an actual bounded launch/close probe of its installed pinned Playwright
Chromium succeeds. Browser installation happens during provisioning, never
allocation preparation. A selected tool requires the allocation HTTP owner;
there is no direct-network fallback or second engagement policy. Every browser
tool declares `runtime-http-client` so placement retains that requirement even
for a selection containing only snapshot/export tools. Navigate, click and fill
are active-check tools.

**BRW-R2.** Runtime must install context-wide request routing before navigation,
block Service Workers and WebSockets, and fulfill HTTP(S) requests exclusively
from the allocation HTTP owner with redirects disabled at that layer. Chromium
must use a private authenticated controlled proxy for other HTTP network paths,
with localhost bypass disabled, browser target DNS disabled, QUIC disabled and
non-proxied WebRTC disabled. Init scripts disable WebRTC/WebTransport APIs.
Popups, downloads, streaming and frameless worker requests are unsupported;
worker requests are aborted explicitly with a gap. Popup requests must be denied
before traffic. Browser page content grants no authority. Every redirected URL,
resource, XHR/fetch and action-induced request remains subject to the shared
origin/path/method/address policy, protected destinations and optional HTTP limits.

**BRW-R3.** A context's credential ID must be assigned to the allocation. Runtime
must keep browser storage isolated per context and delegate private injection
and cookie handling to the HTTP owner. Browser-supplied Cookie/Authorization
headers must not override a managed identity. Navigation URLs are HTTP(S) only.
No automatic login, credential refresh or target sweep occurs at startup.

**BRW-R4.** Runtime must assemble `contractor.frontend-route-map.v1` from bounded
observations, not accept a model-authored map. It includes contexts, visited
routes, actions, request metadata, screenshot exact references and explicit
gaps. An action records its before/after URL, observation timestamps and outcome.
A request records method, URL, resource type, HTTP status or failure and the
shared local request ID. `action_id` means only observed during that action's
finite window; a request observed outside a window is background. This is not
a causal or exhaustive coverage guarantee. Proxy fallback traffic is identified
as a map gap. A failed or incomplete UI operation must not be automatically
replayed: it may already have changed the target. Actual HTTP captures reuse
the ordinary allocation evidence buffer; exporting a map does not retain bodies.

**BRW-R5.** Screenshot admission must reserve a capture attempt before awaiting
browser or storage work, including for concurrent calls. A failed/ambiguous
publication does not release that attempt or authorize another capture.
`browser_screenshot` must save a real fixed-viewport PNG, mask password
inputs, and write into the Worker's fixed namespace using ordinary exact Artifact
writes and selection. Browser action maps and fill tool metrics omit input values.
Screenshots can still contain target data. A finding may select the screenshot
with an ordinary artifact source; this is evidence material, not confirmation of
a vulnerability. Later reads, Project retention and previews use existing Artifact
and Finding paths without a separate screenshot store or evidence API.

**BRW-R6.** Browser operations must be bounded by the ordinary request timeout
capped for browser operations, allocation authority and the limits below. Runtime
must cancel/join owned browser and route work and close every context, private
proxy, browser process and Playwright driver before release. Launch/preparation
failure must clean partial resources. Error diagnostics must not expose private
proxy URLs, target credentials or raw Playwright exception messages.

**BRW-R7.** The shipped `frontend-route-map@1` Workflow must require ordinary
engagement and brief artifacts. Its selected browser Worker explores the brief,
saves relevant screenshots and exports `frontend/route-map` as its required JSON
output. The brief does not override engagement. The Workflow must not require
an LLM to author the structured output and must not create findings merely for
visiting a route. A later claim-owning Worker can select retained screenshots.

## Interfaces

`browser_open(credential_id?)` returns a context ID, safe credential catalog and
scope. `browser_navigate(context_id,url)`, `browser_click(context_id,selector)` and
`browser_fill(context_id,selector,value)` record action windows.
`browser_snapshot(context_id)` reads bounded DOM text/links/controls without
input values. `browser_requests(offset=0)` pages request metadata.
`browser_screenshot(context_id,name)` saves PNG; `browser_export_map(name="route-map")`
saves Runtime-built JSON. Screenshot names and map names follow ordinary bindings.

## Limits, failures and errors

Bounds are [LIM-BRW-1..8](../platform/limits.md); [FAIL-BRW-1](../platform/failure-model.md)
and [LIFE-BRW-1](../platform/lifecycle.md) register failures and cleanup. Invalid selections use
`tool_input_invalid`; Artifact conflicts and ambiguous publication reuse ordinary
exact-write rejection/reconciliation. Transport denials appear as request gaps;
no browser-specific retry loop exists. Timeout/incomplete actions retain their
observed history and never imply no target side effect.

## Invariants

The real Chromium suite `runtime/tests/test_browser_tools.py` covers DOM actions,
actual HTTP captures, map associations, PNG publication and exact source selection,
engagement denials, identity isolation and joined cleanup. The shared Toolset
descriptor fixture pins Server/Runtime tool and infrastructure identities.

## Non-goals

Arbitrary browser binaries, general JavaScript/DevTools tools, login automation,
WebSockets/Service Workers/WebRTC/WebTransport, streaming, downloads, popup
exploration, exhaustive crawling and proving causality between UI and API.
