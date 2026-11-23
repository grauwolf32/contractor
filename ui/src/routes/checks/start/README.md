# Start a check (`/checks/new`)

The Start page replaces the New Audit dialog and the Start Audit time-limit
dialog ([build contract](../../../../../docs/design/ui/v3b-build-contract.md)
§5; mockup `docs/design/ui/mockups/v3b/start.html`). `StartCheckRoute`
(`../start.tsx`) is the route component.

## URL

`/checks/new?project=<projectId>&objective=<text>&type=<check type name>`

- Without `project` the page lists projects (`useProjectsIndex`) and keeps
  the other parameters when one is picked. An invalid or unknown project
  shows the list again with a notice. "Change project" carries the
  objective typed so far and the `type` parameter.
- `objective` prefills the objective. A new `project` or `objective` in the
  URL starts a fresh form.
- `type` selects a check type (an AuditProfile name, any scope variant). It
  is the list selection: rows link to it, J / K replace it. Without it the
  page selects the suggestion for the opening objective, else the first
  ready check type, and narrow screens initially show the chooser. A name that is not
  among the check types read so far gets a notice (while more pages exist,
  it suggests loading more).

## Check types and readiness (`check-types.ts`)

- Versions of one name form a family; the newest published version is selected.
  Advanced options can pick another loaded version. Each published program stays
  an exact selectable choice; the UI does not reconstruct retired inventory variants.
- Known shipped names have concise labels and distinct shared glyphs. Unknown
  programs keep a neutral presentation and their published input contract.
- Readiness compares required input formats with current materials, using a maximum
  matching so one JSON is not counted as two inputs. A format match does not
  establish content validity. Web scan additionally requires its ordinary target.
- Catalog/material reads remain paged. Failed refresh/continuation retains the
  loaded choices and an explicit Retry; a partial inventory never implies a unique
  automatic match. The Library link exposes every published check type.

## Suggestions (`suggestions.ts`)

Static rules in order: words in the objective and a ready check type of the
rule. The first rule that applies is the suggestion, the next one with a
different check type the alternative. Every reason is a fixed sentence that
names the words and what the check type does, followed by "Your project has
a material whose format matches each input it needs."

## Materials

A slot with exactly one format match in a fully read inventory is attached
automatically, unless that material is also the only match of another
input (then both wait for a choice); several matches need a choice;
explicit choices (including "Don't use" on an optional input) last while
the input contract stays the same. The page reads at most four pages of
materials on its own and offers "Load more materials"; a partial inventory
is never treated as unique. Completeness comes from the pages read, so a
failed refresh keeps a complete inventory and its attachments ("Retry
loading materials" reads every page again; after a failed further page it
reads that page). A material chosen for several inputs gets a warning.

## Starting (`launch.ts`)

- **Start check**: `POST /v1/projects/{projectId}/audits` (Idempotency-Key
  `create-audit-ui-…`, the same while the request is the same), then
  `POST /v1/audits/{auditId}/start` (If-Match the draft's revision,
  `deadlineSeconds`, Idempotency-Key `start-audit-ui-…`), then the check
  page. A draft this page created is reused for the same request when the
  start fails.
- **Save as draft**: the create request only, then the draft's page.
- A lost answer (status 0) shows "The check may already exist" / "The check
  may already have started" with "Retry same request" (same key). A changed
  form uses a new key and says so. After a lost start answer a changed form
  would make a second check while the first may be running, so Start check
  waits until the first request is retried (or the earlier values are back).
- Any other failed start reads the check again (`GET /v1/audits/{auditId}`)
  and keeps that copy:
  - no longer a draft: "This check has already started" with "Open the
    check"; Start check and Save as draft stay off for that request;
  - still a draft after a refusal (4xx): "saved as a draft but did not
    start" with the Server's error, and on 412 (or a newer revision) "The
    check changed on the server…"; the next Start check uses the revision
    read;
  - still a draft after a server or gateway failure (5xx), or a read that
    failed: "may already have started" with "Retry same request". Such
    answers never say "did not start".
- Create refusals show `ErrorNotice`.
- After every answer: project check lists, the check and the cross-project
  lists are invalidated (`invalidateCrossProject`).

## Time limit (`time-limit.ts`)

24 hours (default), 7 days, No time limit (`0`) or Custom hours from 0.01 to 8760. Drafts do not need one.

## Keyboard

J / K move through check types. Ctrl+Enter / ⌘+Enter (`useShortcuts`
"mod+enter", declared on the Start check button) starts the check from any
field of the setup, or with focus anywhere else but on a link, when nothing
keeps it from starting. The type chooser excludes this shortcut so search does
not launch a check. On a link the binding declines the key (its
`when`), so the browser opens the link in a new tab as usual.

## Approved OpenDesign transfer (UI-R227–230)

The objective precedes a three-column workspace. At medium widths summary/actions
follow setup, and phones use normal document flow with a collapsible type chooser.
The existing list and launch owners still handle pagination, partial reads,
compatibility, exact versions, request keys and context changes.

`assertion-material-editor.tsx` starts with one empty common assertion row and
keeps stable local keys through Add/remove. Shared rationale is required;
conditions/evidence/version are available on demand. Explicit Save and attach
uses `authored-material.ts` and the existing exact Project Artifact CAS path.
Failures keep the draft; editing alone performs no write or launch.

Scope is descriptive, with no extra content reads. It appears only for known API
programs, Web scan or explicitly typed engagement inputs. Source-only checks have
no separate Scope. Task limits are not endpoint counts. Exact selected material
links open the existing reader; preparation establishes the accepted worklist.
Concise known-profile labels and shared glyphs change no execution capability.
