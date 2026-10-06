# Start a check (`/checks/new`)

The Start page replaces the New Audit dialog and the Start Audit time-limit
dialog ([build contract](../../../../../docs/design/ui/v3b-build-contract.md)
§5; mockup `docs/design/ui/mockups/v3b/start.html`). `StartCheckRoute`
(`../start.tsx`) is the route component.

## URL

`/checks/new?project=<projectId>&objective=<text>&type=<check type name>`

- Without `project` the page lists projects (`useProjectsIndex`) and keeps
  the other parameters when one is picked. An invalid or unknown project
  shows the list again with a notice.
- `objective` prefills the objective. A new `project` or `objective` in the
  URL starts a fresh form.
- `type` selects a check type (an AuditProfile name, any scope variant). It
  is the list selection: rows link to it, J / K replace it. Without it the
  page selects the suggestion for the opening objective, else the first
  ready check type, and narrow screens show the list.

## Check types and readiness (`check-types.ts`)

- Versions of one name form a family; the preferred version is the newest
  the Server can run. Advanced options can pick another loaded version.
- Families that differ only in scope (same standard scheme, mode, inventory,
  inputs and interaction rules; for example ASVS review and its pilot, WSTG
  and its fast pass) share a row; the Scope card chooses between them.
- Readiness: the Server can run it, every required input can have a current
  material of its own whose media type matches (a maximum matching, so one
  JSON is not counted as both an API spec and scan settings), and a check
  type whose workflows read the `target` scope field finds a live target in
  the project settings. Groups: Ready with your materials, Needs more
  materials (with what is missing and links to add it), Can't run on this
  server (with the Server's reasons). If the materials cannot be read, the
  types are listed without readiness.
- Media types are all that is compared ("format matches", UUS:80-81).

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
is never treated as unique. A material chosen for several inputs gets a
warning.

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
  form uses a new key and says so. Refusals show `ErrorNotice`.
- After every answer: project check lists, the check and the cross-project
  lists are invalidated (`invalidateCrossProject`).

## Time limit (`time-limit.ts`)

24 hours (default), 7 days, No time limit (`0`) or Custom hours from 0.01 to 8760. Drafts do not need one.

## Keyboard

J / K move through check types; Ctrl+Enter / ⌘+Enter in the objective
starts the check when nothing is missing.
