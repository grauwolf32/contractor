# Inbox OpenDesign review — 2026-11-04

Task: V429-029. Design exploration only; the production Inbox remains unchanged.

The preview renders the **real React Application and Inbox route**, with schema-valid synthetic responses adapted from `ui/src/routes/inbox/inbox.test.tsx`. It uses the shared shell, fonts, theme tokens, status icons, list/detail navigation and decision components. Every API request is intercepted in memory. Mutations return an explicit preview-only rejection; they cannot reach the live Server.

## Review

```sh
export PATH=/Users/ruslan/.cache/contractor-v425-quick-tools/node-v24.20.0-darwin-arm64/bin:$PATH
node docs/design/ui/experiments/inbox-opendesign-2026-11-04/build-fixes.mjs
node docs/design/ui/experiments/inbox-opendesign-2026-11-04/prepare.mjs
# Use the repository's local Vite preview at 127.0.0.1:17560.
node docs/design/ui/experiments/inbox-opendesign-2026-11-04/verify.mjs
```

Open `/test-results/inbox-design/index.html`. The review controls select Current UI, OpenDesign or Compare, six selections, four read states, three themes and four widths. The UI itself preserves queue selection, J/K, previous/next, back-to-list and existing destinations. Destination pages remain synthetic; this exploration verifies the Inbox rather than supplying a whole demo backend.

`current-fixture.tsx` owns synthetic transport/wiring only. `opendesign.css` is the exact native contribution. `layout-polish.css` records host-authored layout refinements separately. `build-fixes.mjs` emits `layout-fixes.css`: it restores shared decision/facts layout after overly broad native selectors and maps the four native overview actions to the real component nesting. This compatibility repair is host-owned, not represented as native output. No application component or global stylesheet is edited; the existing type scale and font families are retained. In the Running example the synthetic active check has no coverage gaps, making its Running section unambiguous; other examples retain two explicit follow-up gaps.

## Native inputs and preservation

`brief.md` and frozen `reference/inbox-reference.html` were copied into a new isolated native project. The DOM reference includes the actual baseline's six selections and existing theme tokens; SVG paths are abbreviated in the reference only. `reference/baseline.json` records actual current text and schema checks. Images in `reference/` and `review/` support visual review.

OpenDesign has no host repository mount or unrestricted network. Its existing read-only, unprivileged container uses the internal-only network and an expiring relay limited to the existing Contractor model, 24 requests and one active request. Previous native projects, files and history are preserved. Complete recoverable state and sandbox backups remain outside Git; `preservation.json` verifies SQLite integrity, original project/Run/input file hashes and historical rows in both the application and OpenCode databases. The shared OpenCode project access timestamp and append-only runtime log change normally; the prior log/store are retained in the pre-run backup and all prior session/message/event content is identical. `provenance.json` records the native outcome, output hash and relay cleanup without credentials.

`verification.json` records actual layout, schema, navigation and network checks. Review captures include desktop/mobile and all themes. UI copy and domain states are source-owned; execution success remains separate from result quality and proposals remain separate from human decisions.
