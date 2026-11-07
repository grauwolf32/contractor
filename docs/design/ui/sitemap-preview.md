# Sitemap material preview

The production reader presents the existing exact `sitemap/map` material using
the public `SitemapDocument` contract. The accepted behavior is
[UI-R236](../../spec/ui/ui.md#ui-r236); Artifact reads and the domain reducer keep
their existing owners. This record describes the visual composition, not a new
sitemap API or editor.

OpenDesign supplied the connected navigation rail, selected-route treatment,
compact overview and distinct bodies for observations, declarations and notes.
Production uses shared controls, Facts, formatting, typography and theme tokens.
Navigation keeps the complete saved route, with one display-folder level to
bound deep paths. It does not change canonical groups or normalize identifiers.
Status colors follow actual saved values, and POST uses the same color family
as the existing OpenAPI reader; the static reference's positional color rule was
removed. Semantic heading/Facts selectors, padding, scrolling, pagination and
phone disclosure behavior were adapted to the production components.

## Preserved native design

The independent native project is `contractor-sitemap-design-20261107`,
conversation `ff008a35-e15a-4c65-8226-aef01c10c7eb`. The original CSS, rationale,
synthetic inputs, host preview and provenance are retained outside the checkout:

`~/Documents/ChatGPT/AppSec/designs/contractor-sitemap-preview-2026-11-07/`.

At the user's follow-up request, the common archive also contains those originals
under `docs/design/ui/experiments/sitemap-preview-opendesign-2026-11-07/`:
`~/Documents/ChatGPT/AppSec/archives/contractor-ui-experiments-2026-11-06/experiments.tar.gz`.
Its updated manifest verifies 529 files: all 509 previous member payloads are
unchanged and 20 sitemap files were added. The exact earlier archive, manifest,
README and verification record remain in `revisions/2026-11-06-original/` beside
the updated archive. Future design additions belong in this common archive.

The sandbox's complete source trial remains at
`~/src/third-party/open-design-sandbox/contractor-sitemap-trial/20261107/`.
Its `provenance.json`, `preservation.json`, `pre-backup.json` and
`post-backup.json` record native output hashes and recoverable state archives.
Only synthetic references and the current theme were supplied; the existing
configured Contractor model was retained. The original generation wrote both
files and then hit the enforced 24-request cap. A completion-only continuation
finished successfully, with 26 requests total under a 28-request cap and the
native child exited. Both outcomes are retained, rather than calling the first
attempt successful. The temporary relay was stopped and its upstream credential
file removed.

All 242 pre-existing project files, 13 Project rows, 18 conversations, 75 messages
and the prior OpenCode sessions/messages/parts were compared with the verified
post-run backup and are unchanged. The retained 2026-11-06 original archive revision
has SHA-256
`d3d80bd75e5cc45cf648e64fba6deb8a206943b794529003f3b390e9473877bd`.

## Review and tests

The interactive host imports the actual production `LoadedArtifactPreview` and
synthetic [sitemap fixture](../../../ui/src/test/fixtures/sitemap-preview.json).
It offers the three themes, phone/desktop widths and normal, empty, declared-only
and unsupported examples. The native version is a static composition reference;
the production version owns actual filtering, selection and paging.

To restore the local review host without restoring the old experiment archive:

```sh
mkdir -p ui/test-results/sitemap-preview
cp ~/Documents/ChatGPT/AppSec/designs/contractor-sitemap-preview-2026-11-07/host-preview/* ui/test-results/sitemap-preview/
cd ui && corepack pnpm dev --host 127.0.0.1 --port 17560 --strictPort
```

Open `/test-results/sitemap-preview/index.html?theme=dark&width=1440` on that
server. The host contains no demo credentials or target calls.

`ui/e2e/sitemap-preview.spec.ts` exercises the ordinary exact-revision Material
route, rather than this host. Shared negative document fixtures, the generated
schema, maximum preview bytes, 1,200 routes, maximum-size concrete collections,
three themes and 390px remain executable in the normal UI gates. The synthetic
base fixture was also accepted by the existing Go sitemap domain decoder.
