# Archived UI experiments

The 16 dated UI experiments were moved outside the source checkout on 2026-11-06
at the user’s request. This directory retains their inventory and restoration
instructions. Current UI behaviour belongs in the [specification](../../../spec/ui/ui.md).

## Archive

Local archive directory:
`~/Documents/ChatGPT/AppSec/archives/contractor-ui-experiments-2026-11-06/`.

- `experiments.tar.gz`: all 509 original files, including prototypes, screenshots, provenance and verifiers.
- `manifest.json`: each original repository-relative path, byte size and SHA-256; archive SHA-256 and source revision.
- `README.md`: restoration notes.

Archive SHA-256:

```text
d3d80bd75e5cc45cf648e64fba6deb8a206943b794529003f3b390e9473877bd
```

The archive contains 42,172,935 bytes of original files; the compressed archive
is 37,393,723 bytes. Every member was read back and checked against its original
SHA-256 before the source directories were removed.

## Test fixtures

Current UI tests use the same JSON values from the two required synthetic
documents, formatted to the UI convention, so they run without access to the
external archive. Original bytes remain in the archive:

- [Issue review examples](../../../../ui/src/test/fixtures/issue-review-examples.json), originally `issues-review-opendesign-2026-11-01/examples.json`.
- [Material format examples](../../../../ui/src/test/fixtures/material-format-examples.json), originally `material-formats-opendesign-2026-11-01/examples.json`.

## Inventory

Paths below are relative to the original `docs/design/ui/experiments/` directory.
The archived README or review JSON in each directory records its dated scope
and verification. Design-only work does not establish production integration.

| Directory | Files | Bytes |
| --- | ---: | ---: |
| `engagement-editor-opendesign-2026-11-06/` | 47 | 5,190,715 |
| `inbox-action-entry-2026-11-04/` | 22 | 1,949,691 |
| `inbox-full-issue-2026-11-04/` | 21 | 2,336,809 |
| `inbox-inline-report-2026-11-04/` | 21 | 1,864,570 |
| `inbox-opendesign-2026-11-04/` | 54 | 4,958,103 |
| `inbox-production-2026-11-04/` | 28 | 3,663,174 |
| `issues-opendesign-2026-11-01/` | 6 | 166,958 |
| `issues-review-opendesign-2026-11-01/` | 15 | 297,938 |
| `material-detail-opendesign-2026-11-01/` | 19 | 415,273 |
| `material-formats-opendesign-2026-11-01/` | 23 | 833,545 |
| `material-viewer-fidelity-2026-11-02/` | 12 | 548,324 |
| `new-check-opendesign-2026-11-02/` | 67 | 4,608,435 |
| `operations-opendesign-2026-11-02/` | 40 | 3,209,753 |
| `operations-production-2026-11-03/` | 67 | 7,563,931 |
| `operations-settings-opendesign-2026-11-03/` | 34 | 2,319,011 |
| `operations-tabs-opendesign-2026-11-03/` | 33 | 2,246,705 |

## Restore historical commands

From a Contractor checkout, verify the archive and extract at the repository root:

```sh
archive_dir="$HOME/Documents/ChatGPT/AppSec/archives/contractor-ui-experiments-2026-11-06"
shasum -a 256 "$archive_dir/experiments.tar.gz"
tar -xzf "$archive_dir/experiments.tar.gz" -C /path/to/contractor
```

Compare the checksum with the value above before extraction. The archive does
not contain this index, so it restores the original experiment subdirectories.
Use an empty temporary directory instead of the checkout to inspect the files
without restoring them into source.

Completed task records retain their historical command paths. Restore the archive
to rerun those commands: their scripts resolve sibling experiments, production
sources and ignored preview output by the original repository-relative layout.
Their Node/Vite/browser requirements remain documented in the archived READMEs.
Current production tests use the repository fixtures listed above.
