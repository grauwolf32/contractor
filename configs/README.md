# Contractor configuration

This directory contains the executable default configuration. It includes the
small artifact-copy fixture and the four-Stage `openapi-from-source@1` and
`likec4-from-source@1` project workflows. Validate the complete set from the
repository root with:

```sh
go run ./cmd/contractor-server config validate --root ./configs
```

Manifest identity comes from `kind`, `metadata.name`, and `metadata.version`;
file names and nesting are only organizational. Instruction references are
paths relative to this directory.

`examples/` contains copyable multi-Stage, bounded-retry, single-Worker
`streamline@1`, and multi-Worker `router@1` Workflow manifests. They are
intentionally outside `workflows/`, so they document supported shapes without
changing the default end-to-end fixture.
