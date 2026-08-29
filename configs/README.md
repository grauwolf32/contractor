# Contractor configuration

This directory contains the executable default configuration for the first
slice. Validate the complete set from the repository root with:

```sh
go run ./cmd/contractor-server config validate --root ./configs
```

Manifest identity comes from `kind`, `metadata.name`, and `metadata.version`;
file names and nesting are only organizational. Instruction references are
paths relative to this directory.

`examples/` contains copyable multi-Stage, bounded-retry, and two-Worker
`streamline@1` Workflow manifests. They are intentionally outside `workflows/`,
so they document supported shapes without changing the default end-to-end
fixture.
