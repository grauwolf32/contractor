# Server helpers

Shared primitives live in named Go packages under this directory. Import the
specific package, for example `internal/helpers/strictjson`; the directory is
not a single `helpers` Go package.

These packages retain their explicit foundation-layer classifications in
`internal/archtest/layers_test.go`. They may depend on other foundation
primitives, but do not own domain services, persistence or process wiring.
Small domain packages, HTTP adapters and owner-specific test support stay
with their owners; file count alone does not make a package a helper.

The directory groups existing implementations for navigation. Their formats,
validation rules, limits and error behavior remain owned by the same named
packages and specification documents.
