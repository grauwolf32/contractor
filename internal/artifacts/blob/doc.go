// Package blob defines immutable physical payload objects, integrity checks and
// process-wide transfer admission. Its Storage protocol is implemented by the
// filesystem and inline PostgreSQL adapters in blob/backends. Logical Artifact
// identity, authorization, history and registry transactions belong to their
// ArtifactStore owners.
package blob
