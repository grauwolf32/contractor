// Package artifacts defines the ArtifactStore's scopes, references, validation
// and generic Service. Repository is the storage port; scope views apply the
// generic Namespace policy, while trusted primitives serve callers that own
// their purpose-specific authorization and Namespace rules.
//
// Package artifacts/postgres implements the registry: bindings, immutable
// revisions, lineage, retained-history admission, metadata reads and lifecycle
// purge. Its transaction-only operations compose derived context and Git
// provenance with ordinary Artifact publication in the caller's transaction.
//
// Package artifacts/blob defines physical payload objects, verification and
// transfer admission. The blob/backends factory selects the write policy;
// its filesystem and postgres adapters store physical files and inline bytes.
// PostgreSQL remains the registry for either payload backend.
//
// A prepared payload verifies a reusable physical generation before publication
// locks. Publication may attach only that verified key or its complete candidate.
// Physical deletion follows a definite registry commit; uncertain outcomes and
// displaced generations are left for offline cleanup.
package artifacts
