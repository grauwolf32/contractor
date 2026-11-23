// Package controlplane owns Runtime Agent capacity and allocation lifecycle.
//
// Placement stages:
//   - placement.go coordinates reservation, admission and Registry commit.
//   - placement_candidates.go builds compatible edges and refreshes principals.
//   - placement_resolution.go resolves settings with the attempt-scoped cache.
//   - placement_catalogs.go reads Gateway and credential metadata.
//   - placement_pinning.go rechecks selected settings and pins them in one transaction.
//
// Runtime protocol stages:
//   - runtime_client.go constructs and closes the connection owner.
//   - runtime_prepare.go builds allocation specs and validates returned Agent Cards.
//   - runtime_lifecycle.go finalizes, aborts and releases allocations.
//   - runtime_worker_state.go reads conditional Worker State snapshots.
//   - runtime_transport.go sends principal-bound HTTP requests and bounds responses.
//
// RuntimeBatchController coordinates these lifecycle calls across a Stage's
// reservations. The separately composed engagementadapter owns Audit integration
// through AllocationSpecPreparation; the core does not import it.
package controlplane
