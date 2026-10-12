// Package artifacts contains the single physical ArtifactStore and its
// authenticated UserScope, ProjectScope and RunScope views.
//
// Service is the only entry point of the store's operations and validates
// every request once. Its scope views apply the generic Namespace policy; its
// trusted primitives (Create, CreateFrozen, Fork, Freeze and RequireExact)
// serve the purpose-specific flows of other packages, which own their
// Namespaces: runrepeat retains the repeat request, runservice forks inputs,
// the Scheduler binds, freezes and publishes outputs, findingintake writes
// proposals, agentskills and auditstandards seed and fork catalog packages,
// and the Audit packages import and write Audit material.
//
// Payload integration inventory:
//   - PostgresRepository.Write: public/private writes, Planner Memory, Skill
//     uploads/seeding, Run repeat requests and trusted finding proposals.
//   - PostgresRepository.CreateFrozen, through Service.CreateFrozen: every
//     Server-generated audit-* binding (manifests, task packages, worklists,
//     retained standards, reports and direct-verification evidence contracts).
//   - PostgresRepository.Read: all scoped content consumers.
//   - PostgresRepository.Fork: metadata only, sharing the immutable blob
//     without opening payload bytes.
//   - PostgresPurger: transactional reference checks; external physical deletion
//     belongs after the owning Run/Project/Audit transaction commits.
package artifacts
