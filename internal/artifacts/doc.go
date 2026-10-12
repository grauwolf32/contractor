// Package artifacts contains the single physical ArtifactStore and its
// authenticated UserScope and RunScope views.
//
// Payload integration inventory:
//   - PostgresRepository.Write: public/private writes, Planner Memory, Skill
//     uploads/seeding and trusted finding proposals.
//   - PostgresRepository.WriteAuditArtifact, through Service.WriteAuditArtifact:
//     every Server-generated audit-* binding (manifests, task packages,
//     worklists, retained standards, reports and direct-verification evidence
//     contracts), created frozen.
//   - PostgresRepository.Read: all scoped content consumers.
//   - ForkInput, ForkSkill, ImportAuditArtifact and PublishRunOutput: metadata
//     only, sharing the immutable blob without opening payload bytes.
//   - PostgresPurger: transactional reference checks; external physical deletion
//     belongs after the owning Run/Project/Audit transaction commits.
package artifacts
