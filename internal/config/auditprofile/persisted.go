package auditprofile

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

// JSON members and PostgreSQL text-array paths of persisted snapshots that SQL
// reads: Audit queries read the ResolvedAuditProfile an Audit retains and
// Finding intake its interaction policy. Renaming a tag must rename its
// constant; internal/persistence pins each one against an encoded value and
// the migrations.
const (
	AuditProfileWorkflowsMember         = "workflows"
	AuditWorkflowKindMember             = "kind"
	AuditWorkflowMaxRunAttemptsMember   = "maxRunAttempts"
	AuditProfileFindingConfirmationPath = "{interaction,findingConfirmation}"
)

// DecodeResolvedAuditProfileSnapshot strictly decodes and validates the
// complete immutable profile closure retained by an Audit draft. The embedded
// digest must still match the effective workflows and policies in the body;
// every Workflow role must retain its explicit kind. No role inference or
// current catalog lookup participates in recovery or start replay.
func DecodeResolvedAuditProfileSnapshot(data []byte) (ResolvedAuditProfile, error) {
	var profile ResolvedAuditProfile
	if err := strictjson.Decode(data, &profile); err != nil {
		return ResolvedAuditProfile{}, err
	}
	_, err := resolved.ParseSelector(profile.Ref.Name + "@" + profile.Ref.Version)
	if err != nil || profile.Ref.Digest == "" || !profile.Mode.valid() ||
		len(profile.Inputs) == 0 || len(profile.Workflows) == 0 {
		return ResolvedAuditProfile{}, fmt.Errorf("persisted AuditProfile identity or shape is invalid")
	}
	for role, binding := range profile.Workflows {
		if err := validateAuditMapKey("persisted AuditProfile workflow role", role); err != nil {
			return ResolvedAuditProfile{}, err
		}
		if !binding.Kind.valid() {
			return ResolvedAuditProfile{}, fmt.Errorf(
				"persisted AuditProfile workflow %q kind is invalid", role,
			)
		}
		if err := ValidateAuditWorkerCompletion(binding); err != nil {
			return ResolvedAuditProfile{}, err
		}
		if err := resolved.ValidateWorkflowGraph(binding.Workflow); err != nil {
			return ResolvedAuditProfile{}, fmt.Errorf("persisted AuditProfile workflow %q: %w", role, err)
		}
		if _, err := resolved.WorkflowSkillRefs(binding.Workflow); err != nil {
			return ResolvedAuditProfile{}, fmt.Errorf("persisted AuditProfile workflow %q: %w", role, err)
		}
	}
	if err := ValidateAuditPreparationProfile(profile); err != nil {
		return ResolvedAuditProfile{}, err
	}
	if err := ValidateAuditTaskProfile(profile); err != nil {
		return ResolvedAuditProfile{}, err
	}
	expected, err := Digest(profile)
	if err != nil || expected != profile.Ref.Digest {
		return ResolvedAuditProfile{}, fmt.Errorf("persisted AuditProfile digest is invalid")
	}
	return cloneAuditProfile(profile), nil
}
