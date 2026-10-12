package config

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/strictjson"
)

// MaxWorkflowSnapshotBytes is LIM-CAT-4: the largest Workflow snapshot a Run
// may store, counted in bytes of the encoding EncodeResolvedWorkflowSnapshot
// produces (CAT-R76).
const MaxWorkflowSnapshotBytes = 2 << 20

// ErrWorkflowSnapshotTooLarge classifies a resolved Workflow whose snapshot
// exceeds MaxWorkflowSnapshotBytes. It is a permanent validation failure.
var ErrWorkflowSnapshotTooLarge = errors.New("Workflow snapshot exceeds its size limit")

// EncodeResolvedWorkflowSnapshot returns the snapshot a Run stores for
// workflow. The catalog, Run creation, the Run store and the Scheduler all
// measure this encoding, never the text PostgreSQL returns for the stored
// jsonb, so a snapshot accepted when its Run is created stays executable
// (CAT-R76).
func EncodeResolvedWorkflowSnapshot(workflow ResolvedWorkflow) ([]byte, error) {
	data, err := json.Marshal(workflow)
	if err != nil {
		return nil, fmt.Errorf("encode Workflow %s@%s snapshot: %w", workflow.Ref.Name, workflow.Ref.Version, err)
	}
	if len(data) > MaxWorkflowSnapshotBytes {
		return nil, fmt.Errorf(
			"%w: Workflow %s@%s snapshot has %d bytes, more than %d",
			ErrWorkflowSnapshotTooLarge, workflow.Ref.Name, workflow.Ref.Version, len(data), MaxWorkflowSnapshotBytes,
		)
	}
	return data, nil
}

// JSON members and PostgreSQL text-array paths of persisted snapshots that SQL
// reads: Audit queries read the ResolvedAuditProfile an Audit retains, Finding
// intake its interaction policy and Eval metrics the Planner of a stored
// ResolvedStage. Renaming a tag must rename its constant; internal/persistence
// pins each one against an encoded value and the migrations.
const (
	AuditProfileWorkflowsMember         = "workflows"
	AuditWorkflowKindMember             = "kind"
	AuditWorkflowMaxRunAttemptsMember   = "maxRunAttempts"
	AuditProfileFindingConfirmationPath = "{interaction,findingConfirmation}"
	StagePlannerIDPath                  = "{planner,plannerId}"
	StagePlannerVersionPath             = "{planner,version}"
)

// Clone returns a deep copy of workflow that shares no mutable state with it,
// so a decoded snapshot can be reused without a holder seeing another's edits.
func (workflow ResolvedWorkflow) Clone() ResolvedWorkflow { return cloneWorkflow(workflow) }

// DecodeResolvedWorkflowSnapshot strictly decodes immutable Run authority.
// Every Stage must retain its explicit Worker session mode. Authoring defaults
// are resolved before persistence and are never inferred while reading a snapshot.
func DecodeResolvedWorkflowSnapshot(data []byte) (ResolvedWorkflow, error) {
	var workflow ResolvedWorkflow
	if err := strictjson.Decode(data, &workflow); err != nil {
		return ResolvedWorkflow{}, err
	}
	// Every reader gets a Workflow that passes the same complete graph check
	// as a catalog load (CAT-R78), so no caller validates it again.
	if err := ValidateWorkflowGraph(workflow); err != nil {
		return ResolvedWorkflow{}, fmt.Errorf("persisted Workflow: %w", err)
	}
	return workflow, nil
}

// DecodeResolvedStageSnapshot strictly decodes the Stage copy retained by
// StageExecution, including its explicit Worker session mode.
func DecodeResolvedStageSnapshot(data []byte) (ResolvedStage, error) {
	var stage ResolvedStage
	if err := strictjson.Decode(data, &stage); err != nil {
		return ResolvedStage{}, err
	}
	if err := validateStageAgentNames("persisted Stage", stage.Agents); err != nil {
		return ResolvedStage{}, err
	}
	if err := stage.Session.Validate(); err != nil {
		return ResolvedStage{}, err
	}
	if err := ValidateScanPlanStage(stage); err != nil {
		return ResolvedStage{}, err
	}
	if stage.ScanPlan != nil {
		if err := validateStageExecutionConfig("persisted", stage); err != nil {
			return ResolvedStage{}, err
		}
	}
	return stage, nil
}

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
	selector, err := ParseSelector(profile.Ref.Name + "@" + profile.Ref.Version)
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
		if err := ValidateWorkflowGraph(binding.Workflow); err != nil {
			return ResolvedAuditProfile{}, fmt.Errorf("persisted AuditProfile workflow %q: %w", role, err)
		}
		if _, err := WorkflowSkillRefs(binding.Workflow); err != nil {
			return ResolvedAuditProfile{}, fmt.Errorf("persisted AuditProfile workflow %q: %w", role, err)
		}
	}
	if err := ValidateAuditPreparationProfile(profile); err != nil {
		return ResolvedAuditProfile{}, err
	}
	if err := ValidateAuditTaskProfile(profile); err != nil {
		return ResolvedAuditProfile{}, err
	}
	expected, err := auditProfileDigest(selector, profile)
	if err != nil || expected != profile.Ref.Digest {
		return ResolvedAuditProfile{}, fmt.Errorf("persisted AuditProfile digest is invalid")
	}
	return cloneAuditProfile(profile), nil
}
