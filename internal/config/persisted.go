package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// DecodeResolvedWorkflowSnapshot strictly decodes immutable Run authority.
// Every Stage must retain its explicit Worker session mode. Authoring defaults
// are resolved before persistence and are never inferred while reading a snapshot.
func DecodeResolvedWorkflowSnapshot(data []byte) (ResolvedWorkflow, error) {
	var workflow ResolvedWorkflow
	if err := decodeStrictSnapshot(data, &workflow); err != nil {
		return ResolvedWorkflow{}, err
	}
	for name, stage := range workflow.Stages {
		if err := stage.Session.Validate(); err != nil {
			return ResolvedWorkflow{}, fmt.Errorf("persisted Workflow Stage %q: %w", name, err)
		}
		if err := ValidateScanPlanStage(stage); err != nil {
			return ResolvedWorkflow{}, fmt.Errorf("persisted Workflow Stage %q: %w", name, err)
		}
		if stage.ScanPlan != nil {
			if err := validateStageExecutionConfig(name, stage); err != nil {
				return ResolvedWorkflow{}, err
			}
			if err := validateScanPlanInputMedia(workflow, stage); err != nil {
				return ResolvedWorkflow{}, err
			}
		}
	}
	return workflow, nil
}

// DecodeResolvedStageSnapshot strictly decodes the Stage copy retained by
// StageExecution, including its explicit Worker session mode.
func DecodeResolvedStageSnapshot(data []byte) (ResolvedStage, error) {
	var stage ResolvedStage
	if err := decodeStrictSnapshot(data, &stage); err != nil {
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
	if err := decodeStrictSnapshot(data, &profile); err != nil {
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
	expected, err := auditProfileDigest(selector, profile)
	if err != nil || expected != profile.Ref.Digest {
		return ResolvedAuditProfile{}, fmt.Errorf("persisted AuditProfile digest is invalid")
	}
	return cloneAuditProfile(profile), nil
}

func decodeStrictSnapshot(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return fmt.Errorf("persisted snapshot contains trailing JSON")
	}
	return nil
}
