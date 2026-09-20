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
// no current catalog lookup participates in recovery or start replay.
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
	legacyRoles, err := persistedAuditProfileLegacyRoles(data, profile)
	if err != nil {
		return ResolvedAuditProfile{}, err
	}
	for role, binding := range profile.Workflows {
		if err := validateAuditMapKey("persisted AuditProfile workflow role", role); err != nil {
			return ResolvedAuditProfile{}, err
		}
		if legacyRoles {
			// Before role kinds were persisted every Audit Workflow binding had
			// the check-role behavior. Infer that historical meaning only after
			// the legacy digest has authenticated the untouched snapshot.
			binding.Kind = AuditWorkflowCheck
			profile.Workflows[role] = binding
		} else if !binding.Kind.valid() {
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
	var expected string
	if legacyRoles {
		expected, err = auditProfileLegacyDigest(selector, profile)
	} else {
		expected, err = auditProfileDigest(selector, profile)
	}
	if err != nil || expected != profile.Ref.Digest {
		return ResolvedAuditProfile{}, fmt.Errorf("persisted AuditProfile digest is invalid")
	}
	return cloneAuditProfile(profile), nil
}

// persistedAuditProfileLegacyRoles distinguishes the one historical snapshot
// schema that omitted Workflow role kinds. A mixed snapshot is neither a
// legacy document nor a current one and is rejected instead of guessed.
func persistedAuditProfileLegacyRoles(
	data []byte, profile ResolvedAuditProfile,
) (bool, error) {
	var shape struct {
		Workflows map[string]map[string]json.RawMessage `json:"workflows"`
	}
	if err := json.Unmarshal(data, &shape); err != nil {
		return false, err
	}
	if len(shape.Workflows) != len(profile.Workflows) {
		return false, fmt.Errorf("persisted AuditProfile Workflow map is invalid")
	}
	missing := 0
	for role := range profile.Workflows {
		fields, exists := shape.Workflows[role]
		if !exists {
			return false, fmt.Errorf("persisted AuditProfile workflow %q is absent", role)
		}
		if _, exists := fields["kind"]; !exists {
			missing++
		}
	}
	if missing != 0 && missing != len(profile.Workflows) {
		return false, fmt.Errorf("persisted AuditProfile mixes legacy and current Workflow roles")
	}
	return missing == len(profile.Workflows), nil
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
