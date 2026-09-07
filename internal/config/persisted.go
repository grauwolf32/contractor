package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// DecodeResolvedWorkflowSnapshot strictly decodes immutable Run authority.
// Snapshots written before Worker session modes existed are the only boundary
// where an absent Stage session means shared; newly authored Workflows are
// normalized to an explicit isolated value before persistence.
func DecodeResolvedWorkflowSnapshot(data []byte) (ResolvedWorkflow, error) {
	var workflow ResolvedWorkflow
	if err := decodeStrictSnapshot(data, &workflow); err != nil {
		return ResolvedWorkflow{}, err
	}
	var shape struct {
		Stages map[string]json.RawMessage `json:"stages"`
	}
	if err := json.Unmarshal(data, &shape); err != nil {
		return ResolvedWorkflow{}, err
	}
	if len(shape.Stages) != len(workflow.Stages) {
		return ResolvedWorkflow{}, fmt.Errorf("persisted Workflow Stage map is invalid")
	}
	for name, stage := range workflow.Stages {
		raw, ok := shape.Stages[name]
		if !ok {
			return ResolvedWorkflow{}, fmt.Errorf("persisted Workflow Stage %q is absent", name)
		}
		if err := normalizePersistedStageSession(raw, &stage); err != nil {
			return ResolvedWorkflow{}, fmt.Errorf("persisted Workflow Stage %q: %w", name, err)
		}
		workflow.Stages[name] = stage
	}
	return workflow, nil
}

// DecodeResolvedStageSnapshot applies the same legacy boundary to the Stage
// copy retained by StageExecution.
func DecodeResolvedStageSnapshot(data []byte) (ResolvedStage, error) {
	var stage ResolvedStage
	if err := decodeStrictSnapshot(data, &stage); err != nil {
		return ResolvedStage{}, err
	}
	if err := normalizePersistedStageSession(data, &stage); err != nil {
		return ResolvedStage{}, err
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

func normalizePersistedStageSession(data []byte, stage *ResolvedStage) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	raw, present := fields["session"]
	if !present {
		stage.Session = contracts.WorkerSessionShared
		return nil
	}
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return fmt.Errorf("session must be isolated or shared")
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return fmt.Errorf("session must be isolated or shared")
	}
	mode := contracts.WorkerSessionMode(value)
	if err := mode.Validate(); err != nil {
		return err
	}
	stage.Session = mode
	return nil
}
