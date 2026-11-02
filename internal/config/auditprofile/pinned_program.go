package auditprofile

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// PinnedProgram is the common program and the complete allowed ordinary
// Workflow closures retained by its owner. Recovery uses these exact closures.
type PinnedProgram struct {
	Ref       AuditProfileRef                      `json:"ref"`
	Program   auditcontract.Program                `json:"program"`
	Workflows map[string]resolved.ResolvedWorkflow `json:"workflows"`
}

func PinProgram(selector resolved.Selector, data []byte, lookup func(string) (resolved.ResolvedWorkflow, error)) (PinnedProgram, error) {
	program, err := auditcontract.DecodeProgram(data)
	if err != nil {
		return PinnedProgram{}, err
	}
	if _, err := resolved.ParseSelector(selector.String()); err != nil {
		return PinnedProgram{}, err
	}
	workflows, err := ResolveProgramWorkflows(program, lookup)
	if err != nil {
		return PinnedProgram{}, err
	}
	for alias, workflow := range workflows {
		if err := resolved.ValidateWorkflowGraph(workflow); err != nil {
			return PinnedProgram{}, err
		}
		if err := resolved.ValidateWorkflowExecutionConfigs(workflow); err != nil {
			return PinnedProgram{}, err
		}
		if _, err := resolved.WorkflowSkillRefs(workflow); err != nil {
			return PinnedProgram{}, err
		}
		workflows[alias] = workflow.Clone()
	}
	result := PinnedProgram{Ref: AuditProfileRef{Name: selector.ID, Version: selector.Version}, Program: program, Workflows: workflows}
	result.Ref.Digest, err = result.digest()
	if err != nil {
		return PinnedProgram{}, err
	}
	return result, nil
}

// DecodePinnedProgram validates immutable content without consulting a mutable
// catalog. No old profile reader or Workflow inference participates in replay.
func DecodePinnedProgram(data []byte) (PinnedProgram, error) {
	var pinned PinnedProgram
	if len(data) > 8<<20 || strictjson.Decode(data, &pinned) != nil || !contentdigest.Valid(pinned.Ref.Digest) {
		return PinnedProgram{}, invalidPinnedProgram()
	}
	selector, err := resolved.ParseSelector(pinned.Ref.Name + "@" + pinned.Ref.Version)
	if err != nil || len(pinned.Workflows) != len(pinned.Program.Workflows) {
		return PinnedProgram{}, invalidPinnedProgram()
	}
	byRef := map[string]resolved.ResolvedWorkflow{}
	for alias, workflow := range pinned.Workflows {
		binding, exists := pinned.Program.Workflows[alias]
		ref := workflow.Ref.Name + "@" + workflow.Ref.Version
		if !exists || binding.Ref != ref {
			return PinnedProgram{}, invalidPinnedProgram()
		}
		if prior, exists := byRef[ref]; exists {
			a, aErr := contentdigest.CanonicalJSON(prior)
			b, bErr := contentdigest.CanonicalJSON(workflow)
			if aErr != nil || bErr != nil || a != b {
				return PinnedProgram{}, invalidPinnedProgram()
			}
		}
		byRef[ref] = workflow
	}
	program, err := strictjson.Canonical(pinned.Program)
	if err != nil {
		return PinnedProgram{}, invalidPinnedProgram()
	}
	verified, err := PinProgram(selector, program, func(ref string) (resolved.ResolvedWorkflow, error) {
		workflow, exists := byRef[ref]
		if !exists {
			return resolved.ResolvedWorkflow{}, invalidPinnedProgram()
		}
		return workflow, nil
	})
	if err != nil || verified.Ref.Digest != pinned.Ref.Digest {
		return PinnedProgram{}, invalidPinnedProgram()
	}
	return verified, nil
}

func (p PinnedProgram) digest() (string, error) {
	return contentdigest.CanonicalJSON(map[string]any{
		"apiVersion": contracts.APIVersion, "kind": Kind,
		"metadata": map[string]string{"name": p.Ref.Name, "version": p.Ref.Version},
		"program":  p.Program, "workflows": p.Workflows,
	})
}

func invalidPinnedProgram() error { return fmt.Errorf("pinned Audit program is invalid") }
