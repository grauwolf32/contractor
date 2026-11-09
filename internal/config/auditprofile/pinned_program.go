package auditprofile

import (
	"fmt"
	"maps"
	"slices"

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

const MaxAuditSnapshotBytes = 16 << 20

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
	result.Ref.Digest, err = result.Digest()
	if err != nil {
		return PinnedProgram{}, err
	}
	encoded, err := strictjson.Canonical(result)
	if err != nil || len(encoded) > MaxAuditSnapshotBytes {
		return PinnedProgram{}, invalidPinnedProgram()
	}
	return result, nil
}

// DecodePinnedProgram validates immutable content without consulting a mutable
// catalog. No old profile reader or Workflow inference participates in replay.
func DecodePinnedProgram(data []byte) (PinnedProgram, error) {
	var pinned PinnedProgram
	if len(data) > MaxAuditSnapshotBytes || strictjson.Decode(data, &pinned) != nil || !contentdigest.Valid(pinned.Ref.Digest) {
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

func (p PinnedProgram) Digest() (string, error) {
	return contentdigest.CanonicalJSON(map[string]any{
		"apiVersion": contracts.APIVersion, "kind": Kind,
		"metadata": map[string]string{"name": p.Ref.Name, "version": p.Ref.Version},
		"program":  p.Program, "workflows": p.Workflows,
	})
}

func (p PinnedProgram) Clone() PinnedProgram {
	if p.Program.SeedSource != nil {
		source := *p.Program.SeedSource
		source.EntryIDs = slices.Clone(source.EntryIDs)
		if source.Standard != nil {
			standard := *source.Standard
			source.Standard = &standard
		}
		p.Program.SeedSource = &source
	}
	p.Workflows = maps.Clone(p.Workflows)
	for alias, workflow := range p.Workflows {
		p.Workflows[alias] = workflow.Clone()
	}
	p.Program.Parameters = maps.Clone(p.Program.Parameters)
	p.Program.Inputs = maps.Clone(p.Program.Inputs)
	for name, input := range p.Program.Inputs {
		input.MediaTypes = slices.Clone(input.MediaTypes)
		p.Program.Inputs[name] = input
	}
	p.Program.Workflows = maps.Clone(p.Program.Workflows)
	for alias, binding := range p.Program.Workflows {
		if binding.MaxAttempts != nil {
			ceiling := *binding.MaxAttempts
			binding.MaxAttempts = &ceiling
		}
		binding.ParameterInputs = maps.Clone(binding.ParameterInputs)
		binding.Inputs, binding.Parameters, binding.Outputs = maps.Clone(binding.Inputs), maps.Clone(binding.Parameters), maps.Clone(binding.Outputs)
		p.Program.Workflows[alias] = binding
	}
	p.Program.Hypotheses = maps.Clone(p.Program.Hypotheses)
	for key, seed := range p.Program.Hypotheses {
		seed.Scope = maps.Clone(seed.Scope)
		p.Program.Hypotheses[key] = seed
	}
	p.Program.InitialWork = slices.Clone(p.Program.InitialWork)
	for index := range p.Program.InitialWork {
		p.Program.InitialWork[index].Inputs = maps.Clone(p.Program.InitialWork[index].Inputs)
	}
	if p.Program.Continuation != nil {
		continuation := *p.Program.Continuation
		p.Program.Continuation = &continuation
	}
	return p
}

func invalidPinnedProgram() error { return fmt.Errorf("pinned Audit program is invalid") }
