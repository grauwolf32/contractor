package runservice

import (
	"context"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// Proposal context is assembled only after ordinary inputs have real bindings.
// The native material-set owner supplies original identities for accounting;
// models see exact local refs, never new access from a descriptor's text.
func materializeProposalContext(ctx context.Context, service *artifacts.Service, intent auditstore.RunCreationIntent, p auditRunCreate, pinned auditprofile.PinnedProgram, runID string, forks map[string]contracts.ArtifactRef, originals []contracts.MaterialDescriptor) error {
	binding := pinned.Program.Workflows[intent.Execution.WorkflowRole]
	target, err := auditprofile.ResolveProposalTarget(p.Workflow, binding)
	if err != nil {
		return err
	}
	if target == nil {
		return nil
	}
	scope, _ := artifacts.RunScope(runID)
	run, _ := service.Scoped(scope)
	value := auditcontract.ProposalContext{Schema: auditcontract.ProposalContextSchema, RunID: runID, Program: pinned.Program, Materials: map[string]auditcontract.Material{}, InputSources: map[string]auditcontract.Material{}}
	origins := map[string]contracts.MaterialDescriptor{}
	for _, m := range originals {
		origins[contracts.MaterialHandle(m)] = m
	}
	for _, slot := range sortedInputSlots(p.Inputs) {
		mapping := binding.Inputs[slot]
		if mapping.Source == "task-context" || mapping.Source == "prior-results" {
			continue
		}
		local, exists := forks[slot]
		if !exists {
			return ErrInvalid
		}
		if p.Workflow.Inputs[slot].MaterialSet {
			raw, err := run.Read(ctx, local)
			if err != nil {
				return err
			}
			set, err := contracts.DecodeMaterialSet(raw.Payload.Data)
			if err != nil {
				return err
			}
			for _, m := range set.Materials {
				original, found := origins[m.Ref.Name]
				if !found || m.ScopeKind != "run" || m.ScopeID != runID {
					return ErrInvalid
				}
				if _, exists := value.Materials[m.Ref.Name]; exists {
					continue
				}
				value.Materials[m.Ref.Name] = m
				value.InputSources[m.Ref.Name] = original
			}
		} else {
			metadata, err := run.Metadata(ctx, local)
			if err != nil {
				return err
			}
			input := p.Inputs[slot]
			original := auditcontract.Material{ScopeKind: "audit", ScopeID: intent.Execution.AuditID, Ref: input.Ref, Digest: input.Digest, MediaType: input.MediaType, SizeBytes: input.SizeBytes}
			handle := contracts.MaterialHandle(original)
			if _, exists := value.Materials[handle]; exists {
				continue
			}
			value.Materials[handle] = auditcontract.Material{ScopeKind: "run", ScopeID: runID, Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}
			value.InputSources[handle] = original
		}
	}
	budget := &contracts.MaterialBudget{}
	for _, m := range value.InputSources {
		if err := budget.Add(m); err != nil {
			return err
		}
	}
	for _, source := range intent.Observations {
		key := "observation-" + strings.TrimPrefix(contentdigest.Bytes([]byte(source.TaskID)), "sha256:")
		if source.Result != nil {
			key = contracts.MaterialHandle(*source.Result)
			if _, exists := value.Materials[key]; !exists {
				return ErrInvalid
			}
		}
		item := auditcontract.ProposalObservation{TaskID: source.TaskID, Key: key, Hypothesis: source.Hypothesis}
		if source.GapCode != nil {
			item.Gap = *source.GapCode
		}
		value.Observations = append(value.Observations, item)
	}
	data, err := strictjson.Canonical(value)
	if err != nil {
		return err
	}
	data, err = auditcontract.Decode(data, nil)
	if err != nil {
		return err
	}
	written, err := service.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: "inputs", Name: target.ContextSlot}, artifacts.Payload{MediaType: "application/json", Data: data})
	if err != nil {
		return err
	}
	forks[target.ContextSlot] = written.Ref
	roles := auditcontract.ProposalBinding{Schema: auditcontract.ProposalBindingSchema, RunID: runID, Context: written.Ref, Result: target.Result}
	data, err = strictjson.Canonical(roles)
	if err != nil {
		return err
	}
	_, err = service.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: artifactpolicy.AuditContextNamespace, Name: auditcontract.ProposalBindingName}, artifacts.Payload{MediaType: "application/json", Data: data})
	return err
}
