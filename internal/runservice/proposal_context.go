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
func materializeProposalContext(ctx context.Context, service *artifacts.Service, intent auditstore.RunCreationIntent, p auditRunCreate, pinned auditprofile.PinnedProgram, runID string, prepared *artifacts.PreparedInputs) error {
	binding := pinned.Program.Workflows[intent.Execution.WorkflowRole]
	target, err := auditprofile.ResolveProposalTarget(p.Workflow, binding)
	if err != nil {
		return err
	}
	if target == nil {
		return nil
	}
	scope, _ := artifacts.RunScope(runID)
	value := auditcontract.ProposalContext{Schema: auditcontract.ProposalContextSchema, RunID: runID, Program: pinned.Program, Materials: map[string]auditcontract.Material{}, InputSources: map[string]auditcontract.Material{}}
	for handle, material := range prepared.Materials {
		value.Materials[handle] = material.Local
		value.InputSources[handle] = material.Source
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
	prepared.Inputs[target.ContextSlot] = contracts.ExactContent(written.Ref, written.MediaType, data)
	roles := auditcontract.ProposalBinding{Schema: auditcontract.ProposalBindingSchema, RunID: runID, Context: written.Ref, Result: target.Result}
	data, err = strictjson.Canonical(roles)
	if err != nil {
		return err
	}
	_, err = service.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: artifactpolicy.AuditContextNamespace, Name: auditcontract.ProposalBindingName}, artifacts.Payload{MediaType: "application/json", Data: data})
	return err
}
