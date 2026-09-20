package evalservice

import (
	"context"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func (b preflightBinding) evaluateCase(ctx context.Context, db pg.DBTX, service *artifacts.Service, owner string, c evaldomain.Case, v evaldomain.Variant, capabilities []string) (Eligibility, error) {
	for _, ref := range c.Inputs {
		if err := verifyArtifact(ctx, db, service, owner, ref); err != nil {
			return Eligibility{}, err
		}
	}
	eligibility := caseEligibility(c, v, b.snapshot, capabilities)
	if b.snapshot.Audit == nil || eligibility.State != "eligible" {
		return eligibility, nil
	}
	inputs, err := readAuditInputs(ctx, service, c, v)
	if err != nil {
		return Eligibility{}, err
	}
	params := MapParameters(c, v)
	scope := auditservice.Scope{
		Objective:          c.Task.Objective,
		Target:             params["target"],
		AuthorizationScope: params["authorizationScope"],
	}
	if objective, ok := params["objective"]; ok {
		scope.Objective = objective
	}
	if err := auditservice.ValidateInputPreview(*b.snapshot.Audit, scope, inputs, b.standards); err != nil {
		return unavailable("The Audit input or inventory is not supported by the selected profile."), nil
	}
	return eligibility, nil
}

func readAuditInputs(ctx context.Context, service *artifacts.Service, c evaldomain.Case, v evaldomain.Variant) (map[string]artifacts.ReadResult, error) {
	mapped, err := MapInputs(c, v)
	if err != nil {
		return nil, err
	}
	inputs := make(map[string]artifacts.ReadResult, len(mapped))
	for slot, ref := range mapped {
		store, err := artifactScope(service, ref)
		if err != nil {
			return nil, err
		}
		read, err := store.Read(ctx, contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &ref.Revision})
		if err != nil {
			return nil, preflightDependencyError(err)
		}
		if evaldomain.Digest(read.Payload.Data) != ref.SHA256 {
			return nil, evaldomain.Failure("eval_pin_mismatch")
		}
		inputs[slot] = read
	}
	return inputs, nil
}

func unavailable(reason string) Eligibility {
	return Eligibility{State: "unsupported", Reason: &reason}
}

func caseEligibility(c evaldomain.Case, v evaldomain.Variant, s BindingSnapshot, capabilities []string) Eligibility {
	known := map[string]bool{}
	for _, cap := range capabilities {
		known[cap] = true
	}
	for _, required := range c.Requires {
		if !known[required] {
			return unavailable("The binding does not declare a required capability.")
		}
	}
	inputs, err := MapInputs(c, v)
	if err != nil {
		return unavailable("The input mapping is incompatible with this case.")
	}
	slots := map[string]config.ArtifactSlot{}
	outputs := map[string]config.ArtifactSlot{}
	if s.Workflow != nil {
		slots = s.Workflow.Inputs
		outputs = s.Workflow.Outputs
		params := MapParameters(c, v)
		for name := range params {
			if _, ok := s.Workflow.Parameters[name]; !ok {
				return unavailable("The case supplies an unknown Workflow parameter.")
			}
		}
		for name, slot := range s.Workflow.Parameters {
			if _, ok := params[name]; slot.Required && !ok {
				return unavailable("A required Workflow parameter is missing.")
			}
		}
	} else {
		for name, slot := range s.Audit.Inputs {
			slots[name] = config.ArtifactSlot{Required: slot.Required, MediaTypes: slot.MediaTypes}
		}
		outputs["report"] = config.ArtifactSlot{MediaTypes: []string{"application/json", "text/markdown"}}
		for name := range MapParameters(c, v) {
			if name != "objective" && name != "target" && name != "authorizationScope" {
				return unavailable("The Audit scope has an unsupported parameter.")
			}
		}
	}
	for name, ref := range inputs {
		slot, ok := slots[name]
		if !ok || !slices.Contains(slot.MediaTypes, ref.MediaType) {
			return unavailable("An input slot or media type is incompatible.")
		}
	}
	for name, slot := range slots {
		if _, ok := inputs[name]; slot.Required && !ok {
			return unavailable("A required input is missing.")
		}
	}
	for role, contract := range c.Outputs {
		name := role
		if mapped, ok := v.OutputMapping[role]; ok {
			name = mapped
		}
		slot, ok := outputs[name]
		if !ok && contract.Required {
			return unavailable("A required output role is unavailable.")
		}
		if ok {
			compatible := false
			for _, media := range contract.MediaTypes {
				compatible = compatible || slices.Contains(slot.MediaTypes, media)
			}
			if !compatible {
				return unavailable("An output media type is incompatible.")
			}
		}
	}
	return Eligibility{State: "eligible"}
}

// Mappings use case role -> executable slot. Omitted entries keep their name.
func MapInputs(c evaldomain.Case, v evaldomain.Variant) (map[string]evaldomain.Artifact, error) {
	out := map[string]evaldomain.Artifact{}
	for role := range v.InputMapping {
		if _, ok := c.Inputs[role]; !ok {
			return nil, evaldomain.Failure("eval_invalid")
		}
	}
	for role, ref := range c.Inputs {
		slot := role
		if mapped, ok := v.InputMapping[role]; ok {
			slot = mapped
		}
		if _, ok := out[slot]; ok {
			return nil, evaldomain.Failure("eval_invalid")
		}
		out[slot] = ref
	}
	return out, nil
}

func MapParameters(c evaldomain.Case, v evaldomain.Variant) map[string]string {
	out := map[string]string{}
	for k, value := range c.Task.Parameters {
		out[k] = value
	}
	for k, value := range v.Parameters {
		if value == "$task.objective" {
			value = c.Task.Objective
		}
		out[k] = value
	}
	return out
}
