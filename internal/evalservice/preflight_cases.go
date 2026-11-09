package evalservice

import (
	"context"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func (b preflightBinding) evaluateCase(ctx context.Context, db pg.DBTX, service *artifacts.Service, owner string, c evaldomain.Case, v evaldomain.Variant, capabilities []string) (Eligibility, error) {
	for _, ref := range c.Inputs {
		if err := verifyArtifact(ctx, db, service, owner, ref); err != nil {
			return Eligibility{}, err
		}
	}
	return caseEligibility(c, v, b.snapshot, capabilities), nil
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
	slots := map[string]resolved.ArtifactSlot{}
	outputs := map[string]resolved.ArtifactSlot{}
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
		for name, slot := range s.Audit.Program.Inputs {
			slots[name] = resolved.ArtifactSlot{Required: slot.Required, MediaTypes: slot.MediaTypes}
		}
		for name, output := range auditOutputContracts {
			outputs[name] = resolved.ArtifactSlot{MediaTypes: []string{output.mediaType}}
		}
		if _, _, err := mapAuditParameters(c, v, s.Audit.Program); err != nil {
			return unavailable("The Audit launch parameters do not match the selected Program.")
		}
	}
	for name, ref := range inputs {
		slot, ok := slots[name]
		if !ok || !slices.Contains(slot.MediaTypes, ref.MediaType) {
			return unavailable("An input slot or media type is incompatible.")
		}
		if slot.SourceScope == "user" && ref.Scope != "user" {
			return unavailable("This Workflow input requires an exact artifact from the owner's UserScope.")
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
