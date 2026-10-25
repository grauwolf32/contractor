package auditprofile_test

import (
	"encoding/json"
	"maps"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
)

func TestSharedRoundResultsBindingRequiresOneRequiredAssessmentInput(t *testing.T) {
	profile := loadProfile(t, copyAuditPreparationCatalog(t), "prepared-openapi-scan@1")
	binding := profile.Workflows["generate-api"]
	binding.Kind = auditprofile.AuditWorkflowAssessment
	binding.MaxRunAttempts = 0
	binding.Workflow.Inputs = maps.Clone(binding.Workflow.Inputs)
	slot := binding.Workflow.Inputs["existing_openapi"]
	slot.Required = true
	binding.Workflow.Inputs["existing_openapi"] = slot
	binding.Inputs = maps.Clone(binding.Inputs)
	binding.Inputs["existing_openapi"] = auditprofile.AuditWorkflowInputMapping{Source: auditprofile.AuditInputFromRoundResults}
	profile.Workflows["review-round"] = binding
	if err := auditprofile.ValidateAuditPreparationProfile(profile); err != nil {
		t.Fatal(err)
	}
	redigest(t, &profile)
	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = auditprofile.DecodeResolvedAuditProfileSnapshot(encoded); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*auditprofile.ResolvedAuditProfile){
		"check consumer": func(p *auditprofile.ResolvedAuditProfile) {
			b := p.Workflows["review-round"]
			b.Kind = auditprofile.AuditWorkflowCheck
			p.Workflows["review-round"] = b
		},
		"optional slot": func(p *auditprofile.ResolvedAuditProfile) {
			b := p.Workflows["review-round"]
			slot := b.Workflow.Inputs["existing_openapi"]
			slot.Required = false
			b.Workflow.Inputs["existing_openapi"] = slot
			p.Workflows["review-round"] = b
		},
		"source name": func(p *auditprofile.ResolvedAuditProfile) {
			p.Workflows["review-round"].Inputs["existing_openapi"] = auditprofile.AuditWorkflowInputMapping{Source: auditprofile.AuditInputFromRoundResults, Name: "latest"}
		},
		"producer role": func(p *auditprofile.ResolvedAuditProfile) {
			p.Workflows["review-round"].Inputs["existing_openapi"] = auditprofile.AuditWorkflowInputMapping{Source: auditprofile.AuditInputFromRoundResults, Role: "scan"}
		},
	} {
		candidate := profile.Clone()
		mutate(&candidate)
		redigest(t, &candidate)
		data, _ := json.Marshal(candidate)
		if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot(data); err == nil {
			t.Errorf("persisted binding accepted %s", name)
		}
	}
}
