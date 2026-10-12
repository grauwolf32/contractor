package auditprofile_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestAuditCompletionRejectsInvalidTargetsAndTrustedMappings(t *testing.T) {
	profile := loadProfile(t, copyCatalog(t, "audit-completion-catalog"), "source-checklist@1")
	cases := map[string]func(*auditprofile.ResolvedAuditWorkflowBinding){
		"unknown-kind":     func(b *auditprofile.ResolvedAuditWorkflowBinding) { b.WorkerCompletion.Kind = "unknown@1" },
		"non-check":        func(b *auditprofile.ResolvedAuditWorkflowBinding) { b.Kind = auditprofile.AuditWorkflowDiscovery },
		"missing-stage":    func(b *auditprofile.ResolvedAuditWorkflowBinding) { b.WorkerCompletion.Stage = "absent" },
		"missing-agent":    func(b *auditprofile.ResolvedAuditWorkflowBinding) { b.WorkerCompletion.Agent = "absent" },
		"wrong-output":     func(b *auditprofile.ResolvedAuditWorkflowBinding) { delete(b.Outputs, "result") },
		"missing-task":     func(b *auditprofile.ResolvedAuditWorkflowBinding) { delete(b.Inputs, "task") },
		"missing-manifest": func(b *auditprofile.ResolvedAuditWorkflowBinding) { delete(b.Inputs, "execution_manifest") },
		"router": func(b *auditprofile.ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			s.Planner = resolved.PlannerRef{PlannerID: "router", Version: "1"}
			b.Workflow.Stages["check"] = s
		},
		"two-workers": func(b *auditprofile.ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			s.Agents["other"] = s.Agents["checker"]
		},
		"legacy-tool": func(b *auditprofile.ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			a := s.Agents["checker"]
			for i := range a.Template.Toolsets {
				if a.Template.Toolsets[i].Ref.ToolsetID == "audit-results" {
					a.Template.Toolsets[i].Ref.Version = "1"
				}
			}
			s.Agents["checker"] = a
		},
		"summarizer": func(b *auditprofile.ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			a := s.Agents["checker"]
			a.Template.Summarizer = &contracts.WorkerSummarizerConfig{}
			s.Agents["checker"] = a
		},
		"foreign-result": func(b *auditprofile.ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			r := s.Result.Artifacts["result"]
			r.From.Namespace = "foreign"
			s.Result.Artifacts["result"] = r
		},
		"duplicate-required-input": func(b *auditprofile.ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			s.Context.Artifacts["duplicate_task"] = s.Context.Artifacts["task"]
		},

		"optional-input": func(b *auditprofile.ResolvedAuditWorkflowBinding) {
			s := b.Workflow.Stages["check"]
			v := s.Context.Artifacts["task"]
			v.Required = false
			s.Context.Artifacts["task"] = v
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			copy := profile.Clone()
			binding := copy.Workflows["check"]
			mutate(&binding)
			if err := auditprofile.ValidateAuditWorkerCompletion(binding); err == nil {
				t.Fatal("invalid completion binding accepted")
			}
			copy.Workflows["check"] = binding
			redigest(t, &copy)
			raw, _ := json.Marshal(copy)
			if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot(raw); err == nil {
				t.Fatal("invalid immutable completion closure accepted")
			}
		})
	}
}
