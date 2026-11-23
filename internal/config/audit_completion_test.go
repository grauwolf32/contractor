package config

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
)

func TestAuditCompletionDerivesOneOrdinaryWorkerTarget(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t, "audit-completion-catalog"))
	pin, err := snapshot.AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	binding := pin.Program.Workflows["check"]
	workflow := pin.Workflows["check"]
	target, err := auditprofile.ResolveCompletionTarget(workflow, binding)
	if err != nil || target == nil || target.Stage != "check" || target.Agent != "checker" || target.ContextSlot != "context" || target.Result.Namespace != "audit-check" || target.Result.Name != "result" {
		t.Fatalf("target=%+v error=%v", target, err)
	}
	for name, change := range map[string]func(*resolved.ResolvedStage){
		"no context": func(s *resolved.ResolvedStage) { s.Context.Artifacts = nil },
		"optional context": func(s *resolved.ResolvedStage) {
			r := s.Context.Artifacts["context"]
			r.Required = false
			s.Context.Artifacts["context"] = r
		},
		"no declared output": func(s *resolved.ResolvedStage) { s.WorkflowOutputs = nil },
		"two Workers":        func(s *resolved.ResolvedStage) { s.Agents["other"] = s.Agents["checker"] },
		"wrong result MIME": func(s *resolved.ResolvedStage) {
			r := s.Result.Artifacts["result"]
			r.MediaTypes = []string{"application/zip"}
			s.Result.Artifacts["result"] = r
		},
	} {
		t.Run(name, func(t *testing.T) {
			copy := pin.Clone().Workflows["check"]
			stage := copy.Stages["check"]
			change(&stage)
			copy.Stages["check"] = stage
			if _, err := auditprofile.ResolveCompletionTarget(copy, binding); err == nil {
				t.Fatal("invalid assembler target accepted")
			}
		})
	}
}
