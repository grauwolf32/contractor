package config

import (
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRepositoryAuditCompletionProfilesPinCurrentWorkers(t *testing.T) {
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	for _, profile := range snapshot.AuditProfiles() {
		t.Run(profile.Ref.Name, func(t *testing.T) {
			if profile.Ref.Version != "1" {
				t.Fatal("bundled Audit profile must start at version 1")
			}
			for role, binding := range profile.Workflows {
				completion := binding.WorkerCompletion
				if completion == nil || completion.Kind != contracts.AuditCheckResultsV1 {
					t.Fatalf("%s lacks explicit completion authority", role)
				}
				if err := ValidateAuditWorkerCompletion(binding); err != nil {
					t.Fatal(err)
				}
				stage := binding.Workflow.Stages[completion.Stage]
				if stage.On.Failed.Kind != TransitionFail || stage.On.Interrupted.Kind != TransitionFail {
					t.Fatal("the child Run must end after failure/interruption")
				}
				worker := stage.Agents[completion.Agent].Template
				if binding.Workflow.Ref.Version != "1" || worker.Ref.Version != "1" {
					t.Fatal("bundled Audit Workflow and Worker versions must start at 1")
				}
				for _, instructions := range []string{worker.Instructions.Text, stage.Instructions.Text} {
					if !strings.Contains(instructions, "expected_revision") || !strings.Contains(instructions, "create-only") {
						t.Fatal("instructions lack incremental submission and publication retry rules")
					}
					if strings.Contains(instructions, "exactly once") || strings.Contains(instructions, "returns the exact artifact receipt") {
						t.Fatal("instructions still require legacy result publication")
					}
				}
			}
		})
	}
	for _, template := range snapshot.templates {
		for _, selection := range template.Toolsets {
			if selection.Ref.ToolsetID == "audit-results" && selection.Ref.Version != "2" {
				t.Fatalf("bundled template %s still selects old Audit tools", template.Ref.TemplateID)
			}
		}
	}
}
