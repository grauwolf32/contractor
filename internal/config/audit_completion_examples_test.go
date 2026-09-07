package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestVersionedAuditCompletionExamplePreservesLegacyClosureAndFailsChildRun(t *testing.T) {
	root := t.TempDir()
	if err := os.CopyFS(root, os.DirFS(filepath.Join("..", "..", "testdata", "configs"))); err != nil {
		t.Fatal(err)
	}
	legacy, err := mustLoad(t, root, MVPDescriptors()).AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"agent-templates/audit_source_checker_v4_completion.yaml",
		"workflows/audit_source_check_v4_completion.yaml",
		"audit-profiles/source_checklist_v3_completion.yaml",
		"model-policies/audit_completion_worker.yaml",
		"instructions/audit-source-checker-completion-worker.md",
	} {
		data, err := os.ReadFile(filepath.Join("..", "..", "configs", name))
		if err != nil {
			t.Fatal(err)
		}
		writeFile(t, filepath.Join(root, name), data)
	}
	snapshot := mustLoad(t, root, MVPDescriptors())
	old, err := snapshot.AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	profile, err := snapshot.AuditProfile("source-checklist@3")
	if err != nil {
		t.Fatal(err)
	}
	if legacy.Ref.Digest != old.Ref.Digest || old.Workflows["check"].WorkerCompletion != nil {
		t.Fatal("adding opt-in versions altered the legacy closure")
	}
	binding := profile.Workflows["check"]
	completion := binding.WorkerCompletion
	if completion == nil || completion.Kind != contracts.AuditCheckResultsV1 || completion.Stage != "check" || completion.Agent != "checker" {
		t.Fatal("example lost explicit completion authority")
	}
	stage := binding.Workflow.Stages["check"]
	if stage.On.Failed.Kind != TransitionFail || stage.On.Interrupted.Kind != TransitionFail {
		t.Fatal("example must end the child Run after failure/interruption")
	}
	worker := stage.Agents["checker"].Template
	if worker.Summarizer != nil || len(stage.Agents) != 1 || stage.Planner.PlannerID != "passthrough" {
		t.Fatal("example has an unsupported completion target")
	}
	selected := false
	for _, toolset := range worker.Toolsets {
		if toolset.Ref.ToolsetID == "audit-results" {
			selected = toolset.Ref.Version == "2"
		}
	}
	if !selected || !strings.Contains(worker.Instructions.Text, "expected_revision") || !strings.Contains(worker.Instructions.Text, "create-only") {
		t.Fatal("example lacks incremental submission/retry instructions")
	}
}
