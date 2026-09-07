package config

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRepositoryTraceAnnotationTemplateIsExactAndImmutable(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	template, err := snapshot.AgentTemplate("workspace_taint_analyst@1")
	if err != nil {
		t.Fatal(err)
	}
	if template.Ref.Digest != "sha256:ffcec4a7b2f921e71def4e21b75c6837a2828c333d84a2da616b26c9c41429ff" ||
		template.Instructions.Digest != "sha256:dda89b2bace334ce4cf08f99bb2a2ba428d62d61ad9f51ac78ee43b358f1f0bf" {
		t.Fatalf(
			"immutable trace template snapshot = template %s instructions %s",
			template.Ref.Digest,
			template.Instructions.Digest,
		)
	}
	assertExactTemplateTools(t, template, map[string][]string{
		"code-analysis@1": {
			"attack_surface", "complexity_hotspots", "entrypoint_paths_to",
			"find_callees", "find_callers", "find_symbol", "functions_that_raise",
			"graph_summary", "list_symbols", "paths_between", "search_def",
		},
		"filesystem@1":        {"glob", "grep", "ls", "read_file"},
		"taint-annotations@1": {"annotate_sink", "annotate_trace", "annotate_validate"},
		"text-artifacts@1":    {"read_text_artifact", "write_text_artifact"},
		"workspace-changes@1": {"changed_paths", "diff", "rollback_changes"},
	})
	if !slices.Equal(template.Skills, []contracts.ArtifactRef{{
		Namespace: contracts.AgentSkillNamespace,
		Name:      "trace",
	}}) {
		t.Fatalf("workspace_taint_analyst@1 skills = %+v", template.Skills)
	}
	if template.Runtime != (contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}) ||
		template.SandboxProfile != (contracts.SandboxProfileRef{
			SandboxProfileID: "local-workdir", Version: "1",
		}) {
		t.Fatalf("workspace_taint_analyst@1 execution boundary = %+v", template)
	}

	lower := strings.ToLower(template.Instructions.Text)
	for _, forbidden := range []string{
		"agenttemplate", "allocationspec", "runtime", "host path", "llm gateway",
		"credential", "scheduler", "toolset", "sandbox",
	} {
		if strings.Contains(lower, forbidden) {
			t.Errorf("Worker instructions expose private concept %q", forbidden)
		}
	}
	for _, required := range []string{
		"load_skill", "graph_summary", "annotate_trace", "annotate_validate",
		"annotate_sink", "changed_paths", "diff", "rollback_changes",
		"read_text_artifact", "write_text_artifact", "analysis/report",
	} {
		if !strings.Contains(template.Instructions.Text, required) {
			t.Errorf("Worker instructions omit %q", required)
		}
	}
	if err := validateRepositoryTraceAnnotationAssignment(repositoryConfigRoot, template); err != nil {
		t.Fatal(err)
	}
}

func TestRepositoryTraceAnnotationWorkflowOwnsOverlayResults(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	workflow, err := snapshot.Workflow("taint-trace-from-workspace@2")
	if err != nil {
		t.Fatal(err)
	}
	if workflow.EntryStage != "trace" || len(workflow.Stages) != 1 ||
		len(workflow.Parameters) != 3 || !workflow.Parameters["target"].Required ||
		workflow.Parameters["objective"].Required || workflow.Parameters["context"].Required {
		t.Fatalf("taint trace top-level parameters = %+v", workflow)
	}
	if !reflect.DeepEqual(workflow.Inputs, map[string]ArtifactSlot{
		"source": {Required: true, MediaTypes: []string{"application/zip"}},
	}) || !reflect.DeepEqual(workflow.Outputs, map[string]ArtifactSlot{
		"taint_report":    {Required: true, MediaTypes: []string{"text/markdown"}, Primary: true},
		"workspace_diff":  {Required: true, MediaTypes: []string{"text/x-diff"}},
		"workspace_state": {Required: true, MediaTypes: []string{"application/vnd.contractor.workspace-overlay+json"}},
	}) {
		t.Fatalf("taint trace artifact contract = inputs %+v outputs %+v", workflow.Inputs, workflow.Outputs)
	}

	stage := workflow.Stages[workflow.EntryStage]
	if stage.Instructions.Digest != "sha256:1109b59750ef54689308646abe1df6f08a092e00e304e7a25e4463ba151dc096" ||
		stage.Planner != (PlannerRef{PlannerID: "passthrough", Version: "1"}) ||
		len(stage.Agents) != 1 || stage.Agents["analyst"].Namespace != "analysis" ||
		stage.Agents["analyst"].Template.Ref.TemplateID != "workspace_taint_analyst" ||
		stage.Agents["analyst"].Template.Ref.Version != "1" {
		t.Fatalf("taint trace Stage refs = %+v", stage)
	}
	wantContext := StageContext{
		Artifacts: map[string]ContextArtifact{
			"source": {Namespace: "inputs", Name: "source", Required: true},
		},
		Workspace: &WorkspaceContext{
			Mode:    contracts.WorkspaceModeOverlay,
			Sources: []WorkspaceSource{{Artifact: "source", Target: ""}},
			Export:  &WorkspaceExport{State: "workspace_state", Diff: "workspace_diff"},
		},
	}
	if !reflect.DeepEqual(stage.Context, wantContext) {
		t.Fatalf("taint trace workspace context = %+v, want %+v", stage.Context, wantContext)
	}
	wantResult := StageResultContract{Artifacts: map[string]ArtifactSlot{
		"report": {
			Required: true, MediaTypes: []string{"text/markdown"},
			From: &ArtifactBinding{Namespace: "analysis", Name: "report"},
		},
		"workspace_diff":  {Required: true, MediaTypes: []string{"text/x-diff"}},
		"workspace_state": {Required: true, MediaTypes: []string{"application/vnd.contractor.workspace-overlay+json"}},
	}}
	if !reflect.DeepEqual(stage.Result, wantResult) || !reflect.DeepEqual(
		stage.WorkflowOutputs,
		map[string]string{
			"taint_report": "report", "workspace_diff": "workspace_diff", "workspace_state": "workspace_state",
		},
	) {
		t.Fatalf("taint trace results = %+v outputs %+v", stage.Result, stage.WorkflowOutputs)
	}
	for name, transition := range map[string]TransitionAction{
		"failed": stage.On.Failed, "interrupted": stage.On.Interrupted,
	} {
		if transition.Kind != TransitionRetry || transition.Retry == nil ||
			transition.Retry.MaxAttempts != 2 || transition.Retry.Then.Kind != TransitionFail {
			t.Errorf("%s transition = %+v", name, transition)
		}
	}
	if stage.On.Succeeded.Kind != TransitionSucceed {
		t.Fatalf("succeeded transition = %+v", stage.On.Succeeded)
	}
}

func TestRepositoryTraceAnnotationAssignmentRejectsToolDrift(t *testing.T) {
	root := copyConfigTree(t)
	path := filepath.Join(root, "agent-templates", "workspace_taint_analyst.yaml")
	replaceFile(t, path, "      tools: [annotate_trace, annotate_validate, annotate_sink]", "      tools: [annotate_trace, annotate_validate]")
	snapshot := mustLoad(t, root, MVPDescriptors())
	template, err := snapshot.AgentTemplate("workspace_taint_analyst@1")
	if err != nil {
		t.Fatal(err)
	}
	if err := validateRepositoryTraceAnnotationAssignment(root, template); err == nil ||
		!strings.Contains(err.Error(), "annotate_sink") {
		t.Fatalf("trace assignment drift error = %v", err)
	}
}

func validateRepositoryTraceAnnotationAssignment(
	root string,
	template contracts.ResolvedAgentTemplate,
) error {
	selected := make(map[string]bool)
	for _, toolset := range template.Toolsets {
		if toolset.Ref.ToolsetID == "edit-files" {
			return fmt.Errorf("trace template must not select edit-files@1")
		}
		for _, operation := range toolset.Tools {
			selected[operation] = true
		}
	}
	for _, relative := range []string{
		"skills/trace/SKILL.md",
		"skills/trace/references/annotations.md",
		"instructions/workspace-taint-analyst-worker.md",
	} {
		contents, err := os.ReadFile(filepath.Join(root, relative))
		if err != nil {
			return err
		}
		for _, operation := range []string{
			"annotate_trace", "annotate_validate", "annotate_sink",
			"changed_paths", "diff", "rollback_changes",
		} {
			if strings.Contains(string(contents), operation) && !selected[operation] {
				return fmt.Errorf("%s names unselected operation %s", relative, operation)
			}
		}
	}
	return nil
}
