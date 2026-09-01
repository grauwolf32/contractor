package config

import (
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

var caidoSkillOperation = regexp.MustCompile(`\b(?:caido|http)_[a-z][a-z_]*\b`)

func TestRepositoryHTTPAndCaidoConfigurationIsClosed(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	if err := validateRepositoryCaidoAssignment(repositoryConfigRoot, snapshot); err != nil {
		t.Fatal(err)
	}

	httpExplorer, err := snapshot.AgentTemplate("http_explorer@1")
	if err != nil {
		t.Fatal(err)
	}
	assertExactTemplateTools(t, httpExplorer, map[string][]string{
		"http-tools@1": {
			"http_history", "http_read_body", "http_request",
			"http_session_clear", "http_session_get", "http_session_set",
		},
		"text-artifacts@1": {"read_text_artifact", "write_text_artifact"},
	})

	caidoAnalyst, err := snapshot.AgentTemplate("caido_analyst@1")
	if err != nil {
		t.Fatal(err)
	}
	assertExactTemplateTools(t, caidoAnalyst, map[string][]string{
		"caido@1": {
			"caido_automate_results", "caido_automate_run", "caido_history",
			"caido_replay", "caido_request_detail", "caido_scope", "caido_sitemap",
			"caido_workflow_findings", "caido_workflow_list", "caido_workflow_run",
		},
		"http-tools@1":     {"http_history", "http_read_body", "http_request"},
		"text-artifacts@1": {"read_text_artifact", "write_text_artifact"},
	})
	if !slices.Equal(caidoAnalyst.Skills, []contracts.ArtifactRef{{
		Namespace: contracts.AgentSkillNamespace,
		Name:      "caido",
	}}) {
		t.Fatalf("caido_analyst@1 skills = %+v", caidoAnalyst.Skills)
	}

	workflow, err := snapshot.Workflow("security-analysis@1")
	if err != nil {
		t.Fatal(err)
	}
	if workflow.EntryStage != "analyze" || len(workflow.Stages) != 1 ||
		len(workflow.Parameters) != 3 || !workflow.Parameters["objective"].Required ||
		!workflow.Parameters["target"].Required || !workflow.Parameters["authorization_scope"].Required ||
		len(workflow.Inputs) != 1 || workflow.Inputs["context"].Required ||
		len(workflow.Outputs) != 1 || !workflow.Outputs["report"].Required {
		t.Fatalf("security-analysis@1 contract = %+v", workflow)
	}
	stage := workflow.Stages[workflow.EntryStage]
	analyst, ok := stage.Agents["analyst"]
	if !ok || analyst.Namespace != "security" || analyst.Template.Ref.TemplateID != "caido_analyst" ||
		stage.Planner.PlannerID != "passthrough" || stage.WorkflowOutputs["report"] != "report" ||
		stage.On.Failed.Kind != TransitionFail || stage.On.Interrupted.Kind != TransitionFail {
		t.Fatalf("security-analysis@1 analyze Stage = %+v", stage)
	}

	for _, relative := range []string{
		"agent-templates/caido_analyst.yaml", "agent-templates/http_explorer.yaml",
		"workflows/security_analysis.yaml", "instructions/caido-analyst-worker.md",
		"instructions/http-explorer-worker.md", "instructions/security-analysis-planner.md",
		"skills/caido/SKILL.md",
	} {
		contents, err := os.ReadFile(filepath.Join(repositoryConfigRoot, relative))
		if err != nil {
			t.Fatal(err)
		}
		lower := strings.ToLower(string(contents))
		for _, forbidden := range []string{
			"http://", "https://", "proxyurl:", "endpoint:", "credential:",
			"token:", "cabundlepem:", "run_skill_script", "scripts/",
		} {
			if strings.Contains(lower, forbidden) {
				t.Errorf("%s embeds forbidden infrastructure or script token %q", relative, forbidden)
			}
		}
	}
}

func TestRepositoryCaidoAssignmentRejectsMissingToolOrMalformedSkillRef(t *testing.T) {
	t.Run("missing named operation", func(t *testing.T) {
		root := copyConfigTree(t)
		path := filepath.Join(root, "agent-templates", "caido_analyst.yaml")
		replaceFile(t, path, "        - caido_replay\n", "")
		snapshot := mustLoad(t, root, MVPDescriptors())
		if err := validateRepositoryCaidoAssignment(root, snapshot); err == nil ||
			!strings.Contains(err.Error(), "caido_replay") {
			t.Fatalf("compatibility validation error = %v", err)
		}
	})

	t.Run("malformed Skill ref", func(t *testing.T) {
		root := copyConfigTree(t)
		path := filepath.Join(root, "agent-templates", "caido_analyst.yaml")
		replaceFile(t, path, "    - namespace: skills\n      name: caido\n", "    - namespace: plugins\n      name: caido\n")
		if snapshot, err := Load(root, MVPDescriptors()); err == nil || snapshot != nil ||
			!strings.Contains(err.Error(), "versionless skills/<portable-name>") {
			t.Fatalf("malformed Skill Load() = (%v, %v)", snapshot, err)
		}
	})
}

func validateRepositoryCaidoAssignment(root string, snapshot *Snapshot) error {
	template, err := snapshot.AgentTemplate("caido_analyst@1")
	if err != nil {
		return err
	}
	selected := make(map[string]bool)
	for _, toolset := range template.Toolsets {
		for _, tool := range toolset.Tools {
			selected[tool] = true
		}
	}
	contents, err := os.ReadFile(filepath.Join(root, "skills", "caido", "SKILL.md"))
	if err != nil {
		return err
	}
	for _, operation := range caidoSkillOperation.FindAllString(string(contents), -1) {
		if !selected[operation] {
			return &repositoryCompatibilityError{operation: operation}
		}
	}
	return nil
}

type repositoryCompatibilityError struct{ operation string }

func (e *repositoryCompatibilityError) Error() string {
	return "configs/skills/caido names operation " + e.operation + " omitted by caido_analyst@1"
}

func assertExactTemplateTools(t *testing.T, template contracts.ResolvedAgentTemplate, want map[string][]string) {
	t.Helper()
	got := make(map[string][]string, len(template.Toolsets))
	for _, selection := range template.Toolsets {
		ref := selection.Ref.ToolsetID + "@" + selection.Ref.Version
		got[ref] = append([]string(nil), selection.Tools...)
		sort.Strings(got[ref])
	}
	for ref := range want {
		sort.Strings(want[ref])
	}
	if len(got) != len(want) {
		t.Fatalf("%s@%s Toolsets = %v, want %v", template.Ref.TemplateID, template.Ref.Version, got, want)
	}
	for ref, operations := range want {
		if !slices.Equal(got[ref], operations) {
			t.Errorf("%s@%s %s tools = %v, want %v", template.Ref.TemplateID, template.Ref.Version, ref, got[ref], operations)
		}
	}
}
