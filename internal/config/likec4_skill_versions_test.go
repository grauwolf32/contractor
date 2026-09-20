package config

import (
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRepositoryCurrentLikeC4TemplatesSelectSkill(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	tests := []struct {
		selector     string
		instructions string
	}{
		{"likec4_builder@4", "instructions/likec4-builder-worker-v2-memory.md"},
		{"likec4_validator@4", "instructions/likec4-validator-worker-v2-memory.md"},
		{"workspace_likec4_builder@3", "instructions/workspace-likec4-builder-worker-memory.md"},
		{"workspace_likec4_validator@3", "instructions/workspace-likec4-validator-worker-memory.md"},
	}
	wantSkills := []contracts.ArtifactRef{{Namespace: contracts.AgentSkillNamespace, Name: "likec4"}}
	for _, test := range tests {
		t.Run(test.selector, func(t *testing.T) {
			template, err := snapshot.AgentTemplate(test.selector)
			if err != nil {
				t.Fatal(err)
			}
			assertDigest(t, template.Ref.Digest)
			assertDigest(t, template.Instructions.Digest)
			if template.Instructions.Ref != test.instructions {
				t.Errorf("instructions = %q, want %q", template.Instructions.Ref, test.instructions)
			}
			if !reflect.DeepEqual(template.Skills, wantSkills) {
				t.Errorf("skills = %+v, want %+v", template.Skills, wantSkills)
			}
		})
	}
}

func TestRepositoryCurrentLikeC4WorkflowsPinSkilledTemplates(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	workflows := []struct {
		selector          string
		buildStage        string
		validateStage     string
		builderTemplate   string
		validatorTemplate string
	}{
		{"likec4-from-analysis@5", "likec4_build", "likec4_validate", "likec4_builder@4", "likec4_validator@4"},
		{"likec4-from-workspace@7", "likec4_build", "likec4_validate", "workspace_likec4_builder@3", "workspace_likec4_validator@3"},
		{"likec4-from-workspace-streamline@4", "likec4_build", "likec4_validate", "workspace_likec4_builder@3", "workspace_likec4_validator@3"},
	}
	for _, test := range workflows {
		t.Run(test.selector, func(t *testing.T) {
			workflow, err := snapshot.Workflow(test.selector)
			if err != nil {
				t.Fatal(err)
			}
			for stageName, wantTemplate := range map[string]string{
				test.buildStage: test.builderTemplate, test.validateStage: test.validatorTemplate,
			} {
				stage := workflow.Stages[stageName]
				if len(stage.Agents) != 1 {
					t.Fatalf("Stage %s agents = %+v", stageName, stage.Agents)
				}
				for _, binding := range stage.Agents {
					got := binding.Template.Ref.TemplateID + "@" + binding.Template.Ref.Version
					if got != wantTemplate || len(binding.Template.Skills) != 1 ||
						binding.Template.Skills[0] != (contracts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "likec4"}) {
						t.Fatalf("Stage %s template = %s skills=%+v", stageName, got, binding.Template.Skills)
					}
				}
			}
		})
	}
}

func TestRepositoryCurrentLikeC4InstructionsKeepMandatoryProcedure(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	checks := map[string][]string{
		"instructions/likec4-builder-worker-v2-memory.md": {
			"named `source` input", "likec4/architecture", "load_likec4", "write_likec4",
			"specification", "model", "views", "relative/path:line", "validate_likec4",
			"valid: true", "selected `likec4` Agent Skill", "references/...",
		},
		"instructions/likec4-validator-worker-v2-memory.md": {
			"repair-only", "architecture_candidate", "validate_likec4", "bounded repair pass",
			"validation-report", "valid: true", "selected `likec4` Agent Skill",
		},
		"instructions/workspace-likec4-builder-worker-memory.md": {
			"private workspace root", "relative/path:line", "Cumulative workspace export is automatic",
		},
		"instructions/workspace-likec4-validator-worker-memory.md": {
			"cumulative workspace state", "repair-only", "Cumulative workspace export is automatic",
		},
	}
	for ref, fragments := range checks {
		instructions, err := snapshot.Instructions(ref)
		if err != nil {
			t.Fatal(err)
		}
		for _, fragment := range fragments {
			if !strings.Contains(instructions.Text, fragment) {
				t.Errorf("%s omits mandatory fragment %q", ref, fragment)
			}
		}
		for _, forbidden := range []string{"StageContentRequest", "StageContentResult", "contractor/v1alpha1", "retryable", "Runtime"} {
			if strings.Contains(instructions.Text, forbidden) {
				t.Errorf("%s leaks Runtime protocol fragment %q", ref, forbidden)
			}
		}
	}
}
