package config

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRepositoryLikeC4SkillTemplateVersionBoundary(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	tests := []struct {
		selector, instructions, templateDigest, instructionsDigest string
		skilled                                                    bool
	}{
		{"likec4_builder@1", "instructions/likec4-builder-worker.md", "sha256:98ea7320f97888f7173550679b47c1a55ccb402a402d72cb3cf85c37b4c6d54f", "sha256:1bfa900469deeed3ec7f899a2810f9e67830c44dc099a094d2bde70b7a000fd2", false},
		{"likec4_builder@2", "instructions/likec4-builder-worker-v2.md", "sha256:992bf2afa60fa993221febd0d60c48cbe88540c64e8080926497a85942a6bab0", "sha256:896832d85381898e18ea1a13107da4e0c10e5445adf34d812ae2703eaef3b572", true},
		{"likec4_validator@1", "instructions/likec4-validator-worker.md", "sha256:16a04c1be667d227f3c2481d43efe42b13a6574867d87a37fd12e337584868ba", "sha256:c7cff50fd30a68553b98df8ea67d86bf2a102ecd6b6ecda69db3420b3cb28fb9", false},
		{"likec4_validator@2", "instructions/likec4-validator-worker-v2.md", "sha256:1562dc568020106ad9bf719dba0463fd1e4828242874d6d5f125d40bfbff2a04", "sha256:3c065642d16e473f9212d16041013dab906c36325d6d2119611ebfab837231cf", true},
	}
	for _, test := range tests {
		t.Run(test.selector, func(t *testing.T) {
			template, err := snapshot.AgentTemplate(test.selector)
			if err != nil {
				t.Fatal(err)
			}
			if template.Ref.Digest != test.templateDigest {
				t.Errorf("template digest = %s, want %s", template.Ref.Digest, test.templateDigest)
			}
			if template.Instructions.Ref != test.instructions ||
				template.Instructions.Digest != test.instructionsDigest {
				t.Errorf("instructions = %+v, want ref=%s digest=%s", template.Instructions, test.instructions, test.instructionsDigest)
			}
			wantSkills := []contracts.ArtifactRef(nil)
			if test.skilled {
				wantSkills = []contracts.ArtifactRef{{Namespace: contracts.AgentSkillNamespace, Name: "likec4"}}
			}
			if !reflect.DeepEqual(template.Skills, wantSkills) {
				t.Errorf("skills = %+v, want %+v", template.Skills, wantSkills)
			}
		})
	}

	allSelectors := []string{
		"artifact_builder@1", "likec4_builder@1", "likec4_builder@2",
		"likec4_validator@1", "likec4_validator@2", "openapi_builder@1",
		"openapi_validator@1", "shared_memory_reviewer@1",
		"shared_memory_worker@1", "source_analyst@1",
	}
	for _, selector := range allSelectors {
		template, err := snapshot.AgentTemplate(selector)
		if err != nil {
			t.Fatal(err)
		}
		wantSkill := selector == "likec4_builder@2" || selector == "likec4_validator@2"
		if (len(template.Skills) > 0) != wantSkill {
			t.Errorf("%s skills = %+v", selector, template.Skills)
		}
	}
}

func TestRepositoryLikeC4SkillWorkflowVersionsPreserveGraphs(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	pairs := []struct{ legacy, skilled string }{
		{"likec4-from-source@2", "likec4-from-source@3"},
		{"likec4-from-source-streamline@1", "likec4-from-source-streamline@2"},
		{"likec4-from-analysis@1", "likec4-from-analysis@2"},
	}
	for _, pair := range pairs {
		t.Run(pair.skilled, func(t *testing.T) {
			legacy, err := snapshot.Workflow(pair.legacy)
			if err != nil {
				t.Fatal(err)
			}
			skilled, err := snapshot.Workflow(pair.skilled)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(legacy.Parameters, skilled.Parameters) ||
				!reflect.DeepEqual(legacy.Inputs, skilled.Inputs) ||
				!reflect.DeepEqual(legacy.Outputs, skilled.Outputs) ||
				legacy.EntryStage != skilled.EntryStage || len(legacy.Stages) != len(skilled.Stages) {
				t.Fatal("additive Skill version drifted from its predecessor graph")
			}
			for name, legacyStage := range legacy.Stages {
				skilledStage, ok := skilled.Stages[name]
				if !ok {
					t.Fatalf("skilled Workflow omits Stage %q", name)
				}
				if legacyStage.Objective != skilledStage.Objective ||
					!reflect.DeepEqual(legacyStage.Instructions, skilledStage.Instructions) ||
					legacyStage.Planner != skilledStage.Planner ||
					!reflect.DeepEqual(legacyStage.ExecutionConfig, skilledStage.ExecutionConfig) ||
					!reflect.DeepEqual(legacyStage.Context, skilledStage.Context) ||
					!reflect.DeepEqual(legacyStage.Result, skilledStage.Result) ||
					!reflect.DeepEqual(legacyStage.WorkflowOutputs, skilledStage.WorkflowOutputs) ||
					!reflect.DeepEqual(legacyStage.On, skilledStage.On) ||
					len(legacyStage.Agents) != len(skilledStage.Agents) {
					t.Fatalf("Stage %q drifted outside its AgentTemplate version", name)
				}
				for logicalName, legacyAgent := range legacyStage.Agents {
					skilledAgent, ok := skilledStage.Agents[logicalName]
					if !ok || legacyAgent.Namespace != skilledAgent.Namespace {
						t.Fatalf("Stage %q Agent %q binding drifted", name, logicalName)
					}
					if name == "likec4_build" || name == "likec4_validate" {
						if legacyAgent.Template.Ref.Version != "1" || skilledAgent.Template.Ref.Version != "2" ||
							legacyAgent.Template.Ref.TemplateID != skilledAgent.Template.Ref.TemplateID ||
							len(legacyAgent.Template.Skills) != 0 || len(skilledAgent.Template.Skills) != 1 ||
							skilledAgent.Template.Skills[0] != (contracts.ArtifactRef{Namespace: "skills", Name: "likec4"}) {
							t.Fatalf("Stage %q template boundary = legacy %+v, skilled %+v", name, legacyAgent.Template, skilledAgent.Template)
						}
					} else if legacyAgent.Template.Ref != skilledAgent.Template.Ref {
						t.Fatalf("Stage %q unrelated template changed: %+v != %+v", name, legacyAgent.Template.Ref, skilledAgent.Template.Ref)
					}
				}
			}
		})
	}
}

func TestRepositoryLikeC4LegacyConfigurationBytesRemainPinned(t *testing.T) {
	t.Parallel()

	expected := map[string]string{
		"agent-templates/likec4_builder.yaml":          "30a014217d0c037812fc4a2edbd6b625ea2c21e3f5716d342f8d50ae67514c0f",
		"agent-templates/likec4_validator.yaml":        "8cf2c3bc4db2568d43ecfdbf347d05316a186dea7f3103c15ada0686f62d7c97",
		"instructions/likec4-builder-worker.md":        "1bfa900469deeed3ec7f899a2810f9e67830c44dc099a094d2bde70b7a000fd2",
		"instructions/likec4-validator-worker.md":      "c7cff50fd30a68553b98df8ea67d86bf2a102ecd6b6ecda69db3420b3cb28fb9",
		"workflows/likec4_from_source.yaml":            "bfd7c727caa0b92d30fdb311c8a9b79d809b0f29fc513a6ab54a84f79a66acdb",
		"workflows/likec4_from_source_v2.yaml":         "b8dfe7e4d092a1121f1b5a94daf66e41a4948e2f9c6141ffa442ddfb6bbdd5e8",
		"workflows/likec4_from_source_streamline.yaml": "9f7e70f911de30a89364e706ca1dc94a262ae1f566fd6bf1db5ba30db34ed777",
		"workflows/likec4_from_analysis.yaml":          "7eef6b0ca98e8be9807e2d92fbfac7d2256f619aaa79dccb3a1174fc3f478c09",
	}
	for relative, want := range expected {
		data, err := os.ReadFile(filepath.Join(repositoryConfigRoot, relative))
		if err != nil {
			t.Fatal(err)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != want {
			t.Errorf("%s sha256 = %s, want %s", relative, got, want)
		}
	}
}

func TestRepositoryLikeC4V2InstructionsKeepMandatoryProcedure(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	checks := map[string][]string{
		"instructions/likec4-builder-worker-v2.md": {
			"StageContentRequest", "artifacts.source", "load_likec4", "write_likec4",
			"specification", "model", "views", "relative/path:line", "validate_likec4",
			"valid: true", "text/vnd.likec4", "contractor/v1alpha1", "StageContentResult",
			"selected `likec4` Agent Skill", "references/...",
		},
		"instructions/likec4-validator-worker-v2.md": {
			"repair-only", "architecture_candidate", "validate_likec4", "bounded repair pass",
			"validation-report", "valid: true", "retryable", "text/markdown",
			"contractor/v1alpha1", "StageContentResult", "selected `likec4` Agent Skill",
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
	}
}
