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
		{"likec4_builder@1", "instructions/likec4-builder-worker.md", "sha256:7411be7001436673cef6be9597122944feaf4c46f819418993d13109e3121b98", "sha256:957cce33bac8613b0f2da73cb52d9f3daef96fd08eb8d33e6ffa57a1c400c9b6", false},
		{"likec4_builder@2", "instructions/likec4-builder-worker-v2.md", "sha256:3991f22fe14b4f09c9066f9d35a0457f557571d5205eddc5dbd31c11677fea31", "sha256:c02e4cba7ee2a7adf4a80abc2f64d82cb98419dabbc041dc992462c7a7d99832", true},
		{"likec4_validator@1", "instructions/likec4-validator-worker.md", "sha256:fed1135b0589de82a360f379983446c2b8d10b65eb1de10ed65c5d2f0bb87ce8", "sha256:7ddc7c81657a49a07adeb0ff9e96c682d8da8927d8d664dedc5f9d67f8ed02f3", false},
		{"likec4_validator@2", "instructions/likec4-validator-worker-v2.md", "sha256:ab8352f1942d307f14a46c99166740bb853daa7ecd493cdadf045504c6c7e093", "sha256:fa1d61e80d968997afe86cea0b89f5a245772911832c31d4a4ced71b2f42be97", true},
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

func TestRepositoryLikeC4ConfigurationBytesRemainPinned(t *testing.T) {
	t.Parallel()

	expected := map[string]string{
		"agent-templates/likec4_builder.yaml":          "30a014217d0c037812fc4a2edbd6b625ea2c21e3f5716d342f8d50ae67514c0f",
		"agent-templates/likec4_validator.yaml":        "8cf2c3bc4db2568d43ecfdbf347d05316a186dea7f3103c15ada0686f62d7c97",
		"instructions/likec4-builder-worker.md":        "957cce33bac8613b0f2da73cb52d9f3daef96fd08eb8d33e6ffa57a1c400c9b6",
		"instructions/likec4-validator-worker.md":      "7ddc7c81657a49a07adeb0ff9e96c682d8da8927d8d664dedc5f9d67f8ed02f3",
		"workflows/likec4_from_source.yaml":            "b3fdf49c24c3d0454e2aeb6ec1377119a7999b371ad52e65e8aa62383dba4f1a",
		"workflows/likec4_from_source_v2.yaml":         "04047c41a75692390f78c9aeb079f7206e819bc925deb2334efdb1c60867ee7c",
		"workflows/likec4_from_source_streamline.yaml": "8c38bfb028a4386691221b78fc2221c5d1e25c689d340ebfaf790003f2a84a70",
		"workflows/likec4_from_analysis.yaml":          "80c63f54adead3519f8f26c96303bd28b1181a1bda24b70fcc189c152a3940c0",
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
			"named `source` input", "likec4/architecture", "load_likec4", "write_likec4",
			"specification", "model", "views", "relative/path:line", "validate_likec4",
			"valid: true", "text/vnd.likec4", "storage revisions", "plain-text summary",
			"selected `likec4` Agent Skill", "references/...",
		},
		"instructions/likec4-validator-worker-v2.md": {
			"repair-only", "architecture_candidate", "validate_likec4", "bounded repair pass",
			"validation-report", "valid: true", "remaining DSL", "text/markdown",
			"storage revisions", "plain-text summary", "selected `likec4` Agent Skill",
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
