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
		{"likec4_builder@1", "instructions/likec4-builder-worker.md", "sha256:6e8d6b5837d97c310e930cd021df2e44acd2ea567f4880d1d20d3d6ad5f5948e", "sha256:f12c247c978e37c29ca02ce36222db777584c1e7a49bb8506eddd03f46d1a62d", false},
		{"likec4_builder@2", "instructions/likec4-builder-worker-v2.md", "sha256:8a7596958f4d6b1c2575505cc9f0b498a59368f497b442dbbfdb2f59ef402741", "sha256:55a719de2b4f6e59103f28a62f25c899dd4c099b9a5c45e6a40003eb059a6001", true},
		{"likec4_validator@1", "instructions/likec4-validator-worker.md", "sha256:3fdcc18d517775f4c9540274d35bb4d3faf2a25871dad363b7a12880f893196c", "sha256:a3ffcbb259405282f89a07e3a112c091cc825eb3af51291ecc2eb438070698dd", false},
		{"likec4_validator@2", "instructions/likec4-validator-worker-v2.md", "sha256:e00544ffa566de62c9a7db61af73a1044c5d8d9e1d988f8d6be42c3a3bf4219c", "sha256:b6d46576790b9504980c2822676733b86b97848e58b7523108fd441f12e1cb28", true},
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
		"instructions/likec4-builder-worker.md":        "f12c247c978e37c29ca02ce36222db777584c1e7a49bb8506eddd03f46d1a62d",
		"instructions/likec4-validator-worker.md":      "a3ffcbb259405282f89a07e3a112c091cc825eb3af51291ecc2eb438070698dd",
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
			"valid: true", "text/vnd.likec4", "storage revisions", "semantic result",
			"selected `likec4` Agent Skill", "references/...",
		},
		"instructions/likec4-validator-worker-v2.md": {
			"repair-only", "architecture_candidate", "validate_likec4", "bounded repair pass",
			"validation-report", "valid: true", "remaining DSL", "text/markdown",
			"storage revisions", "semantic result", "selected `likec4` Agent Skill",
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
