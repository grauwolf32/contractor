package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestAgentTemplateSkillsNormalizeAndPreserveEmptyDigest(t *testing.T) {
	baseline := mustLoad(t, copyConfigTree(t), MVPDescriptors())
	baselineTemplate, _ := baseline.AgentTemplate("artifact_builder@1")
	const fixtureWithoutSkillsDigest = "sha256:76879d45aef4c7b7900341ed7ff0b9de5ea9d7a41c6921d304a1c80a2bd2e9d0"
	if baselineTemplate.Ref.Digest != fixtureWithoutSkillsDigest {
		t.Fatalf("test fixture AgentTemplate digest changed: %s", baselineTemplate.Ref.Digest)
	}

	emptyRoot := copyConfigTree(t)
	addTemplateSkills(t, emptyRoot, "  skills: []\n")
	empty := mustLoad(t, emptyRoot, MVPDescriptors())
	emptyTemplate, _ := empty.AgentTemplate("artifact_builder@1")
	if baselineTemplate.Ref.Digest != emptyTemplate.Ref.Digest || len(emptyTemplate.Skills) != 0 {
		t.Fatalf("omitted/empty Skill normalization changed digest: %s != %s", baselineTemplate.Ref.Digest, emptyTemplate.Ref.Digest)
	}

	firstRoot := copyConfigTree(t)
	addTemplateSkills(t, firstRoot, "  skills:\n    - {namespace: skills, name: beta}\n    - {namespace: skills, name: alpha2}\n")
	secondRoot := copyConfigTree(t)
	addTemplateSkills(t, secondRoot, "  skills:\n    - {namespace: skills, name: alpha2}\n    - {namespace: skills, name: beta}\n")
	first := mustLoad(t, firstRoot, MVPDescriptors())
	second := mustLoad(t, secondRoot, MVPDescriptors())
	firstTemplate, _ := first.AgentTemplate("artifact_builder@1")
	secondTemplate, _ := second.AgentTemplate("artifact_builder@1")
	if firstTemplate.Ref.Digest != secondTemplate.Ref.Digest ||
		firstTemplate.Ref.Digest == baselineTemplate.Ref.Digest {
		t.Fatalf("Skill set digest normalization = baseline %s, first %s, second %s", baselineTemplate.Ref.Digest, firstTemplate.Ref.Digest, secondTemplate.Ref.Digest)
	}
	if got := []string{firstTemplate.Skills[0].Name, firstTemplate.Skills[1].Name}; got[0] != "alpha2" || got[1] != "beta" {
		t.Fatalf("normalized Skill order = %v", got)
	}
}

func TestAgentTemplateSkillDigestMatchesSharedPythonFixture(t *testing.T) {
	t.Parallel()

	encoded, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "v1alpha1", "valid", "allocation-spec-skills.json"))
	if err != nil {
		t.Fatal(err)
	}
	var allocation contracts.AllocationSpec
	if err := json.Unmarshal(encoded, &allocation); err != nil {
		t.Fatal(err)
	}
	digest, err := agentTemplateDigest(
		Selector{ID: allocation.AgentTemplate.Ref.TemplateID, Version: allocation.AgentTemplate.Ref.Version},
		allocation.AgentTemplate,
	)
	if err != nil {
		t.Fatal(err)
	}
	const expected = "sha256:f088d3a6c4b2ecd9e430da5f503db36d023cdb8791efec29468f91d7120293d6"
	if digest != expected {
		t.Fatalf("shared non-empty Skill AgentTemplate digest = %s, want %s", digest, expected)
	}
}

func TestAgentTemplateSkillRefsAreStrict(t *testing.T) {
	tests := []struct {
		name, yaml, want string
	}{
		{"wrong namespace", "  skills: [{namespace: other, name: review}]\n", "versionless skills/<portable-name>"},
		{"exact revision", "  skills: [{namespace: skills, name: review, revision: rev1}]\n", "versionless skills/<portable-name>"},
		{"invalid name", "  skills: [{namespace: skills, name: Review_Skill}]\n", "versionless skills/<portable-name>"},
		{"duplicate", "  skills: [{namespace: skills, name: review}, {namespace: skills, name: review}]\n", "duplicate ref"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := copyConfigTree(t)
			addTemplateSkills(t, root, test.yaml)
			if snapshot, err := Load(root, MVPDescriptors()); err == nil || snapshot != nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() = (%v, %v), want %q", snapshot, err, test.want)
			}
		})
	}

	root := copyConfigTree(t)
	var over strings.Builder
	over.WriteString("  skills:\n")
	for index := range contracts.MaxAgentTemplateSkills + 1 {
		fmt.Fprintf(&over, "    - {namespace: skills, name: skill%02d}\n", index)
	}
	addTemplateSkills(t, root, over.String())
	if snapshot, err := Load(root, MVPDescriptors()); err == nil || snapshot != nil ||
		!strings.Contains(err.Error(), "at most 32") {
		t.Fatalf("over-limit Load() = (%v, %v)", snapshot, err)
	}
}

func TestAgentTemplateSkillsReserveNativeToolsAndRequireToolBudget(t *testing.T) {
	t.Run("reserved collision", func(t *testing.T) {
		root := copyConfigTree(t)
		addTemplateSkills(t, root, "  skills: [{namespace: skills, name: review}]\n")
		path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
		replaceFile(t, path, "        - list_artifacts", "        - load_skill")
		descriptors := MVPDescriptors()
		descriptor := descriptors.Toolsets["run-artifacts@1"]
		descriptor.Tools = append(descriptor.Tools, "load_skill")
		descriptors.Toolsets["run-artifacts@1"] = descriptor
		if snapshot, err := Load(root, descriptors); err == nil || snapshot != nil ||
			!strings.Contains(err.Error(), "reserved by Agent Skills") {
			t.Fatalf("colliding Load() = (%v, %v)", snapshot, err)
		}
	})

	t.Run("Skill-only Worker", func(t *testing.T) {
		root := copyConfigTree(t)
		writeFile(t, filepath.Join(root, "model-policies/test-skill-worker.yaml"), []byte(`apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata: {name: test-skill-worker, version: "1"}
spec:
  model: worker-model
  maxOutputTokens: 4096
  maxModelCalls: 8
  maxTotalTokens: 32768
`))
		path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
		replaceFile(t, path, "  modelPolicy: test-worker@1", "  modelPolicy: test-skill-worker@1")
		replaceFile(t, path, `  toolsets:
    - ref: run-artifacts@1
      tools:
        - list_artifacts
        - read_artifact
        - write_artifact
`, "  toolsets: []\n")
		addTemplateSkills(t, root, "  skills: [{namespace: skills, name: review}]\n")
		if snapshot, err := Load(root, MVPDescriptors()); err == nil || snapshot != nil ||
			!strings.Contains(err.Error(), "tool-using Worker modelPolicy requires maxToolCalls") {
			t.Fatalf("unbudgeted Skill-only Load() = (%v, %v)", snapshot, err)
		}
	})
}

func TestWorkflowSkillUnionIncludesAllRetainedTemplatesAndIsBounded(t *testing.T) {
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	template, _ := snapshot.AgentTemplate("artifact_builder@1")
	workflow := ResolvedWorkflow{Stages: make(map[string]ResolvedStage)}
	for stageIndex := range 5 {
		variant := template
		variant.Skills = make([]contracts.ArtifactRef, contracts.MaxAgentTemplateSkills)
		for skillIndex := range contracts.MaxAgentTemplateSkills {
			variant.Skills[skillIndex] = contracts.ArtifactRef{
				Namespace: contracts.AgentSkillNamespace,
				Name:      fmt.Sprintf("s%03d", stageIndex*contracts.MaxAgentTemplateSkills+skillIndex),
			}
		}
		workflow.Stages[fmt.Sprintf("stage-%d", stageIndex)] = ResolvedStage{
			Agents: map[string]ResolvedAgentBinding{"worker": {Template: variant}},
		}
	}
	if refs, err := WorkflowSkillRefs(workflow); err == nil || refs != nil ||
		!strings.Contains(err.Error(), "at most 128") {
		t.Fatalf("WorkflowSkillRefs() = (%v, %v)", refs, err)
	}
}

func addTemplateSkills(t *testing.T, root, skillsYAML string) {
	t.Helper()
	path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
	replaceFile(t, path, "  sandboxProfile: local-workdir@1\n", skillsYAML+"  sandboxProfile: local-workdir@1\n")
}
