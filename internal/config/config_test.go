package config

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const repositoryConfigRoot = "../../configs"

func TestLoadRepositoryConfig(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	if got, want := snapshot.Counts(), (Counts{Workflows: 1, AgentTemplates: 1, ModelPolicies: 1, Instructions: 2}); got != want {
		t.Fatalf("Counts() = %+v, want %+v", got, want)
	}

	policy, err := snapshot.ModelPolicy("worker@1")
	if err != nil {
		t.Fatalf("resolve ModelPolicy: %v", err)
	}
	assertDigest(t, policy.Ref.Digest)
	if policy.Model != "worker-model" || policy.MaxOutputTokens != 4096 || policy.Temperature == nil || *policy.Temperature != 0.1 {
		t.Fatalf("unexpected resolved ModelPolicy: %+v", policy)
	}

	template, err := snapshot.AgentTemplate("artifact_builder@1")
	if err != nil {
		t.Fatalf("resolve AgentTemplate: %v", err)
	}
	assertDigest(t, template.Ref.Digest)
	assertDigest(t, template.Instructions.Digest)
	if template.ModelPolicy.Ref.Digest != policy.Ref.Digest {
		t.Fatalf("template policy digest = %q, want %q", template.ModelPolicy.Ref.Digest, policy.Ref.Digest)
	}
	if got, want := template.Toolsets[0].Tools, []string{"list_artifacts", "read_artifact", "write_artifact"}; !equalStrings(got, want) {
		t.Fatalf("selected tools = %v, want %v", got, want)
	}

	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatalf("resolve Workflow: %v", err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	if stage.Planner != (PlannerRef{PlannerID: "passthrough", Version: "1"}) {
		t.Fatalf("planner = %+v", stage.Planner)
	}
	if stage.Agents["builder"].Template.Ref != template.Ref {
		t.Fatalf("Workflow did not pin exact AgentTemplate: %+v", stage.Agents["builder"].Template.Ref)
	}
	if stage.WorkflowOutputs["result"] != "copied" || stage.On.Succeeded.Kind != TransitionSucceed {
		t.Fatalf("unexpected resolved Stage: %+v", stage)
	}

	instructions, err := snapshot.Instructions("instructions/copy-planner.md")
	if err != nil {
		t.Fatalf("resolve instructions: %v", err)
	}
	if instructions != stage.Instructions {
		t.Fatalf("resolved instructions differ: %+v != %+v", instructions, stage.Instructions)
	}
}

func TestWorkflowExamplesLoad(t *testing.T) {
	for _, name := range []string{"bounded_retry_workflow.yaml", "multi_stage_workflow.yaml"} {
		t.Run(name, func(t *testing.T) {
			root := copyConfigTree(t)
			example := readFile(t, filepath.Join(repositoryConfigRoot, "examples", name))
			writeFile(t, filepath.Join(root, "workflows", name), example)
			snapshot := mustLoad(t, root, MVPDescriptors())
			if snapshot.Counts().Workflows != 2 {
				t.Fatalf("example Workflow count = %d", snapshot.Counts().Workflows)
			}
		})
	}
}

func TestStoredFixtures(t *testing.T) {
	t.Parallel()

	valid := mustLoad(t, "testdata/valid", MVPDescriptors())
	if got, want := valid.Counts(), (Counts{Workflows: 1, AgentTemplates: 1, ModelPolicies: 1, Instructions: 2}); got != want {
		t.Fatalf("valid fixture Counts() = %+v, want %+v", got, want)
	}

	invalid := []string{
		"unknown-field-policy.yaml",
		"duplicate-key-policy.yaml",
		"multiple-documents-policy.yaml",
		"wrong-kind-policy.yaml",
	}
	for _, name := range invalid {
		t.Run(name, func(t *testing.T) {
			root := copyConfigTree(t)
			fixture := readFile(t, filepath.Join("testdata/invalid", name))
			writeFile(t, filepath.Join(root, "model-policies/worker.yaml"), fixture)
			snapshot, err := Load(root, MVPDescriptors())
			if err == nil || snapshot != nil {
				t.Fatalf("Load() = (%v, %v), want (nil, error)", snapshot, err)
			}
		})
	}
}

func TestStrictManifestFailuresReturnNoSnapshot(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*testing.T, string)
		want   []string
	}{
		{
			name: "unknown field",
			mutate: func(t *testing.T, root string) {
				appendFile(t, filepath.Join(root, "model-policies/worker.yaml"), "unknownField: true\n")
			},
			want: []string{"model-policies/worker.yaml", "unknownField"},
		},
		{
			name: "duplicate mapping key",
			mutate: func(t *testing.T, root string) {
				appendFile(t, filepath.Join(root, "model-policies/worker.yaml"), "kind: ModelPolicy\n")
			},
			want: []string{"model-policies/worker.yaml", "already defined"},
		},
		{
			name: "multiple documents",
			mutate: func(t *testing.T, root string) {
				appendFile(t, filepath.Join(root, "model-policies/worker.yaml"), `---
apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata: {name: other, version: "1"}
spec: {model: other, maxOutputTokens: 1}
`)
			},
			want: []string{"model-policies/worker.yaml", "exactly one"},
		},
		{
			name: "empty document",
			mutate: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, "model-policies/worker.yaml"), nil)
			},
			want: []string{"model-policies/worker.yaml", "exactly one"},
		},
		{
			name: "wrong subtree kind",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "model-policies/worker.yaml"), "kind: ModelPolicy", "kind: Workflow")
			},
			want: []string{"model-policies/worker.yaml", "does not match ModelPolicy subtree"},
		},
		{
			name: "duplicate identity",
			mutate: func(t *testing.T, root string) {
				source := readFile(t, filepath.Join(root, "model-policies/worker.yaml"))
				writeFile(t, filepath.Join(root, "model-policies/z/duplicate.yaml"), source)
			},
			want: []string{"model-policies/z/duplicate.yaml", "duplicate ModelPolicy identity worker@1"},
		},
		{
			name: "unknown runtime",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "agent-templates/artifact_builder.yaml"), "runtime: adk@1", "runtime: missing@1")
			},
			want: []string{"agent-templates/artifact_builder.yaml", "unknown WorkerRuntime"},
		},
		{
			name: "unknown tool",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "agent-templates/artifact_builder.yaml"), "read_artifact", "delete_artifact")
			},
			want: []string{"agent-templates/artifact_builder.yaml", "does not export selected tool"},
		},
		{
			name: "unknown planner",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), "planner: passthrough@1", "planner: missing@1")
			},
			want: []string{"workflows/artifact_copy.yaml", "unknown PlannerFactory"},
		},
		{
			name: "missing required boolean",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), "    source:\n      required: true\n      mediaTypes", "    source:\n      mediaTypes")
			},
			want: []string{"workflows/artifact_copy.yaml", "spec.inputs.source.required is required"},
		},
		{
			name: "incompatible output mapping",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), "  outputs:\n    result:\n      required: true\n      mediaTypes: [text/plain]", "  outputs:\n    result:\n      required: true\n      mediaTypes: [application/json]")
			},
			want: []string{"workflows/artifact_copy.yaml", "incompatible"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			root := copyConfigTree(t)
			test.mutate(t, root)
			snapshot, err := Load(root, MVPDescriptors())
			if err == nil || snapshot != nil {
				t.Fatalf("Load() = (%v, %v), want (nil, error)", snapshot, err)
			}
			for _, expected := range test.want {
				if !strings.Contains(err.Error(), expected) {
					t.Fatalf("error %q does not contain %q", err, expected)
				}
			}
		})
	}
}

func TestInstructionPathAndFilesystemContainment(t *testing.T) {
	t.Parallel()

	invalid := []string{
		"", "/instructions/a.md", "https://example.test/a", `instructions\a.md`,
		"instructions//a.md", "instructions/./a.md", "instructions/../a.md", "instructions",
	}
	for _, value := range invalid {
		if _, err := validateInstructionRef(value); err == nil {
			t.Errorf("validateInstructionRef(%q) succeeded", value)
		}
	}
	if got, err := validateInstructionRef("instructions/team/a.md"); err != nil || got != "instructions/team/a.md" {
		t.Fatalf("valid instruction ref = (%q, %v)", got, err)
	}

	t.Run("symlink escape", func(t *testing.T) {
		root := copyConfigTree(t)
		outside := filepath.Join(filepath.Dir(root), "outside.md")
		writeFile(t, outside, []byte("outside\n"))
		target := filepath.Join(root, "instructions/artifact-builder.md")
		if err := os.Remove(target); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, target); err != nil {
			t.Fatal(err)
		}
		snapshot, err := Load(root, MVPDescriptors())
		if err == nil || snapshot != nil || !strings.Contains(err.Error(), "escapes configuration root") {
			t.Fatalf("Load() = (%v, %v), want symlink escape error", snapshot, err)
		}
	})

	t.Run("invalid utf8", func(t *testing.T) {
		root := copyConfigTree(t)
		writeFile(t, filepath.Join(root, "instructions/artifact-builder.md"), []byte{0xff, 0xfe})
		snapshot, err := Load(root, MVPDescriptors())
		if err == nil || snapshot != nil || !strings.Contains(err.Error(), "strict UTF-8") {
			t.Fatalf("Load() = (%v, %v), want UTF-8 error", snapshot, err)
		}
	})
}

func TestManifestDiscoveryIgnoresSymlinks(t *testing.T) {
	t.Parallel()

	root := copyConfigTree(t)
	if err := os.Symlink("worker.yaml", filepath.Join(root, "model-policies/link.yaml")); err != nil {
		t.Fatal(err)
	}
	snapshot := mustLoad(t, root, MVPDescriptors())
	if snapshot.Counts().ModelPolicies != 1 {
		t.Fatalf("ModelPolicies = %d, want 1", snapshot.Counts().ModelPolicies)
	}
}

func TestToolsetVisibleNameCollision(t *testing.T) {
	t.Parallel()

	root := copyConfigTree(t)
	path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
	replaceFile(t, path, "  sandboxProfile: local-workdir@1", `    - ref: alternate@1
      tools: [read_artifact]
  sandboxProfile: local-workdir@1`)
	descriptors := MVPDescriptors()
	descriptors.Toolsets["alternate@1"] = ToolsetDescriptor{Tools: []string{"read_artifact"}}
	snapshot, err := Load(root, descriptors)
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "collides between Toolsets") {
		t.Fatalf("Load() = (%v, %v), want collision error", snapshot, err)
	}
}

func TestWorkflowGraphLoadsMultiStageAndBoundedRetry(t *testing.T) {
	t.Parallel()
	root := copyConfigTree(t)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(multiStageWorkflowYAML))

	snapshot := mustLoad(t, root, MVPDescriptors())
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	retry := workflow.Stages["build"].On.Failed.Retry
	if len(workflow.Stages) != 2 || retry == nil || retry.MaxAttempts != 3 ||
		retry.Then.Kind != TransitionNext || retry.Then.NextStage != "review" {
		t.Fatalf("resolved multi-Stage Workflow = %+v", workflow)
	}
}

func TestWorkflowGraphRejectsNextCycle(t *testing.T) {
	t.Parallel()
	root := copyConfigTree(t)
	cyclic := strings.Replace(
		multiStageWorkflowYAML,
		"    review:\n"+reviewStageYAML,
		"    review:\n"+strings.Replace(reviewStageYAML, "succeed: {}", "next: build", 1),
		1,
	)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(cyclic))

	snapshot, err := Load(root, MVPDescriptors())
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "Cycle") {
		t.Fatalf("Load(cycle) = (%v, %v)", snapshot, err)
	}
}

func TestWorkflowGraphRejectsUnreachableStage(t *testing.T) {
	t.Parallel()
	root := copyConfigTree(t)
	workflow := multiStageWorkflowYAML + "    orphan:\n" + reviewStageYAML
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(workflow))

	snapshot, err := Load(root, MVPDescriptors())
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "unreachable") {
		t.Fatalf("Load(unreachable) = (%v, %v)", snapshot, err)
	}
}

func TestWorkflowTransitionRejectsSuccessPathWithoutRequiredOutput(t *testing.T) {
	t.Parallel()
	root := copyConfigTree(t)
	workflow := strings.Replace(
		multiStageWorkflowYAML,
		"      workflowOutputs:\n        result: copied\n",
		"",
		1,
	)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(workflow))

	snapshot, err := Load(root, MVPDescriptors())
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "without required output") {
		t.Fatalf("Load(missing output path) = (%v, %v)", snapshot, err)
	}
}

func TestWorkflowTransitionRejectsOptionalResultAsRequiredOutput(t *testing.T) {
	t.Parallel()

	root := copyConfigTree(t)
	workflow := strings.Replace(
		multiStageWorkflowYAML,
		"          copied: {required: true, mediaTypes: [text/plain]}\n      workflowOutputs:",
		"          copied: {required: false, mediaTypes: [text/plain]}\n      workflowOutputs:",
		1,
	)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(workflow))
	if snapshot, err := Load(root, MVPDescriptors()); err == nil || snapshot != nil ||
		!strings.Contains(err.Error(), "without required output") {
		t.Fatalf("Load() = (%v, %v), want required output error", snapshot, err)
	}
}

const multiStageWorkflowYAML = `apiVersion: contractor/v1alpha1
kind: Workflow
metadata:
  name: artifact-copy
  version: "1"
spec:
  parameters: {}
  inputs:
    source: {required: true, mediaTypes: [text/plain]}
  outputs:
    result: {required: true, mediaTypes: [text/plain]}
  entryStage: build
  stages:
    build:
      objective: Build a candidate
      instructions: {ref: instructions/copy-planner.md}
      planner: passthrough@1
      agents:
        builder: {template: artifact_builder@1}
      context:
        artifacts:
          source: {namespace: inputs, name: source, required: true}
      result:
        artifacts:
          copied: {required: true, mediaTypes: [text/plain]}
      on:
        succeeded: {next: review}
        failed:
          retry:
            maxAttempts: 3
            then: {next: review}
        interrupted: {fail: {}}
    review:
` + reviewStageYAML

const reviewStageYAML = `      objective: Review the candidate
      instructions: {ref: instructions/copy-planner.md}
      planner: passthrough@1
      agents:
        reviewer: {template: artifact_builder@1, namespace: builder}
      context:
        artifacts:
          candidate: {namespace: builder, name: copied, required: true}
      result:
        artifacts:
          copied: {required: true, mediaTypes: [text/plain]}
      workflowOutputs:
        result: copied
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`

func mustLoad(t *testing.T, root string, descriptors Descriptors) *Snapshot {
	t.Helper()
	snapshot, err := Load(root, descriptors)
	if err != nil {
		t.Fatalf("Load(%q): %v", root, err)
	}
	return snapshot
}

func assertDigest(t *testing.T, value string) {
	t.Helper()
	if len(value) != len("sha256:")+64 || !strings.HasPrefix(value, "sha256:") {
		t.Fatalf("invalid digest %q", value)
	}
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func copyConfigTree(t *testing.T) string {
	t.Helper()
	destination := filepath.Join(t.TempDir(), "configs")
	err := filepath.WalkDir(repositoryConfigRoot, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(repositoryConfigRoot, path)
		if err != nil {
			return err
		}
		target := filepath.Join(destination, relative)
		if entry.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o644)
	})
	if err != nil {
		t.Fatalf("copy config tree: %v", err)
	}
	return destination
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func appendFile(t *testing.T, path, suffix string) {
	t.Helper()
	data := append(readFile(t, path), []byte(suffix)...)
	writeFile(t, path, data)
}

func replaceFile(t *testing.T, path, old, replacement string) {
	t.Helper()
	data := string(readFile(t, path))
	if !strings.Contains(data, old) {
		t.Fatalf("%s does not contain replacement source %q", path, old)
	}
	writeFile(t, path, []byte(strings.Replace(data, old, replacement, 1)))
}
