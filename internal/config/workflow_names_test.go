package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// sharedNameCases reads the public ConfigId name grammar cases of
// api/testdata/v1alpha1/runtime-capability-cases.json, which Go, the schemas
// and the Runtime share, with the length bound turned into cases.
func sharedNameCases(t *testing.T) (valid, invalid []string) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "v1alpha1", "runtime-capability-cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases struct {
		ToolName struct {
			MaxLength int      `json:"maxLength"`
			Valid     []string `json:"valid"`
			Invalid   []string `json:"invalid"`
		} `json:"toolName"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if cases.ToolName.MaxLength == 0 || len(cases.ToolName.Valid) == 0 || len(cases.ToolName.Invalid) == 0 {
		t.Fatal("shared name cases are incomplete")
	}
	longest := strings.Repeat("a", cases.ToolName.MaxLength)
	return append([]string{longest}, cases.ToolName.Valid...), append([]string{longest + "a"}, cases.ToolName.Invalid...)
}

// yamlString quotes value as a YAML double-quoted scalar; JSON string escapes
// are valid YAML escapes.
func yamlString(t *testing.T, value string) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

// namesDocument is a one-Stage Workflow whose names a case replaces.
type namesDocument struct {
	entry, stage, agent string
	// next, when set, replaces the succeed action with a next Transition.
	next *string
	// extraStages and executionConfigStages are inserted verbatim.
	extraStages, executionConfigStages string
}

func (d namesDocument) yaml(t *testing.T) string {
	t.Helper()
	succeeded := "{succeed: {}}"
	if d.next != nil {
		succeeded = "{next: " + yamlString(t, *d.next) + "}"
	}
	return fmt.Sprintf(`apiVersion: contractor/v1alpha1
kind: Workflow
metadata:
  name: names
  version: "1"
spec:
  parameters: {}
  inputs:
    source: {required: true, mediaTypes: [text/plain]}
  outputs:
    result: {required: true, mediaTypes: [text/plain]}
  executionConfig:
    workers:
      llmGateway: local-litellm@1
      credential: development-worker
%s
  entryStage: %s
  stages:
%s    %s:
      objective: Copy the source artifact into the declared result artifact
      instructions: {ref: instructions/copy-planner.md}
      planner: passthrough@1
      agents:
        %s: {template: artifact_builder@1, namespace: builder}
      context:
        artifacts:
          source: {namespace: inputs, name: source, required: true}
      result:
        artifacts:
          copied: {required: true, mediaTypes: [text/plain], from: {namespace: builder, name: copied}}
      workflowOutputs: {result: copied}
      on:
        succeeded: %s
        failed: {fail: {}}
        interrupted: {fail: {}}
`, d.executionConfigStages, yamlString(t, d.entry), d.extraStages, yamlString(t, d.stage), yamlString(t, d.agent), succeeded)
}

func namesExecutionConfig(t *testing.T, agent string) string {
	t.Helper()
	return `apiVersion: contractor/v1alpha1
kind: ExecutionConfig
metadata:
  name: names
  version: "1"
spec:
  agents:
    ` + yamlString(t, agent) + `:
      modelPolicy: worker@1
`
}

func loadNames(t *testing.T, files map[string]string) (*Snapshot, error) {
	t.Helper()
	root := copyCoreFixture(t)
	for path, content := range files {
		writeFile(t, filepath.Join(root, path), []byte(content))
	}
	return Load(root, MVPDescriptors())
}

// TestWorkflowNamesUseTheOperationsNameGrammar is the regression for a
// logical Agent name such as "my worker" that loaded because the catalog only
// rejected blank names and slashes, while the public API and Operations show
// Stage and logical Agent names as ConfigId values: one such allocation failed
// the whole Operations allocation list (WFL-R84).
func TestWorkflowNamesUseTheOperationsNameGrammar(t *testing.T) {
	t.Parallel()
	valid, invalid := sharedNameCases(t)
	base := namesDocument{entry: "copy", stage: "copy", agent: "builder"}
	if _, err := loadNames(t, map[string]string{
		"workflows/names.yaml": base.yaml(t), "execution-configs/names.yaml": namesExecutionConfig(t, "builder"),
	}); err != nil {
		t.Fatalf("base names Workflow: %v", err)
	}

	for _, name := range valid {
		document := namesDocument{entry: name, stage: name, agent: name}
		snapshot, err := loadNames(t, map[string]string{
			"workflows/names.yaml": document.yaml(t), "execution-configs/names.yaml": namesExecutionConfig(t, name),
		})
		if err != nil {
			t.Errorf("name %q: Load error = %v", name, err)
			continue
		}
		workflow, err := snapshot.Workflow("names@1")
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := workflow.Stages[name].Agents[name]; !ok || workflow.EntryStage != name {
			t.Errorf("name %q was not kept as Stage and logical Agent name", name)
		}
	}

	for _, name := range invalid {
		quoted := yamlString(t, name)
		withStages := func(stages string) namesDocument {
			document := base
			document.executionConfigStages = "    stages:\n" + stages
			return document
		}
		for label, test := range map[string]struct {
			files map[string]string
			field string
		}{
			"Stage key": {map[string]string{"workflows/names.yaml": namesDocument{
				entry: "copy", stage: "copy", agent: "builder", extraStages: "    " + quoted + ": {}\n",
			}.yaml(t)}, "Stage name"},
			"entryStage": {map[string]string{"workflows/names.yaml": namesDocument{
				entry: name, stage: "copy", agent: "builder",
			}.yaml(t)}, "spec.entryStage"},
			"next target": {map[string]string{"workflows/names.yaml": namesDocument{
				entry: "copy", stage: "copy", agent: "builder", next: &name,
			}.yaml(t)}, "on.succeeded.next"},
			"agents key": {map[string]string{"workflows/names.yaml": namesDocument{
				entry: "copy", stage: "copy", agent: name,
			}.yaml(t)}, "logical Agent name"},
			"Workflow executionConfig Stage key": {map[string]string{"workflows/names.yaml": withStages(
				"      " + quoted + ":\n        agents:\n          builder: {modelPolicy: worker@1}\n",
			).yaml(t)}, "executionConfig Stage name"},
			"Workflow executionConfig Agent key": {map[string]string{"workflows/names.yaml": withStages(
				"      copy:\n        agents:\n          " + quoted + ": {modelPolicy: worker@1}\n",
			).yaml(t)}, "executionConfig logical Agent name"},
			"ExecutionConfig agents key": {map[string]string{
				"execution-configs/names.yaml": namesExecutionConfig(t, name),
			}, "spec logical Agent name"},
		} {
			snapshot, err := loadNames(t, test.files)
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.field) {
				t.Errorf("%s %q: Load() = (%v, %v), want a %s error", label, name, snapshot != nil, err, test.field)
			}
		}
	}
}

// TestPersistedSnapshotsRejectNamesOperationsCannotShow holds the strict
// snapshot readers and the graph check to the same grammar, so a persisted Run
// cannot carry a Stage or logical Agent name the catalog would reject.
func TestPersistedSnapshotsRejectNamesOperationsCannotShow(t *testing.T) {
	t.Parallel()
	snapshot := mustLoad(t, copyCoreFixture(t), MVPDescriptors())
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateWorkflowGraph(workflow); err != nil {
		t.Fatal(err)
	}
	_, invalid := sharedNameCases(t)
	for _, name := range invalid {
		renamedAgent := cloneWorkflow(workflow)
		stage := renamedAgent.Stages["copy"]
		stage.Agents[name] = stage.Agents["builder"]
		delete(stage.Agents, "builder")
		stage.ExecutionConfig.Agents[name] = stage.ExecutionConfig.Agents["builder"]
		delete(stage.ExecutionConfig.Agents, "builder")
		renamedAgent.Stages["copy"] = stage

		renamedStage := cloneWorkflow(workflow)
		renamedStage.Stages[name] = renamedStage.Stages["copy"]
		delete(renamedStage.Stages, "copy")
		renamedStage.EntryStage = name

		for label, candidate := range map[string]ResolvedWorkflow{"logical Agent name": renamedAgent, "Stage name": renamedStage} {
			if err := ValidateWorkflowGraph(candidate); err == nil || !strings.Contains(err.Error(), label) {
				t.Errorf("ValidateWorkflowGraph(%s %q) error = %v", label, name, err)
			}
			encoded, err := json.Marshal(candidate)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := DecodeResolvedWorkflowSnapshot(encoded); err == nil || !strings.Contains(err.Error(), label) {
				t.Errorf("DecodeResolvedWorkflowSnapshot(%s %q) error = %v", label, name, err)
			}
		}
		encodedStage, err := json.Marshal(renamedAgent.Stages["copy"])
		if err != nil {
			t.Fatal(err)
		}
		if _, err := DecodeResolvedStageSnapshot(encodedStage); err == nil || !strings.Contains(err.Error(), "logical Agent name") {
			t.Errorf("DecodeResolvedStageSnapshot(logical Agent name %q) error = %v", name, err)
		}
	}
}
