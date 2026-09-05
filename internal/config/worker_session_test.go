package config

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestWorkflowStageSessionAuthoringIsStrictAndDefaultsToIsolated(t *testing.T) {
	t.Parallel()

	baseline := mustLoad(t, filepath.Join("testdata", "valid"), MVPDescriptors())
	workflow, err := baseline.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	if got := workflow.Stages["copy"].Session; got != contracts.WorkerSessionIsolated {
		t.Fatalf("omitted Stage session = %q, want isolated", got)
	}
	encodedWorkflow, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	encodedStage, err := json.Marshal(workflow.Stages["copy"])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encodedWorkflow), `"session":"isolated"`) ||
		!strings.Contains(string(encodedStage), `"session":"isolated"`) {
		t.Fatalf("resolved snapshots omitted explicit session: workflow=%s stage=%s", encodedWorkflow, encodedStage)
	}

	for _, mode := range []contracts.WorkerSessionMode{
		contracts.WorkerSessionIsolated,
		contracts.WorkerSessionShared,
	} {
		mode := mode
		t.Run(string(mode), func(t *testing.T) {
			root := copyConfigTree(t)
			setWorkflowSession(t, root, string(mode))
			snapshot := mustLoad(t, root, MVPDescriptors())
			resolved, err := snapshot.Workflow("artifact-copy@1")
			if err != nil {
				t.Fatal(err)
			}
			if got := resolved.Stages["copy"].Session; got != mode {
				t.Fatalf("Stage session = %q, want %q", got, mode)
			}
		})
	}
}

func TestWorkflowStageSessionRejectsEveryNonEnumAuthoringForm(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name string
		raw  string
	}{
		{name: "null", raw: "null"},
		{name: "empty", raw: `""`},
		{name: "unknown", raw: "reset"},
		{name: "number", raw: "17"},
		{name: "mapping", raw: "{}"},
		{name: "sequence", raw: "[]"},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			root := copyConfigTree(t)
			setWorkflowSession(t, root, test.raw)
			if snapshot, err := Load(root, MVPDescriptors()); err == nil || snapshot != nil ||
				!strings.Contains(err.Error(), "session") {
				t.Fatalf("Load() = (%v, %v), want session validation error", snapshot, err)
			}
		})
	}
}

func TestPersistedSessionCompatibilityAppliesOnlyToMissingLegacyField(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, filepath.Join("testdata", "valid"), MVPDescriptors())
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	legacyWorkflow := mutateWorkflowStageSession(t, workflowJSON, nil, true)
	decodedWorkflow, err := DecodeResolvedWorkflowSnapshot(legacyWorkflow)
	if err != nil {
		t.Fatal(err)
	}
	if got := decodedWorkflow.Stages["copy"].Session; got != contracts.WorkerSessionShared {
		t.Fatalf("legacy Workflow session = %q, want shared", got)
	}

	stageJSON, err := json.Marshal(workflow.Stages["copy"])
	if err != nil {
		t.Fatal(err)
	}
	legacyStage := mutateStageSession(t, stageJSON, nil, true)
	decodedStage, err := DecodeResolvedStageSnapshot(legacyStage)
	if err != nil {
		t.Fatal(err)
	}
	if decodedStage.Session != contracts.WorkerSessionShared ||
		!reflect.DeepEqual(decodedStage, decodedWorkflow.Stages["copy"]) {
		t.Fatalf("legacy Stage and Workflow normalization differ: %+v / %+v", decodedStage, decodedWorkflow.Stages["copy"])
	}

	for _, test := range []struct {
		name  string
		value any
	}{
		{name: "null", value: nil},
		{name: "empty", value: ""},
		{name: "unknown", value: "reset"},
		{name: "number", value: 17},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			invalidWorkflow := mutateWorkflowStageSession(t, workflowJSON, test.value, false)
			if _, err := DecodeResolvedWorkflowSnapshot(invalidWorkflow); err == nil {
				t.Fatal("persisted Workflow accepted explicit invalid session")
			}
			invalidStage := mutateStageSession(t, stageJSON, test.value, false)
			if _, err := DecodeResolvedStageSnapshot(invalidStage); err == nil {
				t.Fatal("persisted Stage accepted explicit invalid session")
			}
		})
	}
}

func setWorkflowSession(t *testing.T, root, raw string) {
	t.Helper()
	path := filepath.Join(root, "workflows", "artifact_copy.yaml")
	content := string(readFile(t, path))
	content = strings.Replace(
		content,
		"      planner: passthrough@1\n",
		"      planner: passthrough@1\n      session: "+raw+"\n",
		1,
	)
	writeFile(t, path, []byte(content))
}

func mutateWorkflowStageSession(
	t *testing.T, data []byte, value any, remove bool,
) []byte {
	t.Helper()
	var document map[string]any
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	stages := document["stages"].(map[string]any)
	stage := stages["copy"].(map[string]any)
	if remove {
		delete(stage, "session")
	} else {
		stage["session"] = value
	}
	encoded, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func mutateStageSession(t *testing.T, data []byte, value any, remove bool) []byte {
	t.Helper()
	var stage map[string]any
	if err := json.Unmarshal(data, &stage); err != nil {
		t.Fatal(err)
	}
	if remove {
		delete(stage, "session")
	} else {
		stage["session"] = value
	}
	encoded, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}
