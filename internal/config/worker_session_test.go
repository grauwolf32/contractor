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
			workflowJSON, err := json.Marshal(resolved)
			if err != nil {
				t.Fatal(err)
			}
			decodedWorkflow, err := DecodeResolvedWorkflowSnapshot(workflowJSON)
			if err != nil || !reflect.DeepEqual(decodedWorkflow, resolved) {
				t.Fatalf("persisted Workflow changed %s session: %v", mode, err)
			}
			stage := resolved.Stages["copy"]
			stageJSON, err := json.Marshal(stage)
			if err != nil {
				t.Fatal(err)
			}
			decodedStage, err := DecodeResolvedStageSnapshot(stageJSON)
			if err != nil || !reflect.DeepEqual(decodedStage, stage) {
				t.Fatalf("persisted Stage changed %s session: %v", mode, err)
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

func TestPersistedSessionRequiresAnExplicitValidMode(t *testing.T) {
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
	stageJSON, err := json.Marshal(workflow.Stages["copy"])
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		value  any
		remove bool
	}{
		{name: "missing", remove: true},
		{name: "null", value: nil},
		{name: "empty", value: ""},
		{name: "unknown", value: "reset"},
		{name: "number", value: 17},
		{name: "boolean", value: true},
		{name: "mapping", value: map[string]any{}},
		{name: "sequence", value: []any{}},
	} {
		t.Run(test.name, func(t *testing.T) {
			invalidWorkflow := mutateWorkflowStageSession(t, workflowJSON, test.value, test.remove)
			decodedWorkflow, err := DecodeResolvedWorkflowSnapshot(invalidWorkflow)
			if err == nil || !strings.Contains(strings.ToLower(err.Error()), "session") ||
				!reflect.DeepEqual(decodedWorkflow, ResolvedWorkflow{}) {
				t.Fatalf("invalid persisted Workflow = (%+v, %v)", decodedWorkflow, err)
			}
			invalidStage := mutateStageSession(t, stageJSON, test.value, test.remove)
			decodedStage, err := DecodeResolvedStageSnapshot(invalidStage)
			if err == nil || !strings.Contains(strings.ToLower(err.Error()), "session") ||
				!reflect.DeepEqual(decodedStage, ResolvedStage{}) {
				t.Fatalf("invalid persisted Stage = (%+v, %v)", decodedStage, err)
			}
		})
	}
}

func TestPersistedAuditProfileRejectsMissingStageSession(t *testing.T) {
	t.Parallel()
	profile, err := mustLoad(t, filepath.Join("..", "..", "testdata", "configs"), MVPDescriptors()).AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := json.Unmarshal(encoded, &document); err != nil {
		t.Fatal(err)
	}
	binding := document["workflows"].(map[string]any)["check"].(map[string]any)
	workflow := binding["workflow"].(map[string]any)
	for _, raw := range workflow["stages"].(map[string]any) {
		delete(raw.(map[string]any), "session")
	}
	invalid, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeResolvedAuditProfileSnapshot(invalid)
	if err == nil || !strings.Contains(err.Error(), "workerSessionMode") ||
		!reflect.DeepEqual(decoded, ResolvedAuditProfile{}) {
		t.Fatalf("invalid persisted AuditProfile = (%+v, %v)", decoded, err)
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
