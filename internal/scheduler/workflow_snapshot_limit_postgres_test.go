package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// TestPostgresSchedulerDecodesAStoredSnapshotAtItsBound is the read half of
// the snapshot bound regression. The Scheduler measured the jsonb text that
// PostgreSQL returns, which is longer than the encoding Run creation bounds,
// so a Run accepted at the bound would fail as unsupported after acceptance.
// It now measures the decoded snapshot's encoding, as creation did (CAT-R76).
func TestPostgresSchedulerDecodesAStoredSnapshotAtItsBound(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	workflow := schedulerWorkflowAtSnapshotBound(t)
	encoded, err := workflowconfig.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil || len(encoded) != workflowconfig.MaxWorkflowSnapshotBytes {
		t.Fatalf("snapshot fixture = (%d bytes, %v), want exactly %d", len(encoded), err, workflowconfig.MaxWorkflowSnapshotBytes)
	}
	store := runstore.NewPostgresStore(isolatedSchedulerPool(t, ctx))
	const runID = "run-snapshot-at-bound"
	if _, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: "owner", WorkflowName: workflow.Ref.Name,
		WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: encoded, Parameters: map[string]string{"objective": "bound"},
		RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatalf("store a Run snapshot at the bound: %v", err)
	}
	run, err := store.GetRun(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	if len(run.WorkflowSnapshot) <= workflowconfig.MaxWorkflowSnapshotBytes {
		t.Fatalf("stored jsonb text = %d bytes; the fixture must show PostgreSQL returning longer text", len(run.WorkflowSnapshot))
	}
	executable, err := decodeExecutableWorkflow(run)
	if err != nil {
		t.Fatalf("Scheduler rejected a Run snapshot accepted at its bound: %v", err)
	}
	if executable.stageName != workflow.EntryStage {
		t.Fatalf("selected Stage = %q, want entry Stage %q", executable.stageName, workflow.EntryStage)
	}

	// The read path still refuses a snapshot one byte over the bound.
	over, err := workflowconfig.DecodeResolvedWorkflowSnapshot(encoded)
	if err != nil {
		t.Fatal(err)
	}
	stage := over.Stages[over.EntryStage]
	stage.Instructions.Text += "a"
	stage.Instructions.Digest = contentdigest.Bytes([]byte(stage.Instructions.Text))
	over.Stages[over.EntryStage] = stage
	encodedOver, err := json.Marshal(over)
	if err != nil || len(encodedOver) != workflowconfig.MaxWorkflowSnapshotBytes+1 {
		t.Fatalf("over-bound fixture = (%d bytes, %v)", len(encodedOver), err)
	}
	run.WorkflowSnapshot = encodedOver
	if _, err := decodeExecutableWorkflow(run); !errors.Is(err, ErrUnsupportedWorkflow) ||
		!strings.Contains(err.Error(), workflowconfig.ErrWorkflowSnapshotTooLarge.Error()) {
		t.Fatalf("over-bound snapshot error = %v, want ErrUnsupportedWorkflow for its size", err)
	}
}

// schedulerWorkflowAtSnapshotBound pads the entry Stage instruction of
// artifact-copy@1 until its snapshot has exactly MaxWorkflowSnapshotBytes.
func schedulerWorkflowAtSnapshotBound(t *testing.T) workflowconfig.ResolvedWorkflow {
	t.Helper()
	root := filepath.Join(t.TempDir(), "configs")
	if err := os.CopyFS(root, os.DirFS("../../testdata/configs")); err != nil {
		t.Fatal(err)
	}
	load := func(instruction string) workflowconfig.ResolvedWorkflow {
		if err := os.WriteFile(filepath.Join(root, "instructions", "copy-planner.md"), []byte(instruction), 0o644); err != nil {
			t.Fatal(err)
		}
		snapshot, err := configload.Load(root)
		if err != nil {
			t.Fatal(err)
		}
		workflow, err := snapshot.Workflow("artifact-copy@1")
		if err != nil {
			t.Fatal(err)
		}
		return workflow
	}
	base, err := workflowconfig.EncodeResolvedWorkflowSnapshot(load("a"))
	if err != nil {
		t.Fatal(err)
	}
	// The ASCII instruction is embedded once, so its length adds exactly.
	return load(strings.Repeat("a", 1+workflowconfig.MaxWorkflowSnapshotBytes-len(base)))
}
