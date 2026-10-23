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

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
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
	encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil || len(encoded) != resolved.MaxWorkflowSnapshotBytes {
		t.Fatalf("snapshot fixture = (%d bytes, %v), want exactly %d", len(encoded), err, resolved.MaxWorkflowSnapshotBytes)
	}
	store := runstore.NewPostgresStore(isolatedSchedulerPool(t, ctx))
	const runID = "run-snapshot-at-bound"
	if _, err := runstoretest.CreateRun(ctx, store, runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel,
		RunID:     runID, OwnerID: "owner", WorkflowName: workflow.Ref.Name,
		WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: encoded, Parameters: map[string]string{"objective": "bound"},
		RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatalf("store a Run snapshot at the bound: %v", err)
	}
	run, err := store.GetRun(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	if len(run.WorkflowSnapshot) <= resolved.MaxWorkflowSnapshotBytes {
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
	over, err := resolved.DecodeResolvedWorkflowSnapshot(encoded)
	if err != nil {
		t.Fatal(err)
	}
	stage := over.Stages[over.EntryStage]
	stage.Instructions.Text += "a"
	stage.Instructions.Digest = contentdigest.Bytes([]byte(stage.Instructions.Text))
	over.Stages[over.EntryStage] = stage
	encodedOver, err := json.Marshal(over)
	if err != nil || len(encodedOver) != resolved.MaxWorkflowSnapshotBytes+1 {
		t.Fatalf("over-bound fixture = (%d bytes, %v)", len(encodedOver), err)
	}
	run.WorkflowSnapshot = encodedOver
	if _, err := decodeExecutableWorkflow(run); !errors.Is(err, ErrUnsupportedWorkflow) ||
		!strings.Contains(err.Error(), resolved.ErrWorkflowSnapshotTooLarge.Error()) {
		t.Fatalf("over-bound snapshot error = %v, want ErrUnsupportedWorkflow for its size", err)
	}
}

// schedulerWorkflowAtSnapshotBound pads the entry Stage instruction of
// artifact-copy@1 until its snapshot has exactly MaxWorkflowSnapshotBytes.
func schedulerWorkflowAtSnapshotBound(t *testing.T) resolved.ResolvedWorkflow {
	t.Helper()
	root := filepath.Join(t.TempDir(), "configs")
	if err := os.CopyFS(root, os.DirFS("../../testdata/configs")); err != nil {
		t.Fatal(err)
	}
	load := func(instruction string) resolved.ResolvedWorkflow {
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
	base, err := resolved.EncodeResolvedWorkflowSnapshot(load("a"))
	if err != nil {
		t.Fatal(err)
	}
	// The ASCII instruction is embedded once, so its length adds exactly.
	return load(strings.Repeat("a", 1+resolved.MaxWorkflowSnapshotBytes-len(base)))
}
