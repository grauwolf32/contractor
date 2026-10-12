package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

// TestPostgresStoreBoundsTheWorkflowSnapshotItStores is the regression for a
// store that accepted any snapshot size, so a Run the Scheduler could never
// execute was accepted first. Every create path refuses a snapshot larger
// than LIM-CAT-4 and stores nothing; a snapshot at the bound is stored with
// its content unchanged, although PostgreSQL returns longer jsonb text for it
// (CAT-R76).
func TestPostgresStoreBoundsTheWorkflowSnapshotItStores(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	store := NewPostgresStore(isolatedRunStorePool(t, ctx))

	over := testRunParams("run-snapshot-over")
	over.WorkflowSnapshot = workflowSnapshotOfSize(t, resolved.MaxWorkflowSnapshotBytes+1)
	project := "project-snapshot"
	overInProject := over
	overInProject.ProjectID = &project
	for name, create := range map[string]func() error{
		"CreateRun": func() error {
			_, err := store.CreateRun(ctx, over)
			return err
		},
		"CreateRunIdempotent": func() error {
			_, _, err := store.CreateRunIdempotent(ctx, CreateRunIdempotentParams{
				CreateRunParams: over, IdempotencyKey: "snapshot-over",
				RequestDigest: "sha256:" + strings.Repeat("a", 64),
			})
			return err
		},
		"CreateAuditRun": func() error {
			_, err := store.CreateAuditRun(ctx, CreateAuditRunParams{
				CreateRunParams: overInProject, AuditExecutionID: "execution-snapshot",
				AuditSubmissionKey: "submission-snapshot",
			})
			return err
		},
	} {
		if err := create(); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "workflowSnapshot") {
			t.Fatalf("%s with a snapshot over the bound error = %v, want ErrInvalid for workflowSnapshot", name, err)
		}
	}
	if _, err := store.GetRun(ctx, over.RunID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("rejected Run lookup error = %v, want ErrNotFound", err)
	}

	atBound := testRunParams("run-snapshot-at-bound")
	atBound.WorkflowSnapshot = workflowSnapshotOfSize(t, resolved.MaxWorkflowSnapshotBytes)
	if _, err := store.CreateRun(ctx, atBound); err != nil {
		t.Fatalf("CreateRun with a snapshot at the bound: %v", err)
	}
	stored, err := store.GetRun(ctx, atBound.RunID)
	if err != nil {
		t.Fatal(err)
	}
	var want, got map[string]any
	if err := json.Unmarshal(atBound.WorkflowSnapshot, &want); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(stored.WorkflowSnapshot, &got); err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("stored snapshot content changed (decode error %v)", err)
	}
	if len(stored.WorkflowSnapshot) <= resolved.MaxWorkflowSnapshotBytes {
		t.Fatalf("stored jsonb text = %d bytes; the fixture must show PostgreSQL returning longer text than was stored", len(stored.WorkflowSnapshot))
	}
}

// workflowSnapshotOfSize returns a compact JSON object of exactly size bytes
// whose keys and separators PostgreSQL's jsonb text output spaces out.
func workflowSnapshotOfSize(t *testing.T, size int) json.RawMessage {
	t.Helper()
	const prefix = `{"ref":{"name":"artifact-copy","version":"1"},"pad":"`
	const suffix = `"}`
	padding := size - len(prefix) - len(suffix)
	if padding < 0 {
		t.Fatalf("snapshot size %d is below the fixture minimum", size)
	}
	snapshot := json.RawMessage(prefix + strings.Repeat("a", padding) + suffix)
	if len(snapshot) != size || !json.Valid(snapshot) {
		t.Fatalf("snapshot fixture has %d bytes, want %d valid JSON bytes", len(snapshot), size)
	}
	return snapshot
}
