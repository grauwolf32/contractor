package memorysnapshot

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/operations"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func snapshotOperationFixture(t *testing.T, project bool, names string, configure ...func(*pgxpool.Config)) (*Module, operations.Call, *pgxpool.Pool) {
	t.Helper()
	ctx := t.Context()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{Configure: func(cfg *pgxpool.Config) {
		cfg.MaxConns = 1
		for _, f := range configure {
			f(cfg)
		}
	}})
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	root := configtest.CopyWithPolicies(t, "../../config/testdata/valid")
	data, err := os.ReadFile("../../config/testdata/memory-snapshot/workflows/snapshot.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "workflows", "snapshot.yaml"), data, 0600); err != nil {
		t.Fatal(err)
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := catalog.ResolveRunWorkflow(ctx, "memory-snapshot@1", nil)
	if err != nil {
		t.Fatal(err)
	}
	var projectID *string
	if project {
		id := "snapshot-project"
		projectID = &id
		if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: id, OwnerID: "owner", Kind: projectstore.KindProject, Name: "Snapshot", IdempotencyKey: "create", RequestDigest: "sha256:" + strings.Repeat("a", 64)}); err != nil {
			t.Fatal(err)
		}
	}
	storage := artifactpgtest.NewStorage()
	runs := runstore.NewPostgresStore(pool, storage)
	encoded, _ := json.Marshal(workflow)
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{RunID: "snapshot-run", OwnerID: "owner", ProjectID: projectID, LaneClass: resolved.LaneClassTool, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: encoded, Parameters: map[string]string{"note_names": names}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, "snapshot-run", runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.ClaimRunnableRun(ctx, "snapshot-claim", time.Minute, nil, resolved.LaneClassTool); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, "snapshot-run", runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	stage, _ := json.Marshal(workflow.Stages[workflow.EntryStage])
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{StageExecutionID: "snapshot-stage", RunID: "snapshot-run", StageName: workflow.EntryStage, Attempt: 1, LaneClass: resolved.LaneClassTool, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stage, StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{Parameters: map[string]string{"note_names": names}, Artifacts: map[string]runstore.PinnedContextArtifact{}}}); err != nil {
		t.Fatal(err)
	}
	if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		s := runstore.NewPostgresTxStore(tx, storage)
		if _, err := s.LockRunState(ctx, "snapshot-run", runstore.RunRunning); err != nil {
			return err
		}
		if _, err := s.LockStageState(ctx, "snapshot-run", "snapshot-stage", runstore.StagePreparing); err != nil {
			return err
		}
		if _, err := s.MarkStageAdmitted(ctx, "snapshot-stage"); err != nil {
			return err
		}
		_, err := s.StartStageOperation(ctx, "snapshot-run", "snapshot-stage", "snapshot-claim")
		return err
	}); err != nil {
		t.Fatal(err)
	}
	store, err := memory.NewPostgresStore(pool, storage)
	if err != nil {
		t.Fatal(err)
	}
	call := operations.Call{RunID: "snapshot-run", StageExecutionID: "snapshot-stage", ClaimID: "snapshot-claim", Arguments: map[string]any{"namespace": "writer", "notes": names}, Inputs: map[string]contracts.ArtifactRef{}, Outputs: map[string]contracts.ArtifactRef{"snapshot": {Namespace: "snapshot", Name: "selected-notes"}}}
	return New(store), call, pool
}

func writeSnapshotNote(t *testing.T, ctx context.Context, pool *pgxpool.Pool, run, name, content string) artifacts.Metadata {
	t.Helper()
	store, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(run)
	target := contracts.ArtifactRef{Namespace: "writer", Name: "memory." + name}
	var revision *string
	var ordinal uint64
	if strings.HasPrefix(name, "note_") {
		_, _ = fmt.Sscanf(name, "note_%d", &ordinal)
	} else if name == "ignored" {
		ordinal = 1
	}
	if metadata, err := store.Metadata(ctx, target); err == nil {
		revision = metadata.Ref.Revision
	} else if !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatal(err)
	}
	data, err := memory.Encode(memory.StoredNote{SchemaVersion: memory.SchemaVersion, Name: name, Content: content, Tags: []string{}, Ordinal: ordinal})
	if err != nil {
		t.Fatal(err)
	}
	result, err := store.Write(ctx, target, artifacts.Payload{MediaType: memory.MediaType, Data: data}, revision)
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := store.Metadata(ctx, result.Ref)
	if err != nil {
		t.Fatal(err)
	}
	return metadata
}

func TestPostgresMemorySnapshotSelectsNotesAndReplaysBeforeMutableReads(t *testing.T) {
	for _, project := range []bool{false, true} {
		t.Run(fmt.Sprint(project), func(t *testing.T) {
			m, call, pool := snapshotOperationFixture(t, project, "selected")
			ctx := t.Context()
			writeSnapshotNote(t, ctx, pool, call.RunID, "selected", "selected confidential marker: nginx")
			writeSnapshotNote(t, ctx, pool, call.RunID, "ignored", "ignored confidential marker")
			first, err := m.Run(ctx, call)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(first.Summary, "confidential") {
				t.Fatal("content leaked into Stage summary")
			}
			store, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(call.RunID)
			before, err := store.Read(ctx, first.Artifacts["snapshot"])
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(before.Payload.Data, []byte("nginx")) || bytes.Contains(before.Payload.Data, []byte("ignored")) || bytes.Contains(before.Payload.Data, []byte("memory.")) || bytes.Contains(before.Payload.Data, []byte("revision")) || bytes.Contains(before.Payload.Data, []byte("ordinal")) {
				t.Fatalf("snapshot exported wrong fields: %s", before.Payload.Data)
			}
			writeSnapshotNote(t, ctx, pool, call.RunID, "selected", "changed after committed snapshot")
			replay, err := m.Run(ctx, call)
			if err != nil || !first.Artifacts["snapshot"].SameExact(replay.Artifacts["snapshot"]) {
				t.Fatal(replay, err)
			}
			after, err := store.Read(ctx, replay.Artifacts["snapshot"])
			if err != nil || !bytes.Equal(before.Payload.Data, after.Payload.Data) {
				t.Fatal("recovery substituted mutable note bytes", err)
			}
			metadata, err := store.Metadata(ctx, replay.Artifacts["snapshot"])
			if err != nil || !metadata.Frozen {
				t.Fatal(metadata, err)
			}
			// Independent ordinary writes get their own version; a frozen exact fork reuses it.
			written, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "copy", Name: "independent"}, before.Payload, nil)
			if err != nil {
				t.Fatal(err)
			}
			independent, err := store.Metadata(ctx, written.Ref)
			if err != nil || independent.ContentCacheKey == metadata.ContentCacheKey {
				t.Fatal("independent write reused a version", err)
			}
			scope, _ := artifacts.RunScope(call.RunID)
			fork, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Fork(ctx, artifacts.Fork{Kind: artifacts.LineageContextProjection, Source: scope, SourceRef: metadata.Ref, SourceFrozen: true, Target: scope, TargetRef: contracts.ArtifactRef{Namespace: "copy", Name: "fork"}, Frozen: true})
			if err != nil {
				t.Fatal(err)
			}
			forkMeta, err := store.Metadata(ctx, fork.TargetRef)
			if err != nil || forkMeta.ContentCacheKey != metadata.ContentCacheKey {
				t.Fatal("fork lost exact version", err)
			}
		})
	}
}

func TestPostgresMemorySnapshotRejectsSourcesCollisionsAndLostAuthority(t *testing.T) {
	for _, scenario := range []string{"missing-note", "malformed-note", "claim-lost", "cancelled", "project-deleting", "changed-selection", "changed-output", "foreign-run", "collision"} {
		t.Run(scenario, func(t *testing.T) {
			m, call, pool := snapshotOperationFixture(t, true, "selected")
			ctx := t.Context()
			runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
			if scenario != "missing-note" {
				writeSnapshotNote(t, ctx, pool, call.RunID, "selected", "must not enter errors")
			}
			store, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(call.RunID)
			switch scenario {
			case "malformed-note":
				current, _ := store.Metadata(ctx, contracts.ArtifactRef{Namespace: "writer", Name: "memory.selected"})
				if _, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "writer", Name: "memory.selected"}, artifacts.Payload{MediaType: memory.MediaType, Data: []byte(`{"content":"must not enter errors"}`)}, current.Ref.Revision); err != nil {
					t.Fatal(err)
				}
			case "claim-lost":
				if err := runs.ReleaseRunClaim(ctx, call.RunID, call.ClaimID); err != nil {
					t.Fatal(err)
				}
			case "cancelled":
				if _, err := runs.RequestRunCancellation(ctx, call.RunID, runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: time.Now()}); err != nil {
					t.Fatal(err)
				}
			case "project-deleting":
				if _, _, err := projectstore.NewPostgresStore(pool).BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: "owner", ProjectID: "snapshot-project", ExpectedRevision: 1}); err != nil {
					t.Fatal(err)
				}
			case "changed-selection":
				call.Arguments["notes"] = "another"
			case "changed-output":
				call.Outputs["snapshot"] = contracts.ArtifactRef{Namespace: "snapshot", Name: "another"}
			case "foreign-run":
				call.RunID = "foreign-run"
			case "collision":
				if _, err := store.Write(ctx, call.Outputs["snapshot"], artifacts.Payload{MediaType: "application/json", Data: []byte(`{}`)}, nil); err != nil {
					t.Fatal(err)
				}
			}
			_, err := m.Run(ctx, call)
			if err == nil || strings.Contains(err.Error(), "must not enter errors") {
				t.Fatalf("unsafe failure: %v", err)
			}
			if scenario != "collision" {
				if _, err := store.Metadata(ctx, contracts.ArtifactRef{Namespace: "snapshot", Name: "selected-notes"}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
					t.Fatalf("partial snapshot after refusal: %v", err)
				}
			}
		})
	}
}

func TestPostgresMemorySnapshotReadsSixteenSelectedNotesWithinFullNotebook(t *testing.T) {
	names := make([]string, 16)
	for i := range names {
		names[i] = fmt.Sprintf("note_%d", i)
	}
	m, call, pool := snapshotOperationFixture(t, true, strings.Join(names, ","))
	ctx := t.Context()
	for i := range 128 {
		content := "Unselected ordinary working note"
		if i < 16 {
			content = strings.Repeat("x", 32*1024-200)
		}
		writeSnapshotNote(t, ctx, pool, call.RunID, fmt.Sprintf("note_%d", i), content)
	}
	result, err := m.Run(ctx, call)
	if err != nil {
		t.Fatal(err)
	}
	store, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(call.RunID)
	saved, err := store.Read(ctx, result.Artifacts["snapshot"])
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Notes []json.RawMessage `json:"notes"`
	}
	if err := json.Unmarshal(saved.Payload.Data, &document); err != nil || len(document.Notes) != 16 || len(saved.Payload.Data) > 512*1024 || len(saved.Payload.Data) < 500*1024 {
		t.Fatalf("bounded snapshot: notes=%d bytes=%d error=%v", len(document.Notes), len(saved.Payload.Data), err)
	}
}

// A driver trace cancels after the INSERT has executed but before the owning
// transaction commits, exercising rollback without a production test hook.
type cancelSnapshotWrite struct {
	cancel context.CancelFunc
	fired  bool
}

func (q *cancelSnapshotWrite) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	if q.cancel != nil && strings.Contains(data.SQL, "INSERT INTO artifact_bindings") && len(data.Args) > 3 && data.Args[2] == "snapshot" && data.Args[3] == "selected-notes" {
		return context.WithValue(ctx, snapshotWriteKey{}, true)
	}
	return ctx
}

type snapshotWriteKey struct{}

func (q *cancelSnapshotWrite) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryEndData) {
	if ctx.Value(snapshotWriteKey{}) == true && data.Err == nil {
		q.fired = true
		q.cancel()
	}
}
func TestPostgresMemorySnapshotCancellationRollsBackCompletedWrite(t *testing.T) {
	trace := &cancelSnapshotWrite{}
	m, call, pool := snapshotOperationFixture(t, true, "selected", func(c *pgxpool.Config) { c.ConnConfig.Tracer = trace })
	writeSnapshotNote(t, t.Context(), pool, call.RunID, "selected", "selected source")
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	trace.cancel = cancel
	if _, err := m.Run(ctx, call); err == nil || !trace.fired {
		t.Fatalf("write cancellation not exercised: fired=%v error=%v", trace.fired, err)
	}
	store, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(call.RunID)
	if _, err := store.Metadata(t.Context(), call.Outputs["snapshot"]); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("cancelled write survived: %v", err)
	}
}
