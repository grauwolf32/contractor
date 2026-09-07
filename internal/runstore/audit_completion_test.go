package runstore

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestAuditCompletionPostgresOrdinaryOmissionAndSpoofedAuthority(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	runs := NewPostgresStore(pool)
	run := createTestRun(t, ctx, runs, "ordinary-completion")
	if run.AuditCompletion != nil {
		t.Fatal("ordinary creation invented completion")
	}
	loaded, err := NewPostgresStore(pool).GetRun(ctx, run.RunID)
	if err != nil || loaded.AuditCompletion != nil {
		t.Fatal("legacy omission changed on recovery", err)
	}
	task, manifest := "task-r1", "manifest-r1"
	completion := AuditCompletionSnapshot{Stage: "check", Agent: "checker", Contract: contracts.WorkerCompletionContract{
		Kind: contracts.AuditCheckResultsV1, Task: contracts.ArtifactRef{Namespace: "inputs", Name: "task", Revision: &task}, ExecutionManifest: contracts.ArtifactRef{Namespace: "inputs", Name: "manifest", Revision: &manifest}, ResultArtifact: contracts.ArtifactRef{Namespace: "check", Name: "result"}}}
	if err := runs.SetAuditCompletion(ctx, run.RunID, completion); !errors.Is(err, ErrConflict) {
		t.Fatal("ordinary Run gained authority", err)
	}
	_, err = pool.Exec(ctx, `UPDATE workflow_runs SET audit_completion = '{"stage":"check","agent":"checker","contract":{"kind":"audit-check-results@1"}}'::jsonb WHERE run_id = $1`, run.RunID)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatal("ordinary SQL authority injection accepted", err)
	}
}
