package runstore

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestAuditCompletionPostgresOrdinaryOmissionAndSpoofedAuthority(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	runs := NewPostgresStore(pool, artifactpgtest.NewStorage())
	run := createTestRun(t, ctx, runs, "ordinary-completion")
	if run.AuditCompletion != nil {
		t.Fatal("ordinary creation invented completion")
	}
	loaded, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).GetRun(ctx, run.RunID)
	if err != nil || loaded.AuditCompletion != nil {
		t.Fatal("ordinary omission changed on recovery", err)
	}
	task := "context-r1"
	completion := AuditCompletionSnapshot{Stage: "check", Agent: "checker", Contract: contracts.WorkerCompletionContract{
		Kind: contracts.AuditCheckResultsV1, Context: contracts.ArtifactRef{Namespace: "inputs", Name: "context", Revision: &task}, ResultArtifact: contracts.ArtifactRef{Namespace: "check", Name: "result"}}}
	if err := runs.SetAuditCompletion(ctx, run.RunID, completion); !errors.Is(err, ErrConflict) {
		t.Fatal("ordinary Run gained authority", err)
	}
	_, err = pool.Exec(ctx, `UPDATE workflow_runs SET audit_completion = '{"stage":"check","agent":"checker","contract":{"kind":"audit-check-results@1"}}'::jsonb WHERE run_id = $1`, run.RunID)
	if persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation {
		t.Fatal("ordinary SQL authority injection accepted", err)
	}
}
