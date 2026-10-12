package artifacts_test

import (
	"context"
	"errors"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// TestPostgresThawOutputsOnlyInsideARunContinuation checks the thaw that Run
// continuation reaches through the ArtifactStore: it takes only a RunScope,
// and outside the transaction that records the Run's continuation receipt the
// database refuses it and the outputs stay frozen. The resume tests of
// internal/runstore cover the thaw inside such a transaction.
func TestPostgresThawOutputsOnlyInsideARunContinuation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	createArtifactRun(t, ctx, pool, "thaw-run", true)
	service := NewService(NewPostgresRepository(pool))
	run, err := service.Run("thaw-run")
	if err != nil {
		t.Fatal(err)
	}
	written, err := run.Write(ctx, ArtifactRef{Namespace: "scratch", Name: "result"},
		Payload{MediaType: "text/plain", Data: []byte("frozen result")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.BindOutputExact(ctx, "thaw-run", "result", written.Ref, nil); err != nil {
		t.Fatal(err)
	}
	if err := service.FreezeRunOutputs(ctx, "thaw-run"); err != nil {
		t.Fatal(err)
	}

	project, err := ProjectScope("project-thaw")
	if err != nil {
		t.Fatal(err)
	}
	if err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		return NewPostgresTxRepository(tx).ThawOutputs(ctx, project)
	}); !errors.Is(err, ErrInvalidScope) {
		t.Fatalf("thaw of a ProjectScope = %v", err)
	}
	scope, err := RunScope("thaw-run")
	if err != nil {
		t.Fatal(err)
	}
	err = pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		return NewPostgresTxRepository(tx).ThawOutputs(ctx, scope)
	})
	if persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation {
		t.Fatalf("thaw without a continuation receipt = %v", err)
	}
	output, err := run.Metadata(ctx, ArtifactRef{Namespace: "outputs", Name: "result"})
	if err != nil || !output.Frozen {
		t.Fatalf("output after the refused thaw = (%+v, %v)", output, err)
	}
}
