package artifacts_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresDescribeOriginalInputsPreservesOriginalsAndBoundsWork(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	base := isolatedArtifactPool(t, ctx)
	config := base.Config()
	config.MaxConns = 1
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	createArtifactRun(t, ctx, pool, "original-inputs", false)
	createArtifactRun(t, ctx, pool, "foreign-inputs", false)
	service := NewService(NewPostgresRepository(pool))
	owner, _ := service.User("user-1")
	run, _ := service.Run("original-inputs")
	want := make(map[string]contracts.ExactArtifact)
	sourceRefs := make(map[string]ArtifactRef)
	names := make([]string, MaxExactReadBatchSize)
	for i := range names {
		name := fmt.Sprintf("input-%02d", i)
		names[len(names)-i-1] = name
		source, err := owner.Write(ctx, ArtifactRef{Namespace: "source", Name: name}, Payload{MediaType: "text/plain", Data: []byte(name)}, nil)
		if err != nil {
			t.Fatal(err)
		}
		fork, err := artifactstest.ForkInput(ctx, service, "user-1", source.Ref, "original-inputs", name)
		if err != nil {
			t.Fatal(err)
		}
		metadata, err := run.Metadata(ctx, fork.TargetRef)
		if err != nil {
			t.Fatal(err)
		}
		want[name], sourceRefs[name] = metadata.Exact(), source.Ref
	}
	first := want[names[0]]
	advanced, err := run.Write(ctx, ArtifactRef{Namespace: "inputs", Name: names[0]}, Payload{MediaType: "text/plain", Data: []byte("edited working copy")}, first.Ref.Revision)
	if err != nil {
		t.Fatal(err)
	}
	// An ordinary current input binding without input_fork lineage is not an
	// original creation input, even when its name matches a selected slot.
	if _, err := service.Create(ctx, mustRunScope(t, "original-inputs"), ArtifactRef{Namespace: "inputs", Name: "not-original"}, Payload{MediaType: "text/plain", Data: []byte("working")}); err != nil {
		t.Fatal(err)
	}
	describe := func(selection []string, expected []string, statements int) {
		t.Helper()
		// MaxConns=1 proves the helper uses this transaction instead of opening
		// a nested pool operation. The counter also proves metadata-only I/O.
		tx, err := pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer tx.Rollback(ctx) //nolint:errcheck // cleanup after a successful commit is harmless
		counter := &countingDB{DBTX: tx}
		store, _ := NewService(NewPostgresRepository(counter)).Run("original-inputs")
		got, err := store.DescribeOriginalInputs(ctx, selection)
		if err != nil {
			t.Fatal(err)
		}
		queries, payloads := counter.reset()
		if queries != statements || payloads != 0 || len(got) != len(expected) {
			t.Fatalf("descriptors=%d queries=%d payloads=%d, want %d/%d/0", len(got), queries, payloads, len(expected), statements)
		}
		for i, name := range expected {
			if want[name].Verify(got[i]) != nil || !want[name].Ref.SameExact(got[i].Ref) {
				t.Fatalf("original %s at position %d changed: %+v", name, i, got[i])
			}
		}
		if err := tx.Commit(ctx); err != nil {
			t.Fatal(err)
		}
	}
	describe(nil, nil, 0)
	describe([]string{"absent", "not-original"}, nil, 1)
	describe([]string{names[0]}, names[:1], 2)
	describe([]string{"absent", names[0], "not-original", names[1]}, names[:2], 2)
	describe(names, names, 2)
	unrelated := make([]ArtifactRef, 7000)
	for i := range unrelated {
		unrelated[i] = ArtifactRef{Namespace: "scratch", Name: fmt.Sprintf("unrelated-%04d", i)}
	}
	seedRunBindings(t, ctx, pool, "original-inputs", unrelated)
	describe(names, names, 2)
	foreign, _ := service.Run("foreign-inputs")
	if got, err := foreign.DescribeOriginalInputs(ctx, names); err != nil || len(got) != 0 {
		t.Fatalf("foreign Run returned inputs: %v %v", got, err)
	}
	cancelled, stop := context.WithCancel(ctx)
	stop()
	if _, err := run.DescribeOriginalInputs(cancelled, names); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled description: %v", err)
	}
	// Inject a second original-fork edge at the edited revision. It is valid
	// relational storage but ambiguous original input identity, never latest.
	_, err = pool.Exec(ctx, `INSERT INTO artifact_lineage (
 target_scope_kind, target_scope_id, target_namespace, target_name, target_revision,
 source_scope_kind, source_scope_id, source_namespace, source_name, source_revision, lineage_kind)
 VALUES ('run','original-inputs','inputs',$1,$2,'user','user-1','source',$1,$3,'input_fork')`, names[0], *advanced.Ref.Revision, *sourceRefs[names[0]].Revision)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := run.DescribeOriginalInputs(ctx, names); got != nil || !errors.Is(err, ErrArtifactIntegrity) {
		t.Fatalf("ambiguous original lineage: %v %v", got, err)
	}
}
