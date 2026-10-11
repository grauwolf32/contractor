package runstore

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// A public create-Run key reused with another request digest is the permanent
// contracts.ErrIdempotencyConflict on both the create and the replay lookup,
// never the retryable optimistic ErrConflict, and the original Run is
// unchanged. The same digest replays the original Run.
func TestPostgresPublicRunKeyReuseIsAPermanentIdempotencyConflict(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	store := NewPostgresStore(isolatedRunStorePool(t, ctx))
	digest := func(fill string) string { return "sha256:" + strings.Repeat(fill, 64) }
	first := CreateRunIdempotentParams{
		CreateRunParams: testRunParams("run-idempotent"), IdempotencyKey: "create-run-key", RequestDigest: digest("a"),
	}
	created, inserted, err := store.CreateRunIdempotent(ctx, first)
	if err != nil || !inserted {
		t.Fatalf("create = (%+v, %t, %v)", created, inserted, err)
	}
	replay := first
	replay.RunID = "run-idempotent-retry"
	replayed, inserted, err := store.CreateRunIdempotent(ctx, replay)
	if err != nil || inserted || replayed.RunID != created.RunID {
		t.Fatalf("same-digest replay = (%+v, %t, %v)", replayed, inserted, err)
	}
	changed := replay
	changed.RequestDigest = digest("b")
	if _, _, err := store.CreateRunIdempotent(ctx, changed); !errors.Is(err, contracts.ErrIdempotencyConflict) ||
		errors.Is(err, ErrConflict) {
		t.Fatalf("create with a reused key = %v, want only the permanent idempotency conflict", err)
	}
	if _, found, err := store.LookupRunIdempotency(ctx, first.OwnerID, first.IdempotencyKey, digest("b")); found ||
		!errors.Is(err, contracts.ErrIdempotencyConflict) || errors.Is(err, ErrConflict) {
		t.Fatalf("lookup with a reused key = (%t, %v)", found, err)
	}
	if run, found, err := store.LookupRunIdempotency(ctx, first.OwnerID, first.IdempotencyKey, digest("a")); err != nil ||
		!found || run.RunID != created.RunID {
		t.Fatalf("lookup with the original digest = (%+v, %t, %v)", run, found, err)
	}
	if _, err := store.GetRun(ctx, "run-idempotent-retry"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("a reused key created another Run: %v", err)
	}
}
