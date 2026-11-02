package evalstore

import (
	"context"
	"strconv"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// TestSharedPostgresEvalReceiptsKeepOnlyPublicIdempotencyKeys holds the
// managed Eval receipt store to the public Idempotency-Key grammar even for a
// MutationIdentity that did not come from evaldomain.IdentifyMutation: a key
// outside it is eval_invalid and leaves no receipt or dataset, and every
// valid shared key, a 128-byte key included, records exactly one receipt.
func TestSharedPostgresEvalReceiptsKeepOnlyPublicIdempotencyKeys(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner-a", "eval-a")
	doc := fixture(t, "dataset", "DatasetInput")
	valid := mutation(t, "template", 0, doc)
	cases := contractstest.ReadIdempotencyKeyCases(t)
	receipts := func() int {
		t.Helper()
		var count int
		if err := pool.QueryRow(ctx, `SELECT count(*) FROM mutation_receipts WHERE key_scope[1] = 'eval'`).Scan(&count); err != nil {
			t.Fatal(err)
		}
		return count
	}
	for _, key := range cases.Invalid {
		identity := evaldomain.MutationIdentity{Key: key, RequestSHA256: valid.RequestSHA256}
		err := transaction(pool, func(s *Store) error {
			_, err := s.PutDataset(ctx, scope, "r-invalid", doc, identity)
			return err
		})
		if !code(err, "eval_invalid") {
			t.Fatalf("key %q: error %v, want eval_invalid", key, err)
		}
	}
	if count := receipts(); count != 0 {
		t.Fatalf("rejected keys left %d receipts", count)
	}
	if _, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).Dataset(ctx, scope, "trace-small", "r-invalid"); !code(err, "eval_not_found") {
		t.Fatalf("a rejected key wrote its dataset: %v", err)
	}
	for index, key := range cases.Valid {
		identity := mutation(t, key, 0, doc)
		mustTx(t, pool, func(s *Store) error {
			_, err := s.PutDataset(ctx, scope, "r"+strconv.Itoa(index), doc, identity)
			return err
		})
	}
	if count := receipts(); count != len(cases.Valid) {
		t.Fatalf("valid keys recorded %d receipts, want %d", count, len(cases.Valid))
	}
}
