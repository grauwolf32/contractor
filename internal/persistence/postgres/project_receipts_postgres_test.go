package postgres

import (
	"bytes"
	"errors"
	"fmt"
	"testing"

	"github.com/jackc/pgx/v5"
)

func TestPostgresProjectReceiptScopesAndStrictShape(t *testing.T) {
	f := newKeyedFixture(t)
	for _, value := range []struct{ project, resource, operation string }{
		{"keyed-project", "inventory", "inventory.merge"},
		{"keyed-project", "inventory", "inventory.annotate"},
		{"keyed-project", "other-inventory", "inventory.merge"},
		{"keyed-evaluation", "inventory", "inventory.merge"},
	} {
		key := ProjectReceiptKey("owner", value.project, value.resource, value.operation, "same-key")
		want := Receipt{OwnerID: "owner", ProjectID: value.project, ResourceID: value.resource, Operation: value.operation,
			Response: []byte(fmt.Sprintf(`{"value":%q}`, value.project+"/"+value.resource+"/"+value.operation))}
		for attempt := range 2 {
			if err := InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				got, replayed, err := KeyedMutation(f.ctx, tx, key, keyedDigest("a"), func() (Receipt, error) { return want, nil })
				if err == nil && (replayed != (attempt == 1) || !bytes.Equal(got.Response, want.Response)) {
					return fmt.Errorf("receipt scope collided: got=%+v replay=%t want=%+v", got, replayed, want)
				}
				return err
			}); err != nil {
				t.Fatal(err)
			}
		}
		if _, _, err := FindReceipt(f.ctx, f.pool, key, keyedDigest("b")); !errors.Is(err, ErrReceiptMismatch) {
			t.Fatalf("changed Project intent accepted: %v", err)
		}
	}
	key := ProjectReceiptKey("owner", "keyed-project", "inventory", "inventory.merge", "invalid")
	base := Receipt{OwnerID: "owner", ProjectID: "keyed-project", ResourceID: "inventory", Operation: "inventory.merge", RequestDigest: keyedDigest("a"), Response: []byte(`{}`)}
	for _, change := range []func(*Receipt){
		func(r *Receipt) { r.OwnerID = "other-owner" },
		func(r *Receipt) { r.ProjectID = "keyed-evaluation" },
		func(r *Receipt) { r.ResourceID = "other-resource" },
		func(r *Receipt) { r.Operation = "other.operation" },
		func(r *Receipt) { r.AuditID = "keyed-audit" },
	} {
		receipt := base
		change(&receipt)
		if err := claimReceipt(f.ctx, f.pool, key, receipt); Classify(err) != ClassInvalid {
			t.Fatalf("mismatched Project receipt accepted: %+v %v", receipt, err)
		}
	}
}

func TestPostgresProjectReceiptPurgeIsBoundedAndIsolated(t *testing.T) {
	f := newKeyedFixture(t)
	if _, err := f.pool.Exec(f.ctx, `INSERT INTO mutation_receipts
(key_scope,idempotency_key,operation,request_digest,resource_id,response,owner_id,project_id)
SELECT ARRAY['project','owner','keyed-project','inventory','inventory.merge'], 'key-'||n,
'inventory.merge',$1,'inventory',$2,'owner','keyed-project' FROM generate_series(1,$3) n`, keyedDigest("a"), []byte(`{}`), CleanupBatchRows+1); err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		key     ReceiptKey
		receipt Receipt
	}{
		{ProjectReceiptKey("owner", "keyed-evaluation", "inventory", "inventory.merge", "other-project"), Receipt{OwnerID: "owner", ProjectID: "keyed-evaluation", ResourceID: "inventory", Operation: "inventory.merge", RequestDigest: keyedDigest("a"), Response: []byte(`{}`)}},
		{EvalReceiptKey("owner", "keyed-project", "inventory", "inventory.merge", "other-family"), Receipt{OwnerID: "owner", ProjectID: "keyed-project", ResourceID: "inventory", Operation: "inventory.merge", RequestDigest: keyedDigest("a"), Response: []byte(`{}`)}},
	} {
		if err := claimReceipt(f.ctx, f.pool, item.key, item.receipt); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := f.pool.Exec(f.ctx, `UPDATE projects SET lifecycle_state='deleting',deletion_phase='cancelling',
deletion_requested_at=clock_timestamp(),revision=revision+1,updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond')
WHERE project_id='keyed-project'`); err != nil {
		t.Fatal(err)
	}
	for pass, wantRemaining := range []int64{1, 0, 0} {
		var done bool
		if err := InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			if _, err := tx.Exec(f.ctx, `SELECT project_id FROM projects WHERE project_id='keyed-project' FOR UPDATE`); err != nil {
				return err
			}
			var err error
			done, err = DeleteProjectReceiptsBatch(f.ctx, tx, "keyed-project")
			return err
		}); err != nil {
			t.Fatal(err)
		}
		var remaining, unrelated int64
		if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FILTER (WHERE project_id='keyed-project' AND key_scope[1]='project'),
count(*) FILTER (WHERE project_id<>'keyed-project' OR key_scope[1]<>'project') FROM mutation_receipts`).Scan(&remaining, &unrelated); err != nil {
			t.Fatal(err)
		}
		if remaining != wantRemaining || unrelated != 2 || done != (pass == 2) {
			t.Fatalf("purge pass %d: remaining=%d unrelated=%d done=%t", pass, remaining, unrelated, done)
		}
	}
}
