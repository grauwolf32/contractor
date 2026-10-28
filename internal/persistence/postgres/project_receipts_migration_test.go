package postgres

import (
	"bytes"
	"errors"
	"os"
	"testing"

	"github.com/jackc/pgx/v5"
)

func TestPostgresProjectReceiptMigrationPreservesAcceptedIntent(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, dsn)
	previous, _ := embeddedMigration(t, "project_mutation_receipts")
	installMigrationPrefix(t, ctx, pool, previous)
	if _, err := pool.Exec(ctx, `INSERT INTO projects (project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES ('receipt-project','owner','project','Receipts','create-project',$1)`, keyedDigest("a")); err != nil {
		t.Fatal(err)
	}
	oldKey := ReceiptKey{scope: []string{"project-sitemap", "owner", "receipt-project"}, key: "accepted"}
	digest := keyedDigest("b")
	if _, err := pool.Exec(ctx, `INSERT INTO mutation_receipts
(key_scope,idempotency_key,operation,request_digest,resource_id,response,owner_id,project_id,created_at)
VALUES ($1,$2,'sitemap.mutate',$3,'receipt-project',$4,'owner','receipt-project','2026-10-27T12:34:56Z')`,
		oldKey.scope, oldKey.key, digest, []byte(`{"result":{"revision":"original"},"policy":["unchanged"]}`)); err != nil {
		t.Fatal(err)
	}
	var before []byte
	if err := pool.QueryRow(ctx, `SELECT to_jsonb(receipt)-'key_scope' FROM mutation_receipts receipt WHERE key_scope=$1 AND idempotency_key=$2`, oldKey.scope, oldKey.key).Scan(&before); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	if err := VerifyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	key := ProjectReceiptKey("owner", "receipt-project", "receipt-project", "sitemap.mutate", "accepted")
	var after []byte
	if err := pool.QueryRow(ctx, `SELECT to_jsonb(receipt)-'key_scope' FROM mutation_receipts receipt WHERE key_scope=$1 AND idempotency_key=$2`, key.scope, key.key).Scan(&after); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("migration changed receipt content, authority or timestamp")
	}
	if _, found, err := ReadReceipt(ctx, pool, oldKey); err != nil || found {
		t.Fatalf("obsolete key remains readable: %t %v", found, err)
	}
	stored, found, err := FindReceipt(ctx, pool, key, digest)
	if err != nil || !found {
		t.Fatalf("migrated receipt unavailable: %t %v", found, err)
	}
	if err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		got, replayed, err := KeyedMutation(ctx, tx, key, digest, func() (Receipt, error) {
			return Receipt{}, errors.New("migration lost the accepted receipt")
		})
		if err == nil && (!replayed || !bytes.Equal(got.Response, stored.Response)) {
			return errors.New("migrated replay changed the accepted response")
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if err := claimReceipt(ctx, pool, oldKey, stored); Classify(err) != ClassInvalid {
		t.Fatalf("obsolete receipt shape accepted: %v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE mutation_receipts SET response='\x7b7d' WHERE key_scope=$1`, key.scope); Classify(err) != ClassRejected {
		t.Fatalf("migrated receipt can be changed: %v", err)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM mutation_receipts WHERE key_scope=$1`, key.scope); Classify(err) != ClassRejected {
		t.Fatalf("migrated receipt can be deleted: %v", err)
	}
	if err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := DeleteProjectReceiptsBatch(ctx, tx, "receipt-project")
		return err
	}); Classify(err) != ClassRejected {
		t.Fatalf("active Project receipt purge accepted: %v", err)
	}
}
