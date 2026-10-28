package postgres

import (
	"bytes"
	"os"
	"reflect"
	"testing"

	"github.com/jackc/pgx/v5"
)

// Renumbering unpublished sitemap migrations must leave the existing main
// ledger and receipts unchanged. A fresh-schema test alone cannot prove this.
func TestPostgresProjectSitemapMigrationsPreserveMainLedgerAndReceipts(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "project_sitemap_receipts")
	contextPrevious, _ := embeddedMigration(t, "context_projection_lineage")
	if contextPrevious != previous+1 {
		t.Fatal("sitemap migrations are not consecutive")
	}
	installMigrationPrefix(t, ctx, pool, previous)
	digest := "sha256:" + repeat('a')
	key := RuntimeManagementReceiptKey("sha256:" + repeat('b'))
	before := Receipt{Operation: "runtime-label.delete", ResourceID: "preserved-label", RequestDigest: digest, Response: []byte(`{"deleted":true}`)}
	if err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, _, err := KeyedMutation(ctx, tx, key, digest, func() (Receipt, error) { return before, nil })
		return err
	}); err != nil {
		t.Fatal(err)
	}
	ledger, err := readAppliedMigrations(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.AppliedVersions) < 2 || result.AppliedVersions[0] != previous+1 || result.AppliedVersions[1] != contextPrevious+1 {
		t.Fatalf("upgrade skipped or replaced migrations: %+v", result)
	}
	if err := VerifyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	upgraded, err := readAppliedMigrations(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	for version, value := range ledger {
		if !reflect.DeepEqual(value, upgraded[version]) {
			t.Fatalf("existing main migration %d changed", version)
		}
	}
	receipt, found, err := FindReceipt(ctx, pool, key, digest)
	if err != nil || !found || receipt.Operation != before.Operation || receipt.ResourceID != before.ResourceID || !bytes.Equal(receipt.Response, before.Response) {
		t.Fatalf("existing receipt changed: found=%t error=%v", found, err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO projects (project_id,owner_id,kind,name,request_idempotency_key,request_digest) VALUES ('sitemap-upgrade','owner','project','Sitemap upgrade','create-upgrade',$1)`, digest); err != nil {
		t.Fatal(err)
	}
	sitemapKey := ProjectReceiptKey("owner", "sitemap-upgrade", "sitemap-upgrade", "sitemap.mutate", "publish")
	if err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, _, err := KeyedMutation(ctx, tx, sitemapKey, digest, func() (Receipt, error) {
			return Receipt{OwnerID: "owner", ProjectID: "sitemap-upgrade", ResourceID: "sitemap-upgrade", Operation: "sitemap.mutate", Response: []byte(`{"accepted":true}`)}, nil
		})
		return err
	}); err != nil {
		t.Fatal(err)
	}
	repeated, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	if err != nil || len(repeated.AppliedVersions) != 0 || repeated.CurrentVersion != result.CurrentVersion {
		t.Fatalf("repeated startup migration: %+v %v", repeated, err)
	}
	if _, found, err := FindReceipt(ctx, pool, sitemapKey, digest); err != nil || !found {
		t.Fatalf("sitemap receipt did not survive verification: %t %v", found, err)
	}
	t.Logf("preserved %d main ledger entries; applied sitemap migrations %06d and %06d", len(ledger), previous+1, contextPrevious+1)
}
