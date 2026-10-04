package postgres

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestPostgresRuntimeCredentialSideTablesFoldMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 85)

	_, err := pool.Exec(ctx, `
INSERT INTO runtime_credentials (
    credential_id, credential_kind, encryption_schema_version,
    key_id, nonce, ciphertext, created_by, created_at
)
SELECT id, 'http-proxy-bearer@1', 'contractor.runtime-credentials/v1',
       'sha256:'||repeat('a',64), decode(repeat('00',12),'hex'), decode(repeat('11',32),'hex'),
       'operator', '2026-10-01T00:00:00Z'
FROM (VALUES ('active'), ('deleted')) AS credential(id);
INSERT INTO runtime_credential_creations (
    idempotency_key_digest, request_mac, credential_id, credential_kind, actor_id, created_at
)
SELECT 'sha256:'||repeat(digit,64), decode(repeat('22',32),'hex'), id,
       'http-proxy-bearer@1', 'operator', '2026-10-01T00:00:00Z'
FROM (VALUES ('active','b'), ('deleted','c')) AS credential(id, digit);
INSERT INTO runtime_credential_tombstones (credential_id, actor_id, deleted_at)
VALUES ('deleted', 'cleaner', '2026-10-02T00:00:00Z');
`)
	if err != nil {
		t.Fatal(err)
	}

	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 86) {
		t.Fatalf("fold Runtime credential side tables = %+v, %v", result, err)
	}
	type row struct {
		key       string
		macBytes  int
		deletedBy *string
	}
	rows := map[string]row{}
	query, err := pool.Query(ctx, `
SELECT credential_id, idempotency_key_digest, octet_length(request_mac), deleted_by
FROM runtime_credentials ORDER BY credential_id`)
	if err != nil {
		t.Fatal(err)
	}
	for query.Next() {
		var id string
		var value row
		if err := query.Scan(&id, &value.key, &value.macBytes, &value.deletedBy); err != nil {
			t.Fatal(err)
		}
		rows[id] = value
	}
	if err := query.Err(); err != nil {
		t.Fatal(err)
	}
	active, deleted := rows["active"], rows["deleted"]
	if active.key != "sha256:"+strings.Repeat("b", 64) || active.macBytes != 32 || active.deletedBy != nil ||
		deleted.key != "sha256:"+strings.Repeat("c", 64) || deleted.deletedBy == nil || *deleted.deletedBy != "cleaner" {
		t.Fatalf("folded Runtime credentials = %+v", rows)
	}
	var sideTablesGone bool
	if err := pool.QueryRow(ctx, `
SELECT to_regclass('runtime_credential_creations') IS NULL AND to_regclass('runtime_credential_tombstones') IS NULL`,
	).Scan(&sideTablesGone); err != nil || !sideTablesGone {
		t.Fatalf("side tables dropped = %t, %v", sideTablesGone, err)
	}
	if _, err := pool.Exec(ctx, `
UPDATE runtime_credentials SET deleted_by = 'operator', deleted_at = clock_timestamp()
WHERE credential_id = 'active'`); err != nil {
		t.Fatalf("set deletion marker: %v", err)
	}
	for _, statement := range []string{
		`UPDATE runtime_credentials SET deleted_by = NULL, deleted_at = NULL WHERE credential_id = 'deleted'`,
		`UPDATE runtime_credentials SET request_mac = decode(repeat('33',32),'hex') WHERE credential_id = 'deleted'`,
		`DELETE FROM runtime_credentials WHERE credential_id = 'deleted'`,
	} {
		if _, err := pool.Exec(ctx, statement); err == nil {
			t.Fatalf("immutable Runtime credential changed by %s", statement)
		}
	}
}
