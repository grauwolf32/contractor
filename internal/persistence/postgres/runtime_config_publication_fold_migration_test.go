package postgres

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestPostgresRuntimeConfigPublicationsFoldMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 86)

	_, err := pool.Exec(ctx, `
INSERT INTO runtime_config_versions (name, version, digest, canonical_document, actor_id, created_at)
VALUES ('proxy', '1', 'sha256:'||repeat('a',64), '{}', 'operator', '2026-10-01T00:00:00Z');
INSERT INTO runtime_config_publications (
    idempotency_key_digest, request_digest, config_name, config_version, config_digest, actor_id, published_at
) VALUES ('sha256:'||repeat('b',64), 'sha256:'||repeat('c',64), 'proxy', '1', 'sha256:'||repeat('a',64),
          'operator', '2026-10-01T00:00:00Z');
`)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 87) {
		t.Fatalf("fold RuntimeConfig publications = %+v, %v", result, err)
	}
	var keyDigest, requestDigest string
	var builtInReceipt *string
	var publicationsGone bool
	if err := pool.QueryRow(ctx, `
SELECT published.idempotency_key_digest, published.request_digest, builtin.idempotency_key_digest,
       to_regclass('runtime_config_publications') IS NULL
FROM runtime_config_versions AS published, runtime_config_versions AS builtin
WHERE published.name = 'proxy' AND builtin.name = 'contractor-empty'`,
	).Scan(&keyDigest, &requestDigest, &builtInReceipt, &publicationsGone); err != nil {
		t.Fatal(err)
	}
	if keyDigest != "sha256:"+strings.Repeat("b", 64) || requestDigest != "sha256:"+strings.Repeat("c", 64) ||
		builtInReceipt != nil || !publicationsGone {
		t.Fatalf("folded receipt = (%s, %s), built-in receipt %v, table dropped %t",
			keyDigest, requestDigest, builtInReceipt, publicationsGone)
	}
	if _, err := pool.Exec(ctx, `UPDATE runtime_config_versions SET request_digest = NULL, idempotency_key_digest = NULL WHERE name = 'proxy'`); err == nil {
		t.Fatal("immutable RuntimeConfig receipt changed")
	}
}
