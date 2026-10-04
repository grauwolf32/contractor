package postgres

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestPostgresLLMCredentialSideTablesFoldMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 87)

	_, err := pool.Exec(ctx, `
INSERT INTO llm_credential_identities (credential_id, reserved_at)
VALUES ('active', now()), ('deleted', now());
INSERT INTO credential_operations (
    operation_id, idempotency_key, request_hash, credential_id, operation_kind, phase,
    request_schema_version, request, created_at, updated_at
)
SELECT kind||':'||id, kind||'-'||id, 'sha256:'||repeat('a',64), id, kind, phase,
       'contractor.credentials/v1', jsonb_build_object('credentialId', id), now(), now()
FROM (VALUES ('active','create','completed'), ('deleted','create','completed'),
             ('deleted','delete','completed')) AS operation(id, kind, phase);
INSERT INTO llm_credential_tombstones (credential_id, actor_id, deleted_at)
VALUES ('deleted', 'operator', now());
`)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 88) {
		t.Fatalf("fold LLM credential side tables = %+v, %v", result, err)
	}
	var tablesGone bool
	if err := pool.QueryRow(ctx, `
SELECT to_regclass('llm_credential_identities') IS NULL AND to_regclass('llm_credential_tombstones') IS NULL`,
	).Scan(&tablesGone); err != nil || !tablesGone {
		t.Fatalf("side tables dropped = %t, %v", tablesGone, err)
	}
	var pgErr *pgconn.PgError
	for name, statement := range map[string]string{
		"reused ID": `INSERT INTO credential_operations (
    operation_id, idempotency_key, request_hash, credential_id, operation_kind, phase,
    request_schema_version, request, created_at, updated_at
) VALUES ('create:again', 'create-again', 'sha256:'||repeat('b',64), 'deleted', 'create', 'prepared',
    'contractor.credentials/v1', '{}', now(), now())`,
		"unreserved record": activeCredentialInsert("unreserved"),
		"tombstoned record": activeCredentialInsert("deleted"),
	} {
		_, err := pool.Exec(ctx, statement)
		if !errors.As(err, &pgErr) || (pgErr.Code != "23505" && pgErr.Code != "23503") {
			t.Fatalf("%s accepted or failed unexpectedly: %v", name, err)
		}
	}
	if _, err := pool.Exec(ctx, activeCredentialInsert("active")); err != nil {
		t.Fatalf("reserved record rejected: %v", err)
	}
}

func activeCredentialInsert(credentialID string) string {
	return `INSERT INTO llm_credentials (
    credential_id, llm_gateway_id, llm_gateway_version, llm_gateway_digest, remote_key_id,
    gateway_policy, encryption_schema_version, key_id, nonce, ciphertext, created_at
) VALUES ('` + credentialID + `', 'gateway', '1', 'sha256:'||repeat('c',64), 'remote-` + credentialID + `',
    '{}', 'contractor.credentials/v1', 'sha256:'||repeat('d',64),
    decode(repeat('00',12),'hex'), decode(repeat('11',32),'hex'), now())`
}
