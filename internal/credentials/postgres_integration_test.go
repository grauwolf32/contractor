package credentials

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresCredentialRepositoryEncryptedImmutableLifecyclePrimitives(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedCredentialPool(t, ctx, databaseURL)
	repository := NewRepository(pool)

	if count, err := repository.CountCredentials(ctx); err != nil || count != 0 {
		t.Fatalf("empty credential count = (%d, %v)", count, err)
	}
	secret := "sk-postgres-plaintext-must-not-appear"
	record, cipher := sealedTestRecord(t, "managed-worker", secret)
	if err := repository.ReserveCredentialID(ctx, record.CredentialID, record.CreatedAt); err != nil {
		t.Fatalf("reserve credential ID: %v", err)
	}
	if err := repository.InsertCredential(ctx, record); err != nil {
		t.Fatalf("insert credential: %v", err)
	}
	if err := repository.InsertCredential(ctx, record); err != nil {
		t.Fatalf("idempotent credential insert: %v", err)
	}
	if err := repository.VerifyActiveKey(ctx, cipher.KeyID()); err != nil {
		t.Fatalf("verify active key: %v", err)
	}
	wrongCipher, _ := NewTokenCipher(bytes.Repeat([]byte{0x72}, 32))
	if err := repository.VerifyActiveKey(ctx, wrongCipher.KeyID()); !errors.Is(err, ErrKeyUnavailable) {
		t.Fatalf("wrong active key error = %v", err)
	}
	if count, err := repository.CountCredentials(ctx); err != nil || count != 1 {
		t.Fatalf("credential count = (%d, %v)", count, err)
	}

	stored, err := repository.GetCredential(ctx, record.CredentialID)
	if err != nil || !recordsEqual(stored, record) {
		t.Fatalf("stored credential = (%+v, %v)", stored, err)
	}
	listed, err := repository.ListCredentials(ctx, "", 2)
	if err != nil || len(listed) != 1 || listed[0].CredentialID != record.CredentialID {
		t.Fatalf("listed credentials = (%+v, %v)", listed, err)
	}
	provider, err := NewEncryptedProvider(repository, cipher)
	if err != nil {
		t.Fatal(err)
	}
	token, err := provider.ResolveLLMCredential(ctx, storedCredentialRef(stored), stored.LLMGateway)
	if err != nil || token.Reveal() != secret {
		t.Fatalf("resolved persisted credential = (%s, %v)", token, err)
	}

	var nonce, ciphertext, policy []byte
	var keyID string
	if err := pool.QueryRow(ctx, `
SELECT nonce, ciphertext, gateway_policy::text::bytea, key_id
FROM llm_credentials WHERE credential_id = $1`, record.CredentialID).Scan(
		&nonce, &ciphertext, &policy, &keyID,
	); err != nil {
		t.Fatal(err)
	}
	if len(nonce) != 12 || bytes.Equal(ciphertext, []byte(secret)) || bytes.Contains(ciphertext, []byte(secret)) ||
		bytes.Contains(policy, []byte(secret)) || keyID != cipher.KeyID() {
		t.Fatalf("unsafe or malformed encrypted row: nonce=%d ciphertext=%d keyID=%q", len(nonce), len(ciphertext), keyID)
	}
	encoded, err := json.Marshal(stored)
	if err != nil || bytes.Contains(encoded, []byte(secret)) || bytes.Contains(encoded, ciphertext) {
		t.Fatalf("unsafe stored record JSON = (%s, %v)", encoded, err)
	}

	conflicting := record
	conflicting.RemoteKeyID = strings.Repeat("b", 64)
	if err := repository.InsertCredential(ctx, conflicting); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting immutable insert error = %v", err)
	}
	if err := repository.ReserveCredentialID(ctx, record.CredentialID, record.CreatedAt); !errors.Is(err, ErrConflict) {
		t.Fatalf("duplicate reservation error = %v", err)
	}
	_, err = pool.Exec(ctx, `UPDATE llm_credentials SET label = 'changed' WHERE credential_id = $1`, record.CredentialID)
	assertCredentialSQLState(t, err, "23514")
	if err := repository.InsertTombstone(ctx, Tombstone{
		CredentialID: record.CredentialID, ActorID: "operator", DeletedAt: record.CreatedAt.Add(time.Hour),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("active/tombstone coexistence error = %v", err)
	}

	other, _ := sealedTestRecord(t, "managed-other", "sk-other")
	other.RemoteKeyID = record.RemoteKeyID
	if err := repository.ReserveCredentialID(ctx, other.CredentialID, other.CreatedAt); err != nil {
		t.Fatal(err)
	}
	if err := repository.InsertCredential(ctx, other); !errors.Is(err, ErrConflict) {
		t.Fatalf("duplicate Gateway remote key error = %v", err)
	}

	operation := Operation{
		OperationID: "create:managed-worker:01", IdempotencyKey: "request-01",
		RequestHash: "sha256:" + strings.Repeat("c", 64), CredentialID: record.CredentialID,
		Kind: OperationCreate, Phase: OperationPrepared,
		Request:   json.RawMessage(`{"credentialId":"managed-worker"}`),
		CreatedAt: record.CreatedAt, UpdatedAt: record.CreatedAt,
	}
	if err := repository.InsertOperation(ctx, operation); err != nil {
		t.Fatalf("insert operation: %v", err)
	}
	if err := repository.InsertOperation(ctx, operation); err != nil {
		t.Fatalf("idempotent operation insert: %v", err)
	}
	conflictingOperation := operation
	conflictingOperation.RequestHash = "sha256:" + strings.Repeat("d", 64)
	if err := repository.InsertOperation(ctx, conflictingOperation); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting idempotency key error = %v", err)
	}
	gotOperation, err := repository.GetOperationByIdempotency(ctx, operation.Kind, operation.IdempotencyKey)
	if err != nil || !operationsEqual(gotOperation, operation) {
		t.Fatalf("stored operation = (%+v, %v)", gotOperation, err)
	}
	prepared, err := repository.ListPreparedOperations(ctx, 10)
	if err != nil || len(prepared) != 1 || prepared[0].OperationID != operation.OperationID {
		t.Fatalf("prepared operations = (%+v, %v)", prepared, err)
	}
	completedAt := record.CreatedAt.Add(time.Minute)
	if err := repository.CompleteOperation(ctx, operation.OperationID, completedAt); err != nil {
		t.Fatalf("complete operation: %v", err)
	}
	if err := repository.CompleteOperation(ctx, operation.OperationID, completedAt.Add(time.Minute)); err != nil {
		t.Fatalf("idempotent operation completion: %v", err)
	}
	if prepared, err := repository.ListPreparedOperations(ctx, 10); err != nil || len(prepared) != 0 {
		t.Fatalf("prepared operations after completion = (%+v, %v)", prepared, err)
	}
	_, err = pool.Exec(ctx, `UPDATE credential_operations SET request_hash = $2 WHERE operation_id = $1`,
		operation.OperationID, "sha256:"+strings.Repeat("e", 64))
	assertCredentialSQLState(t, err, "23514")
	_, err = pool.Exec(ctx, `DELETE FROM credential_operations WHERE operation_id = $1`, operation.OperationID)
	assertCredentialSQLState(t, err, "23514")

	tombstone := Tombstone{
		CredentialID: record.CredentialID, ActorID: "operator", DeletedAt: record.CreatedAt.Add(2 * time.Hour),
	}
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		txRepository := NewRepository(tx)
		if err := txRepository.DeleteCredential(ctx, record.CredentialID); err != nil {
			return err
		}
		return txRepository.InsertTombstone(ctx, tombstone)
	}); err != nil {
		t.Fatalf("atomic active-to-tombstone transition: %v", err)
	}
	if _, err := repository.GetCredential(ctx, record.CredentialID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("deleted credential lookup error = %v", err)
	}
	if err := repository.VerifyActiveKey(ctx, wrongCipher.KeyID()); err != nil {
		t.Fatalf("empty active set rejected valid key fingerprint: %v", err)
	}
	if got, err := repository.GetTombstone(ctx, record.CredentialID); err != nil || got.ActorID != tombstone.ActorID {
		t.Fatalf("stored tombstone = (%+v, %v)", got, err)
	}
	if err := repository.ReserveCredentialID(ctx, record.CredentialID, time.Now()); !errors.Is(err, ErrConflict) {
		t.Fatalf("reuse reservation error = %v", err)
	}
	if err := repository.InsertCredential(ctx, record); !errors.Is(err, ErrConflict) {
		t.Fatalf("reuse active insert error = %v", err)
	}
	_, err = pool.Exec(ctx, `UPDATE llm_credential_tombstones SET actor_id = 'changed' WHERE credential_id = $1`, record.CredentialID)
	assertCredentialSQLState(t, err, "23514")
	_, err = pool.Exec(ctx, `DELETE FROM llm_credential_tombstones WHERE credential_id = $1`, record.CredentialID)
	assertCredentialSQLState(t, err, "23514")
}

func isolatedCredentialPool(t *testing.T, ctx context.Context, databaseURL string) *pgxpool.Pool {
	t.Helper()
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	randomBytes := make([]byte, 8)
	if _, err := rand.Read(randomBytes); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "credential_test_" + hex.EncodeToString(randomBytes)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		if _, err := admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop credential test schema: %v", err)
		}
		admin.Close()
	})
	return pool
}

func storedCredentialRef(record Record) contracts.LLMCredentialRef {
	return contracts.LLMCredentialRef{CredentialID: record.CredentialID}
}

func assertCredentialSQLState(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Fatalf("SQL succeeded, want SQLSTATE %s", want)
	}
	if got := persistencepostgres.SQLState(err); got != want {
		t.Fatalf("SQLSTATE = %q, want %q (error: %v)", got, want, err)
	}
}
