package credentials

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestRuntimeCredentialPostgresLifecycleReplayAndSecretBoundary(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedCredentialPool(t, ctx, databaseURL)
	barrier := NewLifecycleBarrier()
	repository := NewRuntimeCredentialRepository(pool)
	cipher, err := NewTokenCipher(bytes.Repeat([]byte{0x71}, 32))
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewRuntimeCredentialService(RuntimeCredentialServiceOptions{
		Pool: pool, Cipher: cipher, Usage: repository, Barrier: barrier,
		Now: func() time.Time { return time.Date(2026, 8, 31, 18, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}

	otlpSecret := "Bearer otlp-secret-never-persist"
	otlp, _ := NewOTLPHeadersCredential(map[string]string{"Authorization": otlpSecret, "X-Tenant": "tenant-a"})
	basic, _ := NewHTTPProxyBasicCredential("proxy-user", "basic-secret-never-persist")
	bearer, _ := NewHTTPProxyBearerCredential("bearer-secret-never-persist")
	tests := []struct {
		id       string
		key      string
		material RuntimeCredentialMaterial
	}{
		{id: "otel-auth", key: "runtime-create-otel", material: otlp},
		{id: "proxy-basic", key: "runtime-create-basic", material: basic},
		{id: "proxy-bearer", key: "runtime-create-bearer", material: bearer},
	}
	for _, test := range tests {
		request := RuntimeCredentialCreateRequest{
			CredentialID: test.id, Material: test.material,
			IdempotencyKey: test.key, ActorID: "operator",
		}
		created, err := service.Create(ctx, request)
		if err != nil || created.Replayed || created.Credential.CredentialID != test.id || created.Credential.Kind != test.material.Kind() {
			t.Fatalf("create %s = (%+v, %v)", test.id, created, err)
		}
		encoded, err := json.Marshal(created.Credential)
		if err != nil || bytes.Contains(encoded, []byte("secret")) || bytes.Contains(encoded, []byte("cipher")) {
			t.Fatalf("unsafe metadata for %s = (%s, %v)", test.id, encoded, err)
		}
		replayed, err := service.Create(ctx, request)
		if err != nil || !replayed.Replayed ||
			replayed.Credential.CredentialID != created.Credential.CredentialID ||
			replayed.Credential.Kind != created.Credential.Kind ||
			replayed.Credential.CreatedBy != created.Credential.CreatedBy ||
			!replayed.Credential.CreatedAt.Equal(created.Credential.CreatedAt) {
			t.Fatalf("replay %s = (%+v, %v)", test.id, replayed, err)
		}
		want := append([]byte(nil), test.material.canonical...)
		if err := service.Use(ctx, test.id, []string{string(test.material.Kind())}, func(resolved *RuntimeCredentialMaterial) error {
			return resolved.WithPlaintext(func(kind RuntimeCredentialKind, plaintext []byte) error {
				if kind != test.material.Kind() || !bytes.Equal(plaintext, want) {
					return errors.New("resolved Runtime credential differs")
				}
				return nil
			})
		}); err != nil {
			t.Fatalf("use %s: %v", test.id, err)
		}
	}
	if count, err := repository.CountActive(ctx); err != nil || count != 3 {
		t.Fatalf("active Runtime credential count = (%d, %v)", count, err)
	}
	listed, err := service.List(ctx, "", 10)
	if err != nil || len(listed) != 3 || listed[0].CredentialID != "otel-auth" || listed[1].CredentialID != "proxy-basic" || listed[2].CredentialID != "proxy-bearer" {
		t.Fatalf("safe Runtime credential list = (%+v, %v)", listed, err)
	}
	if err := repository.VerifyActiveKey(ctx, cipher.KeyID()); err != nil {
		t.Fatalf("verify Runtime credential key: %v", err)
	}
	wrongCipher, _ := NewTokenCipher(bytes.Repeat([]byte{0x72}, 32))
	if err := repository.VerifyActiveKey(ctx, wrongCipher.KeyID()); !errors.Is(err, ErrKeyUnavailable) {
		t.Fatalf("wrong Runtime credential key error = %v", err)
	}
	if err := service.ValidateRuntimeCredential(ctx, "otel-auth", string(RuntimeCredentialProxyBearer)); !errors.Is(err, ErrRuntimeCredentialNotFound) {
		t.Fatalf("wrong kind validation error = %v", err)
	}
	consumerErr := service.Use(ctx, "proxy-basic", []string{string(RuntimeCredentialProxyBasic)}, func(*RuntimeCredentialMaterial) error {
		return errors.New("consumer accidentally included basic-secret-never-persist")
	})
	if consumerErr == nil || strings.Contains(consumerErr.Error(), "basic-secret-never-persist") {
		t.Fatalf("unsafe consumer error = %v", consumerErr)
	}
	changed, _ := NewOTLPHeadersCredential(map[string]string{"Authorization": "changed-secret"})
	if _, err := service.Create(ctx, RuntimeCredentialCreateRequest{
		CredentialID: "otel-auth", Material: changed,
		IdempotencyKey: "runtime-create-otel", ActorID: "operator",
	}); !errors.Is(err, ErrRuntimeCredentialConflict) {
		t.Fatalf("changed replay error = %v", err)
	}

	for _, secret := range []struct {
		credentialID string
		value        string
	}{
		{credentialID: "otel-auth", value: otlpSecret},
		{credentialID: "proxy-basic", value: "basic-secret-never-persist"},
		{credentialID: "proxy-bearer", value: "bearer-secret-never-persist"},
	} {
		var ciphertext, requestMAC []byte
		if err := pool.QueryRow(ctx, `
SELECT c.ciphertext, p.request_mac
FROM runtime_credentials c
JOIN runtime_credential_creations p ON p.credential_id = c.credential_id
WHERE c.credential_id = $1`, secret.credentialID).Scan(&ciphertext, &requestMAC); err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(ciphertext, []byte(secret.value)) || bytes.Contains(requestMAC, []byte(secret.value)) {
			t.Fatalf("database state contains plaintext secret for %q", secret.credentialID)
		}
	}
	publicDigest := sha256.Sum256(otlp.canonical)
	var storedMAC []byte
	if err := pool.QueryRow(ctx, `SELECT request_mac FROM runtime_credential_creations WHERE credential_id = 'otel-auth'`).Scan(&storedMAC); err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(storedMAC, publicDigest[:]) {
		t.Fatal("stored replay authenticator is a public deterministic secret hash")
	}
	_, err = pool.Exec(ctx, `
INSERT INTO runtime_credentials (
    credential_id, credential_kind, encryption_schema_version,
    key_id, nonce, ciphertext, created_by, created_at
) VALUES (
    'corrupt-runtime', 'http-proxy-bearer@1',
    'contractor.runtime-credentials/v1', $1, $2, $3, 'operator', clock_timestamp()
)`, cipher.KeyID(), bytes.Repeat([]byte{0x01}, 12), bytes.Repeat([]byte{0x02}, 17))
	if err != nil {
		t.Fatal(err)
	}
	if err := service.Use(ctx, "corrupt-runtime", []string{string(RuntimeCredentialProxyBearer)}, func(*RuntimeCredentialMaterial) error {
		return nil
	}); !errors.Is(err, ErrCrypto) {
		t.Fatalf("corrupted Runtime credential error = %v", err)
	}

	deleted, err := service.Delete(ctx, "proxy-bearer", "operator")
	if err != nil || deleted.Replayed {
		t.Fatalf("delete unreferenced credential = (%+v, %v)", deleted, err)
	}
	replayedDelete, err := service.Delete(ctx, "proxy-bearer", "other-operator")
	if err != nil || !replayedDelete.Replayed {
		t.Fatalf("replay delete = (%+v, %v)", replayedDelete, err)
	}
	if _, err := service.Get(ctx, "proxy-bearer"); !errors.Is(err, ErrRuntimeCredentialNotFound) {
		t.Fatalf("deleted active lookup error = %v", err)
	}
	if _, err := repository.GetAnyRecord(ctx, "proxy-bearer"); err != nil {
		t.Fatalf("ambiguous-delete retention row missing: %v", err)
	}
	if stored, err := repository.CountStored(ctx); err != nil || stored != 4 {
		t.Fatalf("stored Runtime credential count after tombstone = (%d, %v)", stored, err)
	}
	if err := repository.VerifyStoredKey(ctx, wrongCipher.KeyID()); !errors.Is(err, ErrKeyUnavailable) {
		t.Fatalf("stored tombstone key mismatch error = %v", err)
	}
	if _, err := service.Create(ctx, RuntimeCredentialCreateRequest{
		CredentialID: "proxy-bearer", Material: bearer,
		IdempotencyKey: "runtime-recreate-bearer", ActorID: "operator",
	}); !errors.Is(err, ErrRuntimeCredentialConflict) {
		t.Fatalf("deleted ID recreation error = %v", err)
	}
	if replay, err := service.Create(ctx, RuntimeCredentialCreateRequest{
		CredentialID: "proxy-bearer", Material: bearer,
		IdempotencyKey: "runtime-create-bearer", ActorID: "operator",
	}); err != nil || !replay.Replayed {
		t.Fatalf("original create replay after delete = (%+v, %v)", replay, err)
	}
	for _, credentialID := range []string{"otel-auth", "proxy-basic"} {
		if result, err := service.Delete(ctx, credentialID, "operator"); err != nil || result.Replayed {
			t.Fatalf("delete %s = (%+v, %v)", credentialID, result, err)
		}
		if _, err := service.Get(ctx, credentialID); !errors.Is(err, ErrRuntimeCredentialNotFound) {
			t.Fatalf("deleted %s lookup error = %v", credentialID, err)
		}
	}

	_, err = pool.Exec(ctx, `UPDATE runtime_credentials SET created_by = 'changed' WHERE credential_id = 'otel-auth'`)
	assertRuntimeCredentialSQLState(t, err, "23514")
	_, err = pool.Exec(ctx, `DELETE FROM runtime_credential_tombstones WHERE credential_id = 'proxy-bearer'`)
	assertRuntimeCredentialSQLState(t, err, "23514")
}

func TestRuntimeCredentialDeleteSerializesWithRuntimeConfigBindings(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedCredentialPool(t, ctx, databaseURL)
	barrier := NewLifecycleBarrier()
	repository := NewRuntimeCredentialRepository(pool)
	cipher, _ := NewTokenCipher(bytes.Repeat([]byte{0x73}, 32))
	now := time.Date(2026, 8, 31, 19, 0, 0, 0, time.UTC)
	service, err := NewRuntimeCredentialService(RuntimeCredentialServiceOptions{
		Pool: pool, Cipher: cipher, Usage: repository, Barrier: barrier, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool, RuntimeCredentials: service, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	bindings, err := runtimeconfig.NewBindingService(pool, service)
	if err != nil {
		t.Fatal(err)
	}

	createCredential := func(id string) {
		t.Helper()
		material, _ := NewOTLPHeadersCredential(map[string]string{"Authorization": "Bearer " + id + "-secret"})
		if _, err := service.Create(ctx, RuntimeCredentialCreateRequest{
			CredentialID: id, Material: material, IdempotencyKey: "create-" + id, ActorID: "operator",
		}); err != nil {
			t.Fatal(err)
		}
	}
	publishConfig := func(name, credentialID string) runtimeconfig.Ref {
		t.Helper()
		document := []byte(fmt.Sprintf(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":%q,"version":"1"},
  "spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces","credential":%q}}}
}`, name, credentialID))
		published, err := publisher.Publish(ctx, document, "publish-"+name, "operator")
		if err != nil {
			t.Fatal(err)
		}
		return published.Version.Ref
	}
	wrongKindMaterial, _ := NewHTTPProxyBasicCredential("proxy-user", "proxy-secret")
	if _, err := service.Create(ctx, RuntimeCredentialCreateRequest{
		CredentialID: "wrong-otel-kind", Material: wrongKindMaterial,
		IdempotencyKey: "create-wrong-otel-kind", ActorID: "operator",
	}); err != nil {
		t.Fatal(err)
	}
	wrongKindDocument := []byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"wrong-otel-kind","version":"1"},
  "spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces","credential":"wrong-otel-kind"}}}
}`)
	if _, err := publisher.Publish(ctx, wrongKindDocument, "publish-wrong-otel-kind", "operator"); !errors.Is(err, runtimeconfig.ErrInvalid) {
		t.Fatalf("wrong-kind RuntimeConfig publication error = %v", err)
	}
	if _, err := runtimeconfig.NewRepository(pool).GetVersion(ctx, "wrong-otel-kind", "1"); !errors.Is(err, runtimeconfig.ErrNotFound) {
		t.Fatalf("wrong-kind RuntimeConfig was persisted: %v", err)
	}

	createCredential("active-debug")
	activeRef := publishConfig("active-debug", "active-debug")
	activeBinding, err := bindings.Create(ctx, "debug", activeRef, "operator", now)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Delete(ctx, "active-debug", "operator"); !errors.Is(err, ErrRuntimeCredentialInUse) {
		t.Fatalf("delete active binding error = %v", err)
	}
	if err := bindings.Delete(ctx, activeBinding.Label, activeBinding.Revision); err != nil {
		t.Fatal(err)
	}
	if _, err := service.Delete(ctx, "active-debug", "operator"); err != nil {
		t.Fatalf("delete after unbind: %v", err)
	}
	if _, err := bindings.Create(ctx, "stale-debug", activeRef, "operator", now.Add(time.Second)); !errors.Is(err, runtimeconfig.ErrInvalid) {
		t.Fatalf("binding a config with a deleted credential error = %v", err)
	}

	createCredential("default-debug")
	defaultRef := publishConfig("default-debug", "default-debug")
	defaultBinding, err := runtimeconfig.NewRepository(pool).GetBinding(ctx, runtimeconfig.DefaultLabel)
	if err != nil {
		t.Fatal(err)
	}
	defaultBinding, err = bindings.Rebind(
		ctx, runtimeconfig.DefaultLabel, defaultBinding.Revision, defaultRef, "operator", now.Add(2*time.Second),
	)
	if err != nil {
		t.Fatalf("bind default RuntimeConfig: %v", err)
	}
	if _, err := service.Delete(ctx, "default-debug", "operator"); !errors.Is(err, ErrRuntimeCredentialInUse) {
		t.Fatalf("delete credential referenced by default binding error = %v", err)
	}
	builtInRef := runtimeconfig.Ref{
		Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion, Digest: runtimeconfig.BuiltInDigest,
	}
	if _, err := bindings.Rebind(
		ctx, runtimeconfig.DefaultLabel, defaultBinding.Revision, builtInRef, "operator", now.Add(3*time.Second),
	); err != nil {
		t.Fatalf("restore default RuntimeConfig: %v", err)
	}
	if _, err := service.Delete(ctx, "default-debug", "operator"); err != nil {
		t.Fatalf("delete after restoring default binding: %v", err)
	}

	createCredential("run-debug")
	runRef := publishConfig("run-debug", "run-debug")
	runBinding, err := bindings.Create(ctx, "run-debug", runRef, "operator", now.Add(4*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if err := service.WithCredentialReferences(ctx, func() error {
		tx, txErr := pool.Begin(ctx)
		if txErr != nil {
			return txErr
		}
		defer func() { _ = tx.Rollback(ctx) }()
		store := runstore.NewPostgresStore(tx)
		pinned, pinErr := store.PinRuntimeLabels(ctx, []string{"run-debug"}, nil)
		if pinErr != nil {
			return pinErr
		}
		if _, createErr := store.CreateRun(ctx, runstore.CreateRunParams{
			RunID: "run-runtime-credential", OwnerID: "user", WorkflowName: "workflow", WorkflowVersion: "1",
			WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
			Parameters: map[string]string{}, RuntimeConfig: pinned,
		}); createErr != nil {
			return createErr
		}
		return tx.Commit(ctx)
	}); err != nil {
		t.Fatal(err)
	}
	if err := bindings.Delete(ctx, runBinding.Label, runBinding.Revision); err != nil {
		t.Fatal(err)
	}
	if _, err := service.Delete(ctx, "run-debug", "operator"); !errors.Is(err, ErrRuntimeCredentialInUse) {
		t.Fatalf("delete credential pinned only by Run error = %v", err)
	} else {
		var inUse *RuntimeCredentialInUseError
		if !errors.As(err, &inUse) || len(inUse.Usage.RunIDs) != 1 || inUse.Usage.RunIDs[0] != "run-runtime-credential" {
			t.Fatalf("Run credential usage = %+v", inUse)
		}
	}
	store := runstore.NewPostgresStore(pool)
	if _, err := store.TransitionRun(
		ctx, "run-runtime-credential", runstore.RunInitializing, runstore.RunFailed,
		runstore.Reason{Code: "test_complete"},
	); err != nil {
		t.Fatal(err)
	}
	if _, err := service.Delete(ctx, "run-debug", "operator"); err != nil {
		t.Fatalf("delete Runtime credential after Run terminal: %v", err)
	}

	createCredential("race-debug")
	raceRef := publishConfig("race-debug", "race-debug")
	start := make(chan struct{})
	var binding runtimeconfig.Binding
	var bindingErr, deleteErr error
	var wait sync.WaitGroup
	wait.Add(2)
	go func() {
		defer wait.Done()
		<-start
		binding, bindingErr = bindings.Create(ctx, "race", raceRef, "operator", now.Add(time.Minute))
	}()
	go func() {
		defer wait.Done()
		<-start
		_, deleteErr = service.Delete(ctx, "race-debug", "operator")
	}()
	close(start)
	wait.Wait()
	switch {
	case bindingErr == nil && errors.Is(deleteErr, ErrRuntimeCredentialInUse):
		if _, err := service.Get(ctx, "race-debug"); err != nil {
			t.Fatalf("binding won but credential is inactive: %v", err)
		}
		if err := bindings.Delete(ctx, binding.Label, binding.Revision); err != nil {
			t.Fatal(err)
		}
	case deleteErr == nil && errors.Is(bindingErr, runtimeconfig.ErrInvalid):
		if _, err := runtimeconfig.NewRepository(pool).GetBinding(ctx, "race"); !errors.Is(err, runtimeconfig.ErrNotFound) {
			t.Fatalf("delete won but dangling binding exists: %v", err)
		}
	default:
		t.Fatalf("binding/delete race outcomes = (binding=%v, delete=%v)", bindingErr, deleteErr)
	}
}

func assertRuntimeCredentialSQLState(t *testing.T, err error, want string) {
	t.Helper()
	var postgresError *pgconn.PgError
	if !errors.As(err, &postgresError) || postgresError.Code != want {
		t.Fatalf("PostgreSQL error = %v, want SQLSTATE %s", err, want)
	}
}
