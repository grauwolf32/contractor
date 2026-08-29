package public

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresPublicRunInitializationAndFrozenOutput(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	snapshot, err := config.Load("../../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	runs := runstore.NewPostgresStore(pool)
	nextRunID := "run-public"
	handler, err := NewHandler(Dependencies{
		Config: snapshot, Runs: runs, Artifacts: service,
		Transactions: integrationUnitOfWork{pool: pool},
		BearerToken:  contracts.NewSecretString(testBearerToken), UserID: "user-1",
		NewID:        func(string) (string, error) { return nextRunID, nil },
		NewRequestID: func() (string, error) { return "request-integration", nil },
	})
	if err != nil {
		t.Fatal(err)
	}

	put := authenticatedRequest(http.MethodPut, "/v1/artifacts/projects/source", bytes.NewReader([]byte("original")))
	put.Header.Set("Content-Type", "text/plain")
	putResponse := httptest.NewRecorder()
	handler.ServeHTTP(putResponse, put)
	if putResponse.Code != http.StatusCreated {
		t.Fatalf("PUT source = %d: %s", putResponse.Code, putResponse.Body.String())
	}

	createBody := []byte(`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`)
	create := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(createBody))
	create.Header.Set(idempotencyKeyHeader, "postgres-create-run")
	createResponse := httptest.NewRecorder()
	handler.ServeHTTP(createResponse, create)
	if createResponse.Code != http.StatusAccepted {
		t.Fatalf("POST Run = %d: %s", createResponse.Code, createResponse.Body.String())
	}
	storedRun, err := runs.GetRun(ctx, "run-public")
	if err != nil || storedRun.State != runstore.RunRunning || storedRun.OwnerID != "user-1" || len(storedRun.WorkflowSnapshot) == 0 {
		t.Fatalf("stored Run = (%+v, %v)", storedRun, err)
	}
	nextRunID = "run-duplicate-must-not-exist"
	retry := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(createBody))
	retry.Header.Set(idempotencyKeyHeader, "postgres-create-run")
	retryResponse := httptest.NewRecorder()
	handler.ServeHTTP(retryResponse, retry)
	if retryResponse.Code != http.StatusAccepted || retryResponse.Header().Get("Idempotency-Replayed") != "true" ||
		retryResponse.Body.String() != createResponse.Body.String() {
		t.Fatalf("POST Run retry = %d headers=%v body=%s", retryResponse.Code, retryResponse.Header(), retryResponse.Body.String())
	}
	if _, err := runs.GetRun(ctx, nextRunID); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("response-loss retry created another Run: %v", err)
	}
	differentBody := []byte(`{"workflow":"artifact-copy@1","parameters":{"objective":"different"},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`)
	reused := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(differentBody))
	reused.Header.Set(idempotencyKeyHeader, "postgres-create-run")
	reusedResponse := httptest.NewRecorder()
	handler.ServeHTTP(reusedResponse, reused)
	if reusedResponse.Code != http.StatusConflict {
		t.Fatalf("reused Idempotency-Key = %d %s", reusedResponse.Code, reusedResponse.Body.String())
	}

	user, _ := service.User("user-1")
	current, err := user.Read(ctx, contracts.ArtifactRef{Namespace: "projects", Name: "source"})
	if err != nil {
		t.Fatal(err)
	}
	update := authenticatedRequest(http.MethodPut, "/v1/artifacts/projects/source", bytes.NewReader([]byte("new user revision")))
	update.Header.Set("Content-Type", "text/plain")
	update.Header.Set("If-Match", quotedETag(current.Ref.Revision))
	updateResponse := httptest.NewRecorder()
	handler.ServeHTTP(updateResponse, update)
	if updateResponse.Code != http.StatusOK {
		t.Fatalf("update source = %d: %s", updateResponse.Code, updateResponse.Body.String())
	}
	runArtifacts, _ := service.Run("run-public")
	pinnedInput, err := runArtifacts.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil || string(pinnedInput.Payload.Data) != "original" {
		t.Fatalf("Run input after user update = (%q, %v)", pinnedInput.Payload.Data, err)
	}

	nextRunID = "run-rollback"
	missingBody := []byte(`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"missing"}}}`)
	missing := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(missingBody))
	missing.Header.Set(idempotencyKeyHeader, "postgres-missing-run")
	missingResponse := httptest.NewRecorder()
	handler.ServeHTTP(missingResponse, missing)
	if missingResponse.Code != http.StatusNotFound {
		t.Fatalf("missing source Run = %d: %s", missingResponse.Code, missingResponse.Body.String())
	}
	if _, err := runs.GetRun(ctx, "run-rollback"); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("rolled-back Run lookup error = %v", err)
	}

	intermediate, err := runArtifacts.Write(
		ctx, contracts.ArtifactRef{Namespace: "builder", Name: "result"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("final bytes")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	var outputRef contracts.ArtifactRef
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		txArtifacts := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		bound, err := txArtifacts.BindOutputExact(ctx, "run-public", "result", intermediate.Ref, nil)
		if err != nil {
			return err
		}
		outputRef = bound.TargetRef
		if err := txArtifacts.FreezeRunOutputs(ctx, "run-public"); err != nil {
			return err
		}
		_, err = runstore.NewPostgresStore(tx).TransitionRun(
			ctx, "run-public", runstore.RunRunning, runstore.RunSucceeded,
			runstore.Reason{Code: "outputs_frozen"},
		)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	download := authenticatedRequest(http.MethodGet, "/v1/runs/run-public/outputs/result", bytes.NewReader(nil))
	downloadResponse := httptest.NewRecorder()
	handler.ServeHTTP(downloadResponse, download)
	if downloadResponse.Code != http.StatusOK || downloadResponse.Body.String() != "final bytes" ||
		downloadResponse.Header().Get("ETag") != quotedETag(outputRef.Revision) ||
		downloadResponse.Header().Get("Content-Type") != "text/plain" {
		t.Fatalf("download = status %d, headers %v, body %q", downloadResponse.Code, downloadResponse.Header(), downloadResponse.Body.String())
	}
}

type integrationUnitOfWork struct{ pool *pgxpool.Pool }

func (u integrationUnitOfWork) Do(ctx context.Context, fn func(RunWriter, *artifacts.Service) error) error {
	return persistencepostgres.InTx(ctx, u.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		return fn(runstore.NewPostgresStore(tx), artifacts.NewService(artifacts.NewPostgresRepository(tx)))
	})
}

func isolatedPublicPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_public_test_" + hex.EncodeToString(random)
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
		cleanupContext, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		if _, err := admin.Exec(cleanupContext, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop test schema: %v", err)
		}
		admin.Close()
	})
	return pool
}
