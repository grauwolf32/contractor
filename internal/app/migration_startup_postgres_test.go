package app

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	neturl "net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/localpki"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresServerStartupRequiresMatchingMigrationLedger(t *testing.T) {
	baseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if baseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	databaseURL := isolatedStartupDatabase(t, ctx, baseURL)
	pool, err := persistencepostgres.OpenPool(ctx, databaseURL, persistencepostgres.PoolOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	env, publicAddress := migrationStartupEnvironment(t, databaseURL)
	getenv := func(key string) string { return env[key] }
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	assertStartupError := func(want error, detail string, args ...string) {
		t.Helper()
		command := append([]string{"serve", "--shutdown-timeout=2s"}, args...)
		err := RunCLI(ctx, command, getenv, logger)
		if !errors.Is(err, want) || !strings.Contains(err.Error(), detail) {
			t.Fatalf("serve with %s ledger: %v, want %v with %q", detail, err, want, detail)
		}
	}

	occupied, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer occupied.Close()
	assertStartupError(persistencepostgres.ErrMigrationsPending, "contractor server migrate",
		"--pprof=true", "--pprof-listen="+occupied.Addr().String())
	if _, err := pool.Exec(ctx, `SELECT 1 FROM contractor_schema_migrations`); err == nil {
		t.Fatal("serve against an empty database created the migration ledger")
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	var latestVersion int64
	var latestName string
	var latestChecksum []byte
	if err := pool.QueryRow(ctx, `SELECT version,name,checksum FROM contractor_schema_migrations ORDER BY version DESC LIMIT 1`).
		Scan(&latestVersion, &latestName, &latestChecksum); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM contractor_schema_migrations WHERE version=$1`, latestVersion); err != nil {
		t.Fatal(err)
	}
	assertStartupError(persistencepostgres.ErrMigrationsPending, fmt.Sprintf("%06d", latestVersion))
	if _, err := pool.Exec(ctx, `INSERT INTO contractor_schema_migrations(version,name,checksum) VALUES ($1,$2,$3)`,
		latestVersion, latestName, latestChecksum); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO contractor_schema_migrations(version,name,checksum) VALUES (999999,'999999_future.sql',decode(repeat('00',32),'hex'))`); err != nil {
		t.Fatal(err)
	}
	assertStartupError(persistencepostgres.ErrMigrationDrift, "999999")
	if _, err := pool.Exec(ctx, `DELETE FROM contractor_schema_migrations WHERE version=999999`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET name='renamed.sql' WHERE version=$1`, latestVersion); err != nil {
		t.Fatal(err)
	}
	assertStartupError(persistencepostgres.ErrMigrationDrift, fmt.Sprintf("%06d", latestVersion))
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET name=$2,checksum=decode(repeat('00',32),'hex') WHERE version=$1`,
		latestVersion, latestName); err != nil {
		t.Fatal(err)
	}
	assertStartupError(persistencepostgres.ErrMigrationDrift, fmt.Sprintf("%06d", latestVersion))
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET checksum=$2 WHERE version=$1`, latestVersion, latestChecksum); err != nil {
		t.Fatal(err)
	}

	serveCtx, stop := context.WithCancel(ctx)
	defer stop()
	served := make(chan error, 1)
	go func() { served <- RunCLI(serveCtx, []string{"serve", "--shutdown-timeout=2s"}, getenv, logger) }()
	client := &http.Client{Timeout: 500 * time.Millisecond}
	ready := false
	for !ready {
		select {
		case err := <-served:
			t.Fatalf("matching schema failed startup: %v", err)
		case <-ctx.Done():
			t.Fatalf("wait for matching Server readiness: %v", ctx.Err())
		default:
		}
		response, err := client.Get("http://" + publicAddress + "/readyz")
		if err == nil {
			_ = response.Body.Close()
			ready = response.StatusCode == http.StatusOK
		}
		if !ready {
			time.Sleep(50 * time.Millisecond)
		}
	}
	stop()
	select {
	case err := <-served:
		if err != nil {
			t.Fatalf("matching Server shutdown: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("matching Server did not stop")
	}
}

func isolatedStartupDatabase(t *testing.T, ctx context.Context, databaseURL string) string {
	t.Helper()
	parsed, err := neturl.Parse(databaseURL)
	if err != nil || parsed.Scheme != "postgres" && parsed.Scheme != "postgresql" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL must be a PostgreSQL URL")
	}
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	var suffix [8]byte
	if _, err := rand.Read(suffix[:]); err != nil {
		t.Fatal(err)
	}
	schema := "migration_startup_" + hex.EncodeToString(suffix[:])
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop migration startup schema: %v", err)
		}
		admin.Close()
	})
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	return parsed.String()
}

func migrationStartupEnvironment(t *testing.T, databaseURL string) (map[string]string, string) {
	t.Helper()
	root := t.TempDir()
	hash, err := auth.HashPassword([]byte("migration startup password"))
	if err != nil {
		t.Fatal(err)
	}
	bootstrap, err := auth.BootstrapYAML("migration-startup-user", "admin", hash)
	if err != nil {
		t.Fatal(err)
	}
	authPath := filepath.Join(root, "local-auth.yaml")
	if err := os.WriteFile(authPath, bootstrap, 0o600); err != nil {
		t.Fatal(err)
	}
	masterKeyPath := filepath.Join(root, "credential-master-key")
	masterKey := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x5a}, 32))
	if err := os.WriteFile(masterKeyPath, []byte(masterKey), 0o600); err != nil {
		t.Fatal(err)
	}
	pkiRoot := filepath.Join(root, "pki")
	generator := localpki.Generator{}
	ca, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatal(err)
	}
	controlPlane, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{
		LeafOptions: localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}},
		URI:         "urn:contractor:control-plane:migration-startup-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	publicAddress := startupFreeAddress(t)
	privateAddress := startupFreeAddress(t)
	configRoot, err := filepath.Abs("../../testdata/configs")
	if err != nil {
		t.Fatal(err)
	}
	return map[string]string{
		"CONTRACTOR_DATABASE_URL":               databaseURL,
		"CONTRACTOR_OPERATOR_CONFIG_ROOT":       configRoot,
		"CONTRACTOR_MANAGED_CONFIG_ROOT":        filepath.Join(root, "managed"),
		"CONTRACTOR_PUBLIC_LISTEN":              publicAddress,
		"CONTRACTOR_PRIVATE_LISTEN":             privateAddress,
		"CONTRACTOR_PRIVATE_URL":                "https://" + privateAddress,
		"CONTRACTOR_CA_FILE":                    ca.Certificate,
		"CONTRACTOR_CONTROL_PLANE_CERT_FILE":    controlPlane.Certificate,
		"CONTRACTOR_CONTROL_PLANE_KEY_FILE":     controlPlane.PrivateKey,
		"CONTRACTOR_LLM_GATEWAY_TOKEN":          "migration-fixture-token",
		"CONTRACTOR_PUBLIC_BEARER_TOKEN":        "migration-public-token",
		"CONTRACTOR_LOCAL_AUTH_FILE":            authPath,
		"CONTRACTOR_BROWSER_ORIGINS":            "https://ui.contractor.invalid",
		"CONTRACTOR_CREDENTIAL_MASTER_KEY_FILE": masterKeyPath,
	}, publicAddress
}

func startupFreeAddress(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return address
}
