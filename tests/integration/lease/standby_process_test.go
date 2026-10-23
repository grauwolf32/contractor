package lease_test

import (
	"bytes"
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/grauwolf32/contractor/internal/app"
	"github.com/grauwolf32/contractor/internal/auth"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/publicclient"
	"github.com/grauwolf32/contractor/internal/requestid"
)

// standbyServerEnvironment makes the test binary run the Server instead of
// its tests, so the process test starts a real second Server process.
const standbyServerEnvironment = "CONTRACTOR_LEASE_TEST_RUN_SERVER"

// controlPlaneLeaseKey is the single-active advisory lock key of
// internal/app/single_active.go. Should it drift, the second Server acquires
// the lease, fails its startup on the incomplete configuration and exits, so
// the test fails instead of passing vacuously.
const controlPlaneLeaseKey int64 = 0x636f6e7472616374

func TestMain(m *testing.M) {
	if os.Getenv(standbyServerEnvironment) == "1" {
		os.Exit(runServerProcess())
	}
	os.Exit(m.Run())
}

// runServerProcess is the Server's main: the same entry point as
// cmd/contractor-server, stopped by SIGTERM.
func runServerProcess() int {
	logger := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := app.RunCLI(ctx, []string{"serve"}, os.Getenv, logger); err != nil {
		logger.Error("contractor server stopped", "error", err)
		return 1
	}
	return 0
}

// A Server that starts while another Server holds the database's lease stands
// by. Its public API answers every /v1 request, the event upgrade included,
// with the retryable 503 server_standby error envelope instead of a plain-text
// 404 that clients read as a missing resource (PER-R3), and the CLI client
// classifies that answer as transient.
func TestStandbyServerAnswersPublicAPIWithRetryableEnvelope(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	serverDatabaseURL := isolatedLeaseDatabase(t, ctx, databaseURL)

	holder, err := pgx.Connect(ctx, serverDatabaseURL)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = holder.Close(context.Background()) })
	if _, err := holder.Exec(ctx, `SELECT pg_advisory_lock($1)`, controlPlaneLeaseKey); err != nil {
		t.Fatal(err)
	}

	root := t.TempDir()
	publicAddress := freeLoopbackAddress(t)
	server := startServerProcess(t, map[string]string{
		"CONTRACTOR_DATABASE_URL":            serverDatabaseURL,
		"CONTRACTOR_PUBLIC_LISTEN":           publicAddress,
		"CONTRACTOR_PRIVATE_LISTEN":          freeLoopbackAddress(t),
		"CONTRACTOR_PRIVATE_URL":             "https://127.0.0.1:1",
		"CONTRACTOR_PUBLIC_BEARER_TOKEN":     "standby-public-token",
		"CONTRACTOR_LOCAL_AUTH_FILE":         writeLocalAuth(t, root),
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
		"CONTRACTOR_OPERATOR_CONFIG_ROOT":    filepath.Join(root, "operator-configs"),
		"CONTRACTOR_MANAGED_CONFIG_ROOT":     filepath.Join(root, "managed-configs"),
		"CONTRACTOR_ARTIFACT_BLOB_PATH":      filepath.Join(root, "blobs"),
		"CONTRACTOR_CA_FILE":                 filepath.Join(root, "absent-ca.pem"),
		"CONTRACTOR_CONTROL_PLANE_CERT_FILE": filepath.Join(root, "absent-control-plane.pem"),
		"CONTRACTOR_CONTROL_PLANE_KEY_FILE":  filepath.Join(root, "absent-control-plane-key.pem"),
	})

	baseURL := "http://" + publicAddress
	client := &http.Client{Timeout: 2 * time.Second}
	waitForStandby(t, ctx, server, client, baseURL+"/readyz")

	runs, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/v1/runs", nil)
	if err != nil {
		t.Fatal(err)
	}
	runs.Header.Set("Authorization", "Bearer standby-public-token")
	upgrade, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/v1/events/ws", nil)
	if err != nil {
		t.Fatal(err)
	}
	for name, value := range map[string]string{
		"Connection": "Upgrade", "Upgrade": "websocket", "Sec-WebSocket-Version": "13",
		"Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==", "Sec-WebSocket-Protocol": "contractor.events.v1",
	} {
		upgrade.Header.Set(name, value)
	}
	for _, request := range []*http.Request{runs, upgrade} {
		response, err := client.Do(request)
		if err != nil {
			t.Fatalf("standby %s: %v", request.URL.Path, err)
		}
		body, err := io.ReadAll(response.Body)
		_ = response.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		var envelope struct {
			Code      string `json:"code"`
			Message   string `json:"message"`
			Retryable bool   `json:"retryable"`
			RequestID string `json:"requestId"`
		}
		decoder := json.NewDecoder(bytes.NewReader(body))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&envelope); err != nil {
			t.Fatalf("standby %s = %d %q: %v", request.URL.Path, response.StatusCode, body, err)
		}
		requestID := response.Header.Get(requestid.Header)
		if response.StatusCode != http.StatusServiceUnavailable ||
			response.Header.Get("Content-Type") != "application/json" ||
			response.Header.Get("Cache-Control") != "no-store" ||
			response.Header.Get(publicclient.APIVersionHeader) != publicclient.APIVersion ||
			envelope.Code != "server_standby" || !envelope.Retryable || envelope.Message == "" ||
			!requestid.Valid(requestID) || envelope.RequestID != requestID {
			t.Fatalf("standby %s = %d headers=%v body=%s", request.URL.Path, response.StatusCode, response.Header, body)
		}
		if decoded := publicclient.DecodeError(response.StatusCode, body); !publicclient.IsTransient(decoded) {
			t.Fatalf("CLI client reads the standby answer as permanent: %v", decoded)
		}
	}
	if exited, err := server.exited(); exited {
		t.Fatalf("standby Server exited: %v\n%s", err, server.output())
	}
}

func isolatedLeaseDatabase(t *testing.T, ctx context.Context, databaseURL string) string {
	t.Helper()
	admin, err := pgx.Connect(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	// Cleanups run last-in first-out: the database is dropped before the
	// admin connection closes.
	t.Cleanup(func() { _ = admin.Close(context.Background()) })
	random := make([]byte, 6)
	if _, err := cryptorand.Read(random); err != nil {
		t.Fatal(err)
	}
	// Advisory locks are scoped per database, so a database of its own keeps
	// the held lease away from every other test of the cluster.
	name := "contractor_standby_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{name}.Sanitize()
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+identifier); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, "DROP DATABASE IF EXISTS "+identifier+" WITH (FORCE)"); err != nil {
			t.Errorf("drop standby database: %v", err)
		}
	})
	parsed, err := url.Parse(databaseURL)
	if err != nil || parsed.Scheme != "postgres" && parsed.Scheme != "postgresql" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL must be a PostgreSQL URL")
	}
	parsed.Path = "/" + name
	pool, err := pgxpool.New(ctx, parsed.String())
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return parsed.String()
}

func writeLocalAuth(t *testing.T, root string) string {
	t.Helper()
	hash, err := auth.HashPassword([]byte("contractor standby local password"))
	if err != nil {
		t.Fatal(err)
	}
	document, err := auth.BootstrapYAML("standby-user", "admin", hash)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "local-auth.yaml")
	if err := os.WriteFile(path, document, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func freeLoopbackAddress(t *testing.T) string {
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

// serverProcess is the second Server, running in a child copy of the test
// binary; its bounded output is reported when it fails.
type serverProcess struct {
	cmd    *exec.Cmd
	done   chan struct{}
	err    error
	mu     sync.Mutex
	buffer bytes.Buffer
}

func (p *serverProcess) Write(data []byte) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if remaining := 64<<10 - p.buffer.Len(); remaining > 0 {
		p.buffer.Write(data[:min(len(data), remaining)])
	}
	return len(data), nil
}

func (p *serverProcess) output() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.buffer.String()
}

func (p *serverProcess) exited() (bool, error) {
	select {
	case <-p.done:
		return true, p.err
	default:
		return false, nil
	}
}

func startServerProcess(t *testing.T, environment map[string]string) *serverProcess {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	process := &serverProcess{done: make(chan struct{})}
	process.cmd = exec.Command(executable)
	// The Server reads only the configuration given here, never a
	// CONTRACTOR_ value of the test's own environment.
	for _, entry := range os.Environ() {
		if !strings.HasPrefix(entry, "CONTRACTOR_") {
			process.cmd.Env = append(process.cmd.Env, entry)
		}
	}
	process.cmd.Env = append(process.cmd.Env, standbyServerEnvironment+"=1")
	for name, value := range environment {
		process.cmd.Env = append(process.cmd.Env, name+"="+value)
	}
	process.cmd.Stdout = process
	process.cmd.Stderr = process
	if err := process.cmd.Start(); err != nil {
		t.Fatal(err)
	}
	go func() {
		process.err = process.cmd.Wait()
		close(process.done)
	}()
	t.Cleanup(func() {
		if exited, _ := process.exited(); exited {
			return
		}
		_ = process.cmd.Process.Signal(syscall.SIGTERM)
		select {
		case <-process.done:
			if process.err != nil {
				t.Errorf("standby Server did not stop cleanly: %v\n%s", process.err, process.output())
			}
		case <-time.After(10 * time.Second):
			_ = process.cmd.Process.Kill()
			<-process.done
			t.Errorf("standby Server ignored SIGTERM\n%s", process.output())
		}
	})
	return process
}

// waitForStandby waits until the Server's listener reports 503 readiness,
// which only a standby does before it holds the lease.
func waitForStandby(t *testing.T, ctx context.Context, server *serverProcess, client *http.Client, target string) {
	t.Helper()
	for {
		if exited, err := server.exited(); exited {
			t.Fatalf("Server exited before standing by: %v\n%s", err, server.output())
		}
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
		if err != nil {
			t.Fatal(err)
		}
		response, err := client.Do(request)
		if err == nil {
			_ = response.Body.Close()
			if response.StatusCode != http.StatusServiceUnavailable {
				t.Fatalf("standby readiness = %d, want 503", response.StatusCode)
			}
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("Server did not stand by: %v\n%s", errors.Join(ctx.Err(), err), server.output())
		case <-time.After(50 * time.Millisecond):
		}
	}
}
