package app

import (
	"context"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestHealthHandler(t *testing.T) {
	t.Parallel()

	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	response := httptest.NewRecorder()
	NewHandler().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusOK)
	}
	if got, want := response.Body.String(), "{\"status\":\"ok\"}\n"; got != want {
		t.Fatalf("body = %q, want %q", got, want)
	}
}

func TestServeStopsAfterContextCancellation(t *testing.T) {
	t.Parallel()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	go func() {
		result <- Serve(ctx, listener, time.Second, logger)
	}()

	client := &http.Client{Timeout: time.Second}
	response, err := client.Get("http://" + listener.Addr().String() + "/readyz")
	if err != nil {
		cancel()
		t.Fatalf("get readiness: %v", err)
	}
	_ = response.Body.Close()
	if response.StatusCode != http.StatusOK {
		cancel()
		t.Fatalf("status = %d, want %d", response.StatusCode, http.StatusOK)
	}

	cancel()
	select {
	case err := <-result:
		if err != nil {
			t.Fatalf("Serve returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Serve did not stop after context cancellation")
	}
}

func TestParseConfig(t *testing.T) {
	t.Parallel()

	env := func(key string) string {
		switch key {
		case "CONTRACTOR_PUBLIC_LISTEN":
			return "127.0.0.1:9000"
		case "CONTRACTOR_DATABASE_URL":
			return "postgres://contractor:secret@database/contractor"
		case "CONTRACTOR_CONFIG_ROOT":
			return "/srv/contractor/configs"
		case "CONTRACTOR_PUBLIC_USER_ID":
			return "local-user"
		case "CONTRACTOR_PUBLIC_BEARER_TOKEN":
			return "private-token"
		}
		return ""
	}
	cfg, err := ParseConfig([]string{"serve", "--shutdown-timeout=2s"}, env)
	if err != nil {
		t.Fatalf("ParseConfig returned error: %v", err)
	}
	if cfg.ListenAddress != "127.0.0.1:9000" {
		t.Fatalf("listen address = %q", cfg.ListenAddress)
	}
	if cfg.ShutdownTimeout != 2*time.Second {
		t.Fatalf("shutdown timeout = %s", cfg.ShutdownTimeout)
	}
	if cfg.DatabaseURL != "postgres://contractor:secret@database/contractor" {
		t.Fatalf("database URL was not read from the shared environment setting")
	}
	if cfg.ConfigRoot != "/srv/contractor/configs" || cfg.PublicUserID != "local-user" ||
		cfg.PublicBearerToken.Reveal() != "private-token" {
		t.Fatalf("public API settings were not parsed: %+v", cfg)
	}
}

func TestProcessHandlerKeepsHealthPublicAndMountsAPI(t *testing.T) {
	t.Parallel()
	api := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusAccepted)
	})
	handler := NewHandler(api)

	health := httptest.NewRecorder()
	handler.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("health status = %d", health.Code)
	}
	public := httptest.NewRecorder()
	handler.ServeHTTP(public, httptest.NewRequest(http.MethodPost, "/v1/runs", nil))
	if public.Code != http.StatusAccepted {
		t.Fatalf("public API status = %d", public.Code)
	}
}

func TestRunCLIValidatesConfigurationWithoutStartingServer(t *testing.T) {
	t.Parallel()

	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	err := RunCLI(
		context.Background(),
		[]string{"config", "validate", "--root", "../../configs"},
		func(string) string { panic("config validate must not read serve environment") },
		logger,
	)
	if err != nil {
		t.Fatalf("RunCLI config validate: %v", err)
	}
}

func TestParseMigrationDatabaseURL(t *testing.T) {
	t.Parallel()

	fromEnvironment, err := parseMigrationDatabaseURL(nil, func(key string) string {
		if key == "CONTRACTOR_DATABASE_URL" {
			return "postgres://environment"
		}
		return ""
	})
	if err != nil || fromEnvironment != "postgres://environment" {
		t.Fatalf("environment database URL = (%q, %v)", fromEnvironment, err)
	}
	fromFlag, err := parseMigrationDatabaseURL(
		[]string{"--database-url", "postgres://flag"},
		func(string) string { return "postgres://environment" },
	)
	if err != nil || fromFlag != "postgres://flag" {
		t.Fatalf("flag database URL = (%q, %v)", fromFlag, err)
	}
	if _, err := parseMigrationDatabaseURL(nil, func(string) string { return "" }); err == nil {
		t.Fatal("missing migration database URL succeeded")
	}
}
