package app

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auth"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
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

func TestServeSystemStopsPublicThenSchedulerThenPrivate(t *testing.T) {
	t.Parallel()
	publicListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	privateListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		_ = publicListener.Close()
		t.Fatal(err)
	}
	publicURL := "http://" + publicListener.Addr().String()
	privateURL := "http://" + privateListener.Addr().String()
	runner := &shutdownProbeRunner{
		publicURL: publicURL, privateURL: privateURL, checked: make(chan error, 1),
	}
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	go func() {
		result <- ServeSystem(
			ctx, publicListener, privateListener, 2*time.Second, logger,
			http.HandlerFunc(writeHealthy), http.HandlerFunc(writeHealthy), runner,
		)
	}()

	client := &http.Client{Timeout: time.Second}
	response, err := client.Get(publicURL + "/readyz")
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	_ = response.Body.Close()
	cancel()
	select {
	case err := <-result:
		if err != nil {
			t.Fatalf("ServeSystem: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("ServeSystem did not stop")
	}
	if err := <-runner.checked; err != nil {
		t.Fatal(err)
	}
}

type shutdownProbeRunner struct {
	publicURL  string
	privateURL string
	checked    chan error
}

func (r *shutdownProbeRunner) Run(ctx context.Context) error {
	<-ctx.Done()
	client := &http.Client{
		Timeout:   300 * time.Millisecond,
		Transport: &http.Transport{DisableKeepAlives: true},
	}
	publicResponse, publicErr := client.Get(r.publicURL + "/healthz")
	if publicResponse != nil {
		_ = publicResponse.Body.Close()
	}
	if publicErr == nil {
		r.checked <- errors.New("public listener was still accepting after Scheduler cancellation")
		return nil
	}
	privateResponse, privateErr := client.Get(r.privateURL + "/healthz")
	if privateErr != nil {
		r.checked <- errors.New("private listener stopped before Scheduler")
		return nil
	}
	_ = privateResponse.Body.Close()
	if privateResponse.StatusCode != http.StatusOK {
		r.checked <- errors.New("private listener returned an unexpected status")
		return nil
	}
	r.checked <- nil
	return nil
}

func TestParseConfig(t *testing.T) {
	t.Parallel()

	env := func(key string) string {
		switch key {
		case "CONTRACTOR_PUBLIC_LISTEN":
			return "127.0.0.1:9000"
		case "CONTRACTOR_DATABASE_URL":
			return "postgres://contractor:secret@database/contractor"
		case "CONTRACTOR_PRIVATE_LISTEN":
			return "127.0.0.1:9443"
		case "CONTRACTOR_PRIVATE_URL":
			return "https://control.internal:9443"
		case "CONTRACTOR_OPERATOR_CONFIG_ROOT":
			return "/srv/contractor/configs"
		case "CONTRACTOR_CA_FILE":
			return "/srv/contractor/pki/ca.pem"
		case "CONTRACTOR_CONTROL_PLANE_CERT_FILE":
			return "/srv/contractor/pki/control.pem"
		case "CONTRACTOR_CONTROL_PLANE_KEY_FILE":
			return "/srv/contractor/pki/control-key.pem"
		case "CONTRACTOR_LLM_GATEWAY_TOKEN":
			return "gateway-token"
		case "CONTRACTOR_PUBLIC_BEARER_TOKEN":
			return "private-token"
		case "CONTRACTOR_LOCAL_AUTH_FILE":
			return "/run/secrets/local-auth.yaml"
		case "CONTRACTOR_BROWSER_ORIGINS":
			return "https://ui.example.test,https://ui.example.test:8443"
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
	if cfg.RuntimeRequestTimeout != 30*time.Second {
		t.Fatalf("runtime request timeout = %s", cfg.RuntimeRequestTimeout)
	}
	if cfg.WorkerRequestTimeout != 180*time.Second {
		t.Fatalf("worker request timeout = %s", cfg.WorkerRequestTimeout)
	}
	if cfg.DatabaseURL != "postgres://contractor:secret@database/contractor" {
		t.Fatalf("database URL was not read from the shared environment setting")
	}
	if cfg.PublicBearerToken.Reveal() != "private-token" || cfg.LocalAuthFile != "/run/secrets/local-auth.yaml" ||
		len(cfg.BrowserOrigins) != 2 || cfg.BrowserOrigins[0] != "https://ui.example.test" {
		t.Fatalf("public API settings were not parsed: %+v", cfg)
	}
	if cfg.OperatorConfigRoot != "/srv/contractor/configs" ||
		cfg.ManagedConfigRoot != "/srv/contractor/managed-configs" {
		t.Fatalf("configuration roots were not parsed: %+v", cfg)
	}
	if cfg.PrivateListenAddress != "127.0.0.1:9443" || cfg.PrivateURL != "https://control.internal:9443" ||
		cfg.CAFile != "/srv/contractor/pki/ca.pem" || cfg.CertificateFile != "/srv/contractor/pki/control.pem" ||
		cfg.PrivateKeyFile != "/srv/contractor/pki/control-key.pem" ||
		cfg.DevelopmentWorkerToken.Reveal() != "gateway-token" {
		t.Fatalf("private runtime settings were not parsed: %+v", cfg)
	}
	if cfg.DevelopmentPlannerToken.Reveal() != cfg.DevelopmentWorkerToken.Reveal() ||
		cfg.PlannerTimeout != defaultPlannerTimeout {
		t.Fatalf("development credential fallback or Planner timeout was not parsed: %+v", cfg)
	}
}

func TestParseConfigRequiresWorkerRequestTimeoutOfAtLeastTwoMinutes(t *testing.T) {
	t.Parallel()

	if _, err := ParseConfig([]string{"--worker-request-timeout=119s"}, func(string) string { return "" }); err == nil {
		t.Fatal("sub-120-second Worker request timeout was accepted")
	}
	cfg, err := ParseConfig(
		[]string{"--worker-request-timeout=120s"},
		func(string) string { return "" },
	)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.WorkerRequestTimeout != 120*time.Second {
		t.Fatalf("worker request timeout = %s", cfg.WorkerRequestTimeout)
	}
}

func TestParseConfigRestrictsInsecureBrowserCookieToLoopback(t *testing.T) {
	t.Parallel()
	if _, err := ParseConfig([]string{
		"--listen=0.0.0.0:8080", "--insecure-loopback-cookie", "--browser-origin=http://127.0.0.1:5173",
	}, func(string) string { return "" }); err == nil {
		t.Fatal("insecure cookie on non-loopback listener was accepted")
	}
	cfg, err := ParseConfig([]string{
		"--listen=127.0.0.1:8080", "--insecure-loopback-cookie",
		"--browser-origin=http://127.0.0.1:5173", "--local-auth-file=/run/secrets/local-auth.yaml",
	}, func(string) string { return "" })
	if err != nil || !cfg.InsecureLoopbackCookie || len(cfg.BrowserOrigins) != 1 {
		t.Fatalf("loopback browser settings = (%+v, %v)", cfg, err)
	}
	if _, err := ParseConfig([]string{
		"--browser-origin=https://ui.example.test/path",
	}, func(string) string { return "" }); err == nil {
		t.Fatal("origin containing a path was accepted")
	}
}

func TestAuthHashPasswordCommandReadsTwiceAndEmitsStrictBootstrap(t *testing.T) {
	t.Parallel()
	responses := [][]byte{[]byte("command password value"), []byte("command password value")}
	prompts := make([]string, 0, 2)
	read := func(prompt string) ([]byte, error) {
		prompts = append(prompts, prompt)
		value := append([]byte(nil), responses[0]...)
		responses = responses[1:]
		return value, nil
	}
	var output bytes.Buffer
	if err := runAuthHashPassword(
		[]string{"--user-id=operator", "--username=Admin"}, read, &output,
	); err != nil {
		t.Fatal(err)
	}
	if len(prompts) != 2 || strings.Contains(output.String(), "command password value") ||
		!strings.Contains(output.String(), "$argon2id$v=19$m=65536,t=3,p=1$") {
		t.Fatalf("auth hash-password output or prompts are invalid")
	}
	path := filepath.Join(t.TempDir(), "local-auth.yaml")
	if err := os.WriteFile(path, output.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	bootstrap, err := auth.LoadBootstrap(path)
	if err != nil || bootstrap.Principal.UserID != "operator" || bootstrap.Principal.Username != "Admin" {
		t.Fatalf("generated bootstrap = (%+v, %v)", bootstrap.Principal, err)
	}
	if err := runAuthHashPassword(
		[]string{"--password=forbidden"}, read, &bytes.Buffer{},
	); err == nil {
		t.Fatal("password command-line flag was accepted")
	}
}

func TestParseConfigUsesIndependentDevelopmentPlannerCredentialAndMasterKeyPath(t *testing.T) {
	t.Parallel()
	env := func(key string) string {
		switch key {
		case "CONTRACTOR_LLM_GATEWAY_TOKEN":
			return "worker-token"
		case "CONTRACTOR_PLANNER_LLM_GATEWAY_TOKEN":
			return "planner-token"
		}
		return ""
	}
	cfg, err := ParseConfig([]string{
		"serve", "--planner-timeout=2m", "--credential-master-key-file=/run/secrets/credential-key",
		"--llm-gateway-admin-bindings-file=/run/secrets/gateway-bindings.yaml",
	}, env)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DevelopmentWorkerToken.Reveal() != "worker-token" ||
		cfg.DevelopmentPlannerToken.Reveal() != "planner-token" || cfg.PlannerTimeout != 2*time.Minute {
		t.Fatalf("independent development credentials = %+v", cfg)
	}
	if cfg.CredentialMasterKeyFile != "/run/secrets/credential-key" {
		t.Fatalf("credential master-key file = %q", cfg.CredentialMasterKeyFile)
	}
	if cfg.LLMGatewayAdminBindingsFile != "/run/secrets/gateway-bindings.yaml" {
		t.Fatalf("Gateway admin bindings file = %q", cfg.LLMGatewayAdminBindingsFile)
	}
}

func TestParseConfigReadsConfigurationRootsAndDerivesManagedFlagDefault(t *testing.T) {
	t.Parallel()
	explicit, err := ParseConfig(nil, func(key string) string {
		switch key {
		case "CONTRACTOR_OPERATOR_CONFIG_ROOT":
			return "/srv/operator/configs"
		case "CONTRACTOR_MANAGED_CONFIG_ROOT":
			return "/srv/managed/configs"
		default:
			return ""
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if explicit.OperatorConfigRoot != "/srv/operator/configs" ||
		explicit.ManagedConfigRoot != "/srv/managed/configs" {
		t.Fatalf("explicit configuration roots = %+v", explicit)
	}

	derived, err := ParseConfig(
		[]string{"--operator-config-root=/srv/custom/configs"},
		func(string) string { return "" },
	)
	if err != nil {
		t.Fatal(err)
	}
	if derived.ManagedConfigRoot != "/srv/custom/managed-configs" {
		t.Fatalf("derived managed root = %q", derived.ManagedConfigRoot)
	}
}

func TestDevelopmentCredentialsBindNamedTokensToPinnedLocalGateway(t *testing.T) {
	t.Parallel()
	snapshot, err := workflowconfig.Load("../../configs", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	provider, err := developmentCredentials(snapshot, Config{
		DevelopmentWorkerToken:  contracts.NewSecretString("worker-token"),
		DevelopmentPlannerToken: contracts.NewSecretString("planner-token"),
	})
	if err != nil {
		t.Fatal(err)
	}
	gateway, _ := snapshot.LLMGateway("local-litellm@1")
	for id, want := range map[string]string{
		developmentWorkerCredential: "worker-token", developmentPlannerCredential: "planner-token",
	} {
		metadata, err := provider.LookupLLMCredential(t.Context(), id)
		if err != nil || metadata.LLMGateway != gateway.Ref {
			t.Fatalf("development credential %q metadata = (%+v, %v)", id, metadata, err)
		}
		token, err := provider.ResolveLLMCredential(t.Context(), metadata.Ref, metadata.LLMGateway)
		if err != nil || token.Reveal() != want {
			t.Fatalf("development credential %q token = (%s, %v)", id, token, err)
		}
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

func TestParseMigrationInputs(t *testing.T) {
	t.Parallel()

	fromEnvironment, err := parseMigrationInputs(nil, func(key string) string {
		if key == "CONTRACTOR_DATABASE_URL" {
			return "postgres://environment"
		}
		return ""
	})
	if err != nil || fromEnvironment.databaseURL != "postgres://environment" ||
		fromEnvironment.budgets != persistencepostgres.DefaultMigrationBudgets() {
		t.Fatalf("environment migration inputs = (%+v, %v)", fromEnvironment, err)
	}
	fromFlag, err := parseMigrationInputs(
		[]string{"--database-url", "postgres://flag"},
		func(key string) string {
			if key == "CONTRACTOR_DATABASE_URL" {
				return "postgres://environment"
			}
			return ""
		},
	)
	if err != nil || fromFlag.databaseURL != "postgres://flag" {
		t.Fatalf("flag database URL = (%+v, %v)", fromFlag, err)
	}
	if _, err := parseMigrationInputs(nil, func(string) string { return "" }); err == nil {
		t.Fatal("missing migration database URL succeeded")
	}

	environment := map[string]string{
		"CONTRACTOR_DATABASE_URL":              "postgres://environment",
		"CONTRACTOR_MIGRATE_STATEMENT_TIMEOUT": "30m",
		"CONTRACTOR_MIGRATE_LOCK_TIMEOUT":      "1m",
	}
	getenv := func(key string) string { return environment[key] }
	timeouts, err := parseMigrationInputs(nil, getenv)
	if err != nil || timeouts.budgets != (persistencepostgres.MigrationBudgets{StatementTimeout: 30 * time.Minute, LockTimeout: time.Minute}) {
		t.Fatalf("environment migration timeouts = (%+v, %v)", timeouts, err)
	}
	timeouts, err = parseMigrationInputs([]string{"--statement-timeout", "10m", "--lock-timeout", "30s"}, getenv)
	if err != nil || timeouts.budgets != (persistencepostgres.MigrationBudgets{StatementTimeout: 10 * time.Minute, LockTimeout: 30 * time.Second}) {
		t.Fatalf("flag migration timeouts = (%+v, %v)", timeouts, err)
	}
	for _, args := range [][]string{
		{"--statement-timeout", "0"}, {"--lock-timeout", "-1s"}, {"--lock-timeout", "30m"},
		{"--statement-timeout", "25h"}, {"--statement-timeout", "soon"},
	} {
		if _, err := parseMigrationInputs(args, getenv); err == nil {
			t.Errorf("invalid migration timeouts %v accepted", args)
		}
	}
	environment["CONTRACTOR_MIGRATE_LOCK_TIMEOUT"] = "later"
	if _, err := parseMigrationInputs(nil, getenv); err == nil || !strings.Contains(err.Error(), "CONTRACTOR_MIGRATE_LOCK_TIMEOUT") {
		t.Fatalf("invalid environment migration timeout: %v", err)
	}
}
