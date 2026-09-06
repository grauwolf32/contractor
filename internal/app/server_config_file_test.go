package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestServerConfigFileLoadsNonSecretProcessSettingsAndResolvesPaths(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	path := filepath.Join(directory, "server.yaml")
	writeServerConfigTestFile(t, path, `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  listen: 127.0.0.1:9080
  privateListen: 127.0.0.1:9443
  privateUrl: https://127.0.0.1:9443
  shutdownTimeout: 7s
  runtimeRequestTimeout: 41s
  workerRequestTimeout: 121s
  plannerTimeout: 12m
  artifactBlobBackend: filesystem
  artifactBlobPath: blobs
  operatorConfigRoot: operator
  managedConfigRoot: managed
  credentialMasterKeyFile: secrets/credential.key
  llmGatewayAdminBindingsFile: gateway-bindings.yaml
  localAuthFile: secrets/local-auth.yaml
  browserOrigins: [http://127.0.0.1:4173]
  insecureLoopbackCookie: true
  caFile: pki/ca.crt
  certificateFile: pki/control-plane.crt
  privateKeyFile: pki/control-plane.key
  performanceMetrics: false
  pprof: true
  pprofListen: 127.0.0.1:6061
`)

	cfg, err := ParseConfig([]string{"serve", "--config", path}, func(string) string { return "" })
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ListenAddress != "127.0.0.1:9080" || cfg.PrivateListenAddress != "127.0.0.1:9443" ||
		cfg.PrivateURL != "https://127.0.0.1:9443" {
		t.Fatalf("listeners = %+v", cfg)
	}
	if cfg.ShutdownTimeout != 7*time.Second || cfg.RuntimeRequestTimeout != 41*time.Second ||
		cfg.WorkerRequestTimeout != 121*time.Second || cfg.PlannerTimeout != 12*time.Minute {
		t.Fatalf("timeouts = %+v", cfg)
	}
	if cfg.ArtifactBlobBackend != artifacts.BlobFilesystem ||
		cfg.ArtifactBlobPath != filepath.Join(directory, "blobs") {
		t.Fatalf("blob settings = %+v", cfg)
	}
	for name, got := range map[string]string{
		"operator root":   cfg.OperatorConfigRoot,
		"managed root":    cfg.ManagedConfigRoot,
		"credential key":  cfg.CredentialMasterKeyFile,
		"Gateway binding": cfg.LLMGatewayAdminBindingsFile,
		"local auth":      cfg.LocalAuthFile,
		"CA":              cfg.CAFile,
		"certificate":     cfg.CertificateFile,
		"private key":     cfg.PrivateKeyFile,
	} {
		if !filepath.IsAbs(got) || !strings.HasPrefix(got, directory+string(filepath.Separator)) {
			t.Fatalf("%s path %q was not resolved relative to the ServerConfig", name, got)
		}
	}
	if cfg.PublicUserID != "" || len(cfg.BrowserOrigins) != 1 ||
		cfg.BrowserOrigins[0] != "http://127.0.0.1:4173" || !cfg.InsecureLoopbackCookie ||
		cfg.PerformanceMetrics || !cfg.Pprof || cfg.PprofListen != "127.0.0.1:6061" {
		t.Fatalf("remaining ServerConfig settings = %+v", cfg)
	}
	if cfg.DatabaseURL != "" || cfg.PublicBearerToken.Reveal() != "" ||
		cfg.DevelopmentWorkerToken.Reveal() != "" || cfg.DevelopmentPlannerToken.Reveal() != "" {
		t.Fatal("ServerConfig populated a secret value")
	}
}

func TestServerConfigPrecedenceIsDefaultsThenFileThenEnvironmentThenFlags(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	environmentPath := filepath.Join(directory, "environment.yaml")
	flagPath := filepath.Join(directory, "flag.yaml")
	writeServerConfigTestFile(t, environmentPath, `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  listen: 127.0.0.1:9001
`)
	writeServerConfigTestFile(t, flagPath, `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  listen: 127.0.0.1:9002
  operatorConfigRoot: file-configs
  performanceMetrics: false
  pprof: true
`)
	environment := map[string]string{
		"CONTRACTOR_SERVER_CONFIG":        environmentPath,
		"CONTRACTOR_PUBLIC_LISTEN":        "127.0.0.1:9003",
		"CONTRACTOR_OPERATOR_CONFIG_ROOT": "/environment/configs",
		"CONTRACTOR_PERFORMANCE_METRICS":  "true",
		"CONTRACTOR_PPROF":                "false",
		"CONTRACTOR_BROWSER_ORIGINS":      "https://environment.example",
	}
	cfg, err := ParseConfig([]string{
		"--server-config=" + flagPath,
		"--listen=127.0.0.1:9004",
		"--operator-config-root=/flag/configs",
		"--performance-metrics=false",
		"--pprof=true",
		"--browser-origin=https://flag-one.example",
		"--browser-origin=https://flag-two.example",
	}, func(name string) string { return environment[name] })
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ListenAddress != "127.0.0.1:9004" || cfg.OperatorConfigRoot != "/flag/configs" ||
		cfg.PerformanceMetrics || !cfg.Pprof {
		t.Fatalf("effective precedence = %+v", cfg)
	}
	if cfg.ManagedConfigRoot != "/flag/managed-configs" {
		t.Fatalf("managed config root = %q", cfg.ManagedConfigRoot)
	}
	if len(cfg.BrowserOrigins) != 2 || cfg.BrowserOrigins[0] != "https://flag-one.example" ||
		cfg.BrowserOrigins[1] != "https://flag-two.example" {
		t.Fatalf("browser origins = %v", cfg.BrowserOrigins)
	}
}

func TestServerConfigRejectsAmbiguousOrSecretBearingDocuments(t *testing.T) {
	t.Parallel()
	tests := map[string]string{
		"unknown top-level field": `apiVersion: contractor/v1alpha1
kind: ServerConfig
unexpected: true
spec: {}
`,
		"unknown spec field": `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec: {unexpected: true}
`,
		"duplicate field": `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  listen: 127.0.0.1:8080
  listen: 127.0.0.1:8081
`,
		"multiple documents": `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec: {}
---
apiVersion: contractor/v1alpha1
kind: ServerConfig
spec: {}
`,
		"wrong kind": `apiVersion: contractor/v1alpha1
kind: Workflow
spec: {}
`,
		"secret database URL": `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  databaseURL: postgres://user:secret@database/contractor
`,
		"invalid duration": `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  plannerTimeout: secret-invalid
`,
	}
	for name, document := range tests {
		name, document := name, document
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			path := filepath.Join(t.TempDir(), "server.yaml")
			writeServerConfigTestFile(t, path, document)
			_, err := ParseConfig([]string{"--config=" + path}, func(string) string { return "" })
			if err == nil {
				t.Fatal("invalid ServerConfig was accepted")
			}
			if strings.Contains(err.Error(), "postgres://user:secret") || strings.Contains(err.Error(), "secret-invalid") {
				t.Fatalf("error exposed a supplied value: %v", err)
			}
		})
	}
	if _, err := ParseConfig([]string{"--config"}, func(string) string { return "" }); err == nil {
		t.Fatal("missing --config value was accepted")
	}
	if _, err := ParseConfig([]string{"--config="}, func(string) string { return "" }); err == nil {
		t.Fatal("empty --config value was accepted")
	}
}

func TestServerConfigRejectsInvalidFileContainers(t *testing.T) {
	t.Parallel()

	tests := map[string]func(*testing.T) string{
		"empty": func(t *testing.T) string {
			path := filepath.Join(t.TempDir(), "empty.yaml")
			writeServerConfigTestFile(t, path, "")
			return path
		},
		"directory": func(t *testing.T) string {
			return t.TempDir()
		},
		"oversized": func(t *testing.T) string {
			path := filepath.Join(t.TempDir(), "oversized.yaml")
			writeServerConfigTestFile(t, path, strings.Repeat("#", maximumServerConfig+1))
			return path
		},
	}
	for name, fixture := range tests {
		name, fixture := name, fixture
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if _, err := ParseConfig([]string{"--config", fixture(t)}, func(string) string { return "" }); err == nil {
				t.Fatal("invalid ServerConfig container was accepted")
			}
		})
	}
}

func TestRepositoryLocalServerConfigTracksExecutableDefaults(t *testing.T) {
	t.Parallel()
	path := filepath.Join("..", "..", "configs", "server.local.yaml")
	cfg, err := ParseConfig([]string{"--config", path}, func(string) string { return "" })
	if err != nil {
		t.Fatal(err)
	}
	root, err := filepath.Abs(filepath.Join("..", "..", "configs"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.OperatorConfigRoot != root || cfg.ListenAddress != defaultListenAddress ||
		cfg.PrivateListenAddress != defaultPrivateListenAddress ||
		cfg.WorkerRequestTimeout != defaultWorkerRequestTimeout ||
		cfg.PlannerTimeout != defaultPlannerTimeout || !cfg.InsecureLoopbackCookie ||
		len(cfg.BrowserOrigins) != 1 || cfg.BrowserOrigins[0] != "http://127.0.0.1:4173" {
		t.Fatalf("repository ServerConfig = %+v", cfg)
	}
}

func writeServerConfigTestFile(t *testing.T, path, document string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
}
