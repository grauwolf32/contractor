package app

import (
	"bytes"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestOperationalDefaultsAreIndependentOfHTTPTimeout(t *testing.T) {
	cfg, err := ParseConfig([]string{"--runtime-request-timeout", "7s"}, func(string) string { return "" })
	if err != nil {
		t.Fatal(err)
	}
	if cfg.RuntimeRequestTimeout != 7*time.Second || cfg.Operations != defaultOperationalSettings() {
		t.Fatalf("HTTP override changed operation settings: %+v", cfg.Operations)
	}
}

func TestOperationalSettingsFromFileReachDatabaseAndEffectiveLog(t *testing.T) {
	path := filepath.Join(t.TempDir(), "server.yaml")
	writeServerConfigTestFile(t, path, `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  scheduler: {pollInterval: 3s, claimDuration: 5m, leaseScanInterval: 2s, operationTimeout: 37s, finalizationTimeout: 13s, abortTimeout: 11s}
  runtimeLifecycle: {prepareTimeout: 21s, cleanupTimeout: 17s}
  projectLifecycle: {operationTimeout: 19s, claimDuration: 71s, purgeTimeout: 5m}
  auditController: {pollInterval: 3s, claimLease: 45s, operationTimeout: 12s, claimBatch: 9}
  database: {connectTimeout: 4s, acquireTimeout: 3s, queryTimeout: 23s, statementTimeout: 16s, lockTimeout: 4s, idleTransactionTimeout: 31s}
  llmRecovery: {initialDelay: 2s, maxDelay: 40s, automaticWindow: 6m}
  a2a: {pollInterval: 250ms}
  credentialManagement: {connectTimeout: 2s, requestTimeout: 14s}
`)
	cfg, err := ParseConfig([]string{"--config", path}, func(string) string { return "" })
	if err != nil {
		t.Fatal(err)
	}
	want := OperationalSettings{
		LLMRecovery: gatewayrecovery.Policy{InitialDelay: 2 * time.Second, MaxDelay: 40 * time.Second, AutomaticWindow: 6 * time.Minute},
		Scheduler: SchedulerSettings{
			PollInterval: 3 * time.Second, ClaimDuration: 5 * time.Minute, LeaseScanInterval: 2 * time.Second,
			OperationTimeout: 37 * time.Second, FinalizationTimeout: 13 * time.Second, AbortTimeout: 11 * time.Second,
		},
		RuntimeLifecycle:     RuntimeLifecycleSettings{PrepareTimeout: 21 * time.Second, CleanupTimeout: 17 * time.Second},
		ProjectLifecycle:     ProjectLifecycleSettings{19 * time.Second, 71 * time.Second, 5 * time.Minute},
		AuditController:      AuditControllerSettings{3 * time.Second, 45 * time.Second, 12 * time.Second, 9},
		Database:             DatabaseSettings{4 * time.Second, 3 * time.Second, 23 * time.Second, 16 * time.Second, 4 * time.Second, 31 * time.Second},
		A2A:                  A2ASettings{250 * time.Millisecond},
		CredentialManagement: CredentialManagementSettings{2 * time.Second, 14 * time.Second},
	}
	if cfg.Operations != want {
		t.Fatalf("resolved settings = %+v, want %+v", cfg.Operations, want)
	}
	pool, err := persistencepostgres.PoolConfig("postgres://user:do-not-log@localhost/db?pool_max_conns=9", cfg.Operations.Database.poolOptions(nil))
	if err != nil {
		t.Fatal(err)
	}
	if pool.ConnConfig.ConnectTimeout != 4*time.Second || pool.MaxConns != 9 ||
		pool.ConnConfig.RuntimeParams["statement_timeout"] != "16000" ||
		pool.ConnConfig.RuntimeParams["lock_timeout"] != "4000" ||
		pool.ConnConfig.RuntimeParams["idle_in_transaction_session_timeout"] != "31000" {
		t.Fatal("process database settings did not reach pgx")
	}
	var output bytes.Buffer
	cfg.Operations.logEffective(slog.New(slog.NewJSONHandler(&output, nil)), cfg.RuntimeRequestTimeout)
	for _, effective := range []string{
		`"scheduler.operationTimeout":"37s"`, `"scheduler.claimDuration":"5m0s"`,
		`"runtimeLifecycle.prepareTimeout":"21s"`, `"runtimeLifecycle.cleanupTimeout":"17s"`,
	} {
		if !strings.Contains(output.String(), effective) {
			t.Fatalf("effective settings log lacks %s: %s", effective, output.String())
		}
	}
	if strings.Contains(output.String(), "do-not-log") {
		t.Fatalf("unexpected effective settings log: %s", output.String())
	}
}

func TestOperationalPrecedenceAndStrictValidation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "server.yaml")
	writeServerConfigTestFile(t, path, `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  scheduler: {operationTimeout: 41s}
  auditController: {claimBatch: 11}
`)
	env := func(key string) string {
		return map[string]string{
			"CONTRACTOR_SCHEDULER_OPERATION_TIMEOUT":  "43s",
			"CONTRACTOR_AUDIT_CONTROLLER_CLAIM_BATCH": "12",
		}[key]
	}
	cfg, err := ParseConfig([]string{"--config", path}, env)
	if err != nil || cfg.Operations.Scheduler.OperationTimeout != 43*time.Second || cfg.Operations.AuditController.ClaimBatch != 12 {
		t.Fatalf("environment did not override file: %+v, %v", cfg.Operations, err)
	}
	cfg, err = ParseConfig([]string{"--config", path, "--scheduler-operation-timeout", "47s", "--audit-controller-claim-batch", "13"}, env)
	if err != nil || cfg.Operations.Scheduler.OperationTimeout != 47*time.Second || cfg.Operations.AuditController.ClaimBatch != 13 {
		t.Fatalf("flag did not override environment: %+v, %v", cfg.Operations, err)
	}
	for _, spec := range []string{
		"scheduler: {operationTimeout: 0s}", "runtimeLifecycle: {cleanupTimeout: -1s}",
		"runtimeLifecycle: {prepareTimeout: 0s}", "scheduler: {pollInterval: 0s}",
		"scheduler: {leaseScanInterval: -1s}",
		// The Run claim lease is at least twice the Scheduler operation budget.
		"scheduler: {claimDuration: 59s}", "scheduler: {operationTimeout: 61s}",
		"scheduler: {claimDuration: 999ns, operationTimeout: 1ns}",
		"scheduler: {finalizationTimeout: NaN}", "scheduler: {unknown: 1s}",
		"scheduler: {abortTimeout: 1s, abortTimeout: 2s}",
		"auditController: {claimBatch: 0}", "auditController: {claimBatch: 101}",
		"auditController: {operationTimeout: 16s}", "auditController: {claimLease: 6m}",
		"auditController: {operationTimeout: 2562047h47m16s}",
		"database: {lockTimeout: 16s}", "database: {statementTimeout: 20s}",
		"database: {queryTimeout: 0s}", "database: {acquireTimeout: 1ns}",
		"projectLifecycle: {operationTimeout: 1m}",
		"projectLifecycle: {operationTimeout: 1ns, claimDuration: 2ns}",
		"projectLifecycle: {purgeTimeout: 0s}",
		"credentialManagement: {connectTimeout: 61s}", "credentialManagement: {requestTimeout: 121s}",
		// Model requests keep their worker/planner timeouts; recovery has none.
		"llmRecovery: {requestTimeout: 60s}",
	} {
		t.Run(spec, func(t *testing.T) {
			writeServerConfigTestFile(t, path, "apiVersion: contractor/v1alpha1\nkind: ServerConfig\nspec:\n  "+spec+"\n")
			if _, err := ParseConfig([]string{"--config", path}, func(string) string { return "" }); err == nil {
				t.Fatal("invalid settings accepted")
			}
		})
	}
}

func TestSchedulerAndRuntimeBatchBudgetsFollowPrecedence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "server.yaml")
	writeServerConfigTestFile(t, path, `apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  scheduler: {claimDuration: 1m, operationTimeout: 30s, pollInterval: 4s}
  runtimeLifecycle: {prepareTimeout: 40s}
`)
	env := map[string]string{
		"CONTRACTOR_RUNTIME_LIFECYCLE_PREPARE_TIMEOUT": "44s",
		"CONTRACTOR_SCHEDULER_LEASE_SCAN_INTERVAL":     "2s",
	}
	cfg, err := ParseConfig(
		[]string{"--config", path, "--scheduler-lease-scan-interval", "3s"},
		func(key string) string { return env[key] },
	)
	if err != nil {
		t.Fatal(err)
	}
	scheduler, runtime := cfg.Operations.Scheduler, cfg.Operations.RuntimeLifecycle
	// A lease of exactly twice the operation budget is the accepted boundary.
	if scheduler.ClaimDuration != time.Minute || scheduler.PollInterval != 4*time.Second ||
		scheduler.LeaseScanInterval != 3*time.Second || runtime.PrepareTimeout != 44*time.Second ||
		runtime.CleanupTimeout != defaultOperationalSettings().RuntimeLifecycle.CleanupTimeout {
		t.Fatalf("resolved budgets = %+v, %+v", scheduler, runtime)
	}
}

// The timeout reference is the single table of process budgets (OPS-R18):
// every ServerConfig timeout has one row stating the default the Server
// resolves, and every row names a setting the Server reads.
func TestTimeoutReferenceTableMatchesServerConfig(t *testing.T) {
	document, err := os.ReadFile(filepath.Join("..", "..", "docs", "operations", "timeout-configuration.md"))
	if err != nil {
		t.Fatal(err)
	}
	defaults := defaultServerConfigValues()
	want := map[string]string{
		"shutdownTimeout":            defaults.shutdownTimeout.String(),
		"runtimeRequestTimeout":      defaults.runtimeRequestTimeout.String(),
		"workerRequestTimeout":       defaults.workerRequestTimeout.String(),
		"plannerTimeout":             defaults.plannerTimeout.String(),
		"auditController.claimBatch": strconv.Itoa(defaults.operations.AuditController.ClaimBatch),
	}
	for _, setting := range defaults.operations.durations(operationalSpec{}) {
		want[setting.path] = setting.target.String()
	}
	got := map[string]string{}
	rows := regexp.MustCompile("(?m)^\\| `([A-Za-z0-9.]+)` \\| `([^`]+)` \\|").FindAllStringSubmatch(string(document), -1)
	for _, row := range rows {
		path, value := row[1], row[2]
		if _, duplicate := got[path]; duplicate {
			t.Errorf("timeout reference repeats %s", path)
		}
		if duration, parseErr := time.ParseDuration(value); parseErr == nil {
			value = duration.String()
		}
		got[path] = value
	}
	for path, value := range want {
		if got[path] != value {
			t.Errorf("timeout reference %s default = %q, Server resolves %q", path, got[path], value)
		}
	}
	for path := range got {
		if _, known := want[path]; !known {
			t.Errorf("timeout reference lists %s, which ServerConfig does not read", path)
		}
	}
}

func TestProcessDatabaseSettingsRejectConflictingConnectionTimeouts(t *testing.T) {
	settings := defaultOperationalSettings()
	for _, parameter := range []string{
		"connect_timeout=42", "statement_timeout=42000", "lock_timeout=6000",
		"idle_in_transaction_session_timeout=90000", "options=-c%20statement_timeout%3D0",
	} {
		_, err := persistencepostgres.PoolConfig("postgres://user:secret-canary@localhost/db?"+parameter, settings.Database.poolOptions(nil))
		if err == nil || !strings.Contains(err.Error(), "ServerConfig") || strings.Contains(err.Error(), "secret-canary") {
			t.Fatalf("%s conflict: %v", parameter, err)
		}
	}
}
