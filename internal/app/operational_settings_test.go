package app

import (
	"bytes"
	"log/slog"
	"path/filepath"
	"strings"
	"testing"
	"time"

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
  scheduler: {operationTimeout: 37s, finalizationTimeout: 13s, abortTimeout: 11s}
  runtimeLifecycle: {cleanupTimeout: 17s}
  projectLifecycle: {operationTimeout: 19s, claimDuration: 71s}
  auditController: {pollInterval: 3s, claimLease: 45s, operationTimeout: 12s, claimBatch: 9}
  database: {connectTimeout: 4s, acquireTimeout: 3s, queryTimeout: 23s, statementTimeout: 16s, lockTimeout: 4s, idleTransactionTimeout: 31s}
  a2a: {pollInterval: 250ms}
  credentialManagement: {connectTimeout: 2s, requestTimeout: 14s}
`)
	cfg, err := ParseConfig([]string{"--config", path}, func(string) string { return "" })
	if err != nil {
		t.Fatal(err)
	}
	want := OperationalSettings{
		Scheduler:            SchedulerSettings{37 * time.Second, 13 * time.Second, 11 * time.Second},
		RuntimeLifecycle:     RuntimeLifecycleSettings{17 * time.Second},
		ProjectLifecycle:     ProjectLifecycleSettings{19 * time.Second, 71 * time.Second},
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
	if !strings.Contains(output.String(), `"scheduler.operationTimeout":"37s"`) || strings.Contains(output.String(), "do-not-log") {
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
		"scheduler: {finalizationTimeout: NaN}", "scheduler: {unknown: 1s}",
		"scheduler: {abortTimeout: 1s, abortTimeout: 2s}",
		"auditController: {claimBatch: 0}", "auditController: {claimBatch: 101}",
		"auditController: {operationTimeout: 16s}", "auditController: {claimLease: 6m}",
		"auditController: {operationTimeout: 2562047h47m16s}",
		"database: {lockTimeout: 16s}", "database: {statementTimeout: 20s}",
		"database: {queryTimeout: 0s}", "database: {acquireTimeout: 1ns}",
		"projectLifecycle: {operationTimeout: 1m}",
		"credentialManagement: {connectTimeout: 61s}", "credentialManagement: {requestTimeout: 121s}",
	} {
		t.Run(spec, func(t *testing.T) {
			writeServerConfigTestFile(t, path, "apiVersion: contractor/v1alpha1\nkind: ServerConfig\nspec:\n  "+spec+"\n")
			if _, err := ParseConfig([]string{"--config", path}, func(string) string { return "" }); err == nil {
				t.Fatal("invalid settings accepted")
			}
		})
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
