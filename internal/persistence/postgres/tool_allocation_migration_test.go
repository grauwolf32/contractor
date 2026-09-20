package postgres

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

func TestPostgresToolAllocationConfigurationMigration(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, databaseURL)
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	const upgradeVersion = 61
	found := false
	for _, item := range available {
		found = found || item.version == upgradeVersion
	}
	if !found {
		t.Fatal("model-free allocation migration 61 is missing")
	}
	// Install the recorded pre-upgrade schema, then exercise the normal forward
	// migrator. Existing model-bearing configuration must remain accepted.
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `CREATE TABLE contractor_schema_migrations (
    version bigint PRIMARY KEY CHECK (version > 0),
    name text NOT NULL CHECK (btrim(name) <> ''),
    checksum bytea NOT NULL CHECK (octet_length(checksum) = 32),
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
)`); err != nil {
			return err
		}
		for _, item := range available {
			if item.version >= upgradeVersion {
				break
			}
			if _, err := tx.Exec(ctx, string(item.contents)); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx,
				`INSERT INTO contractor_schema_migrations (version, name, checksum) VALUES ($1, $2, $3)`,
				item.version, item.name, item.checksum[:]); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("install schema before model-free allocations: %v", err)
	}

	assertValidity := func(t *testing.T, value any, want bool) {
		t.Helper()
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		var accepted, rejected bool
		// CHECK constraints accept SQL NULL. Require explicit false for malformed
		// JSON, so a missing field cannot silently pass through three-valued logic.
		if err := pool.QueryRow(ctx, `SELECT
    contractor_valid_allocation_runtime_configuration($1::jsonb) IS TRUE,
    contractor_valid_allocation_runtime_configuration($1::jsonb) IS FALSE`, data).Scan(&accepted, &rejected); err != nil {
			t.Fatalf("validate allocation configuration: %v", err)
		}
		if accepted != want || rejected == want {
			t.Fatalf("validity = (accepted %v, rejected %v), want accepted %v", accepted, rejected, want)
		}
	}
	assertValidity(t, toolAllocationMigrationFixture(false), false)
	assertValidity(t, toolAllocationMigrationFixture(true), true)
	legacyCredential := toolAllocationMigrationFixture(true)
	legacyCredential["provenance"].(map[string]any)["llmCredential"] = map[string]any{"credentialId": "credential"}
	legacyCredential["origins"].(map[string]any)["llmCredential"] = map[string]any{"layer": "workflow"}
	assertValidity(t, legacyCredential, true)
	result, err := ApplyMigrations(ctx, pool)
	if err != nil {
		t.Fatalf("apply forward model-free allocation migration: %v", err)
	}
	if len(result.AppliedVersions) == 0 || result.AppliedVersions[0] != upgradeVersion {
		t.Fatalf("forward migration result = %+v", result)
	}
	assertValidity(t, toolAllocationMigrationFixture(false), true)
	assertValidity(t, toolAllocationMigrationFixture(true), true)
	assertValidity(t, legacyCredential, true)

	for _, name := range []string{"null", "array", "string", "number", "boolean"} {
		t.Run(name+" root", func(t *testing.T) {
			value := map[string]any{"null": nil, "array": []any{}, "string": "invalid", "number": 1, "boolean": false}[name]
			assertValidity(t, value, false)
		})
	}
	for _, field := range []string{"origins", "provenance"} {
		t.Run("missing "+field, func(t *testing.T) {
			value := toolAllocationMigrationFixture(false)
			delete(value, field)
			assertValidity(t, value, false)
		})
		for _, malformed := range []any{nil, false, 1, "invalid", []any{}} {
			t.Run(field+" wrong shape "+string(mustToolAllocationJSON(t, malformed)), func(t *testing.T) {
				value := toolAllocationMigrationFixture(false)
				value[field] = malformed
				assertValidity(t, value, false)
			})
		}
	}
	for _, field := range []string{"default", "runLabels", "agentLabels", "runtimeAdapters", "runtimeCredentialRefs"} {
		t.Run("missing provenance "+field, func(t *testing.T) {
			value := toolAllocationMigrationFixture(false)
			delete(value["provenance"].(map[string]any), field)
			assertValidity(t, value, false)
		})
		t.Run("null provenance "+field, func(t *testing.T) {
			value := toolAllocationMigrationFixture(false)
			value["provenance"].(map[string]any)[field] = nil
			assertValidity(t, value, false)
		})
		t.Run("wrong shape provenance "+field, func(t *testing.T) {
			value := toolAllocationMigrationFixture(false)
			var malformed any = map[string]any{}
			if field == "default" {
				malformed = []any{}
			}
			value["provenance"].(map[string]any)[field] = malformed
			assertValidity(t, value, false)
		})
	}
	for _, field := range []string{"llmGatewayConfig", "llmCredential"} {
		for _, supplied := range []any{nil, map[string]any{"credentialId": "credential"}} {
			t.Run("model-free provenance forbids "+field+" "+string(mustToolAllocationJSON(t, supplied)), func(t *testing.T) {
				value := toolAllocationMigrationFixture(false)
				value["provenance"].(map[string]any)[field] = supplied
				assertValidity(t, value, false)
			})
		}
	}
	for _, field := range []string{"llmGateway", "llmCredential"} {
		for _, supplied := range []any{nil, map[string]any{"layer": "workflow"}} {
			t.Run("model-free origins forbid "+field+" "+string(mustToolAllocationJSON(t, supplied)), func(t *testing.T) {
				value := toolAllocationMigrationFixture(false)
				value["origins"].(map[string]any)[field] = supplied
				assertValidity(t, value, false)
			})
		}
	}
	for _, parent := range []string{"root", "provenance"} {
		for _, field := range []string{"unknown", "llmGatewayUrl", "llmGatewayToken", "password", "headers"} {
			t.Run(parent+" rejects "+field, func(t *testing.T) {
				value := toolAllocationMigrationFixture(false)
				target := value
				if parent == "provenance" {
					target = value["provenance"].(map[string]any)
				}
				target[field] = "forbidden"
				assertValidity(t, value, false)
			})
		}
	}
	for _, field := range []string{"policyId", "version", "digest"} {
		for _, malformed := range []any{nil, 1, false, "", "invalid value"} {
			t.Run("model policy "+field+" "+string(mustToolAllocationJSON(t, malformed)), func(t *testing.T) {
				value := toolAllocationMigrationFixture(true)
				value["modelPolicy"].(map[string]any)[field] = malformed
				assertValidity(t, value, false)
			})
		}
		t.Run("model policy missing "+field, func(t *testing.T) {
			value := toolAllocationMigrationFixture(true)
			delete(value["modelPolicy"].(map[string]any), field)
			assertValidity(t, value, false)
		})
	}
	for _, malformed := range []any{nil, false, "invalid", []any{}, map[string]any{}} {
		t.Run("model policy shape "+string(mustToolAllocationJSON(t, malformed)), func(t *testing.T) {
			value := toolAllocationMigrationFixture(true)
			value["modelPolicy"] = malformed
			assertValidity(t, value, false)
		})
	}
	t.Run("model-bearing requires gateway", func(t *testing.T) {
		value := toolAllocationMigrationFixture(true)
		delete(value["provenance"].(map[string]any), "llmGatewayConfig")
		assertValidity(t, value, false)
	})
	for _, malformed := range []any{nil, false, "invalid", []any{}} {
		t.Run("model gateway shape "+string(mustToolAllocationJSON(t, malformed)), func(t *testing.T) {
			value := toolAllocationMigrationFixture(true)
			value["provenance"].(map[string]any)["llmGatewayConfig"] = malformed
			assertValidity(t, value, false)
		})
	}
	again, err := ApplyMigrations(ctx, pool)
	if err != nil || len(again.AppliedVersions) != 0 {
		t.Fatalf("idempotent migration result = %+v, err = %v", again, err)
	}
}

func toolAllocationMigrationFixture(model bool) map[string]any {
	provenance := map[string]any{
		"default": map[string]any{
			"label": "default", "bindingRevision": 1,
			"config": map[string]any{"name": "contractor-empty", "version": "1", "digest": "sha256:" + strings.Repeat("a", 64)},
		},
		"runLabels": []any{}, "agentLabels": []any{}, "runtimeAdapters": []any{}, "runtimeCredentialRefs": []any{},
	}
	origins := map[string]any{}
	result := map[string]any{"origins": origins, "provenance": provenance}
	if model {
		result["modelPolicy"] = map[string]any{"policyId": "worker", "version": "1", "digest": "sha256:" + strings.Repeat("b", 64)}
		provenance["llmGatewayConfig"] = map[string]any{"gatewayId": "gateway", "version": "1", "digest": "sha256:" + strings.Repeat("c", 64)}
		origins["llmGateway"] = map[string]any{"layer": "workflow"}
	}
	return result
}

func mustToolAllocationJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
