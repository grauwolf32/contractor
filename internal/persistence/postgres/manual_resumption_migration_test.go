package postgres

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

func TestPostgresManualResumptionMigrationPreservesHistory(t *testing.T) {
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
	var upgrade *migration
	for index := range available {
		if available[index].version == 60 {
			upgrade = &available[index]
			break
		}
	}
	if upgrade == nil {
		t.Fatal("manual continuation migration 60 is missing")
	}

	// Reproduce an installed database, including its checksum ledger, before
	// manual continuations had an explicit identity on stage_executions.
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `
CREATE TABLE contractor_schema_migrations (
    version bigint PRIMARY KEY CHECK (version > 0),
    name text NOT NULL CHECK (btrim(name) <> ''),
    checksum bytea NOT NULL CHECK (octet_length(checksum) = 32),
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
)`); err != nil {
			return err
		}
		for _, item := range available {
			if item.version >= upgrade.version {
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
		t.Fatalf("install schema before manual continuation migration: %v", err)
	}

	if _, err := pool.Exec(ctx, `
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, runtime_labels,
    runtime_config_snapshot, state, state_reason_code, finished_at
) VALUES (
    'historical-run', 'owner', 'workflow', '1', '1', '{}', '{}',
    $1::jsonb, 'failed', 'attempts_exhausted', '2026-09-01T12:00:00Z'
)`, builtInRunRuntimeSnapshot); err != nil {
		t.Fatalf("insert historical Run: %v", err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt, previous_execution_id,
    stage_spec_schema_version, stage_spec_snapshot,
    stage_context_schema_version, stage_context_snapshot,
    state, state_reason_code, termination_schema_version, stage_termination,
    abort_id, abort_deadline, terminal_at
)
SELECT execution_id, 'historical-run', 'build', attempt, previous_id,
    '1', '{"objective":"build"}', '1', '{}',
    'interrupted', 'worker_interrupted', '1',
    '{"outcome":"interrupted","code":"worker_interrupted","message":"worker stopped","retryable":false,"phase":"preparing","occurredAt":"2026-09-01T12:00:00Z"}',
    'abort-' || execution_id, '2026-09-01T12:00:00Z', '2026-09-01T12:00:00Z'
FROM (VALUES
    ('historical-source', 1, NULL::text),
    ('historical-target', 2, 'historical-source')
) AS attempts(execution_id, attempt, previous_id);

INSERT INTO run_stage_resumptions (
    source_execution_id, run_id, target_execution_id, requested_by, requested_at
) VALUES (
    'historical-source', 'historical-run', 'historical-target',
    'owner', '2026-09-01T11:00:00Z'
)`); err != nil {
		t.Fatalf("insert continuation using the old schema: %v", err)
	}

	// Removing an absent JSON key also works on the old schema. Compare every
	// historical field, including terminal facts, timestamps and receipt xid.
	const historyQuery = `
SELECT jsonb_build_object(
    'run', (SELECT to_jsonb(r) FROM workflow_runs r WHERE run_id = 'historical-run'),
    'source', (SELECT to_jsonb(s) - 'resume_source_execution_id'
        FROM stage_executions s WHERE stage_execution_id = 'historical-source'),
    'target', (SELECT to_jsonb(s) - 'resume_source_execution_id'
        FROM stage_executions s WHERE stage_execution_id = 'historical-target'),
    'receipt', (SELECT to_jsonb(r) FROM run_stage_resumptions r
        WHERE source_execution_id = 'historical-source')
)::text`
	var before, after string
	if err := pool.QueryRow(ctx, historyQuery).Scan(&before); err != nil {
		t.Fatal(err)
	}
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, string(upgrade.contents)); err != nil {
			return err
		}
		_, err := tx.Exec(ctx,
			`INSERT INTO contractor_schema_migrations (version, name, checksum) VALUES ($1, $2, $3)`,
			upgrade.version, upgrade.name, upgrade.checksum[:])
		return err
	})
	if err != nil {
		t.Fatalf("upgrade historical continuation: %v", err)
	}
	if err := pool.QueryRow(ctx, historyQuery).Scan(&after); err != nil {
		t.Fatal(err)
	}
	if after != before {
		t.Fatalf("migration changed historical fields:\nbefore: %s\nafter:  %s", before, after)
	}
	var sourceResume, targetResume *string
	if err := pool.QueryRow(ctx, `
SELECT source.resume_source_execution_id, target.resume_source_execution_id
FROM stage_executions source CROSS JOIN stage_executions target
WHERE source.stage_execution_id = 'historical-source'
    AND target.stage_execution_id = 'historical-target'`).Scan(&sourceResume, &targetResume); err != nil {
		t.Fatal(err)
	}
	if sourceResume != nil || targetResume == nil || *targetResume != "historical-source" {
		t.Fatalf("backfilled identities: source = %v, target = %v", sourceResume, targetResume)
	}

	// Use the normal migrator to validate recorded checksums. Apply future
	// migrations, if any, without making this version-60 regression brittle.
	if _, err := ApplyMigrations(ctx, pool); err != nil {
		t.Fatalf("validate upgraded migration ledger: %v", err)
	}
	again, err := ApplyMigrations(ctx, pool)
	if err != nil || len(again.AppliedVersions) != 0 || again.CurrentVersion != available[len(available)-1].version {
		t.Fatalf("idempotent upgrade: result = %+v, err = %v", again, err)
	}

	// The receipt and its target now reference each other. A terminal Run
	// purge must still cascade through both rows and satisfy deferred FKs.
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SET LOCAL contractor.lifecycle_purge = 'run'`); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `DELETE FROM workflow_runs WHERE run_id = 'historical-run'`)
		return err
	})
	if err != nil {
		t.Fatalf("purge upgraded Run with continuation receipt: %v", err)
	}
	var remaining int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM workflow_runs WHERE run_id = 'historical-run')
    + (SELECT count(*) FROM stage_executions WHERE run_id = 'historical-run')
    + (SELECT count(*) FROM run_stage_resumptions WHERE run_id = 'historical-run')`).Scan(&remaining); err != nil {
		t.Fatal(err)
	}
	if remaining != 0 {
		t.Fatalf("purge left %d historical rows", remaining)
	}
}
