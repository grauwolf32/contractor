package postgres

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// TestPostgresSchemaKeepsSecretsEncryptedAndLivenessInMemory checks design
// rules on the migrated schema rather than on migration text, so any later
// migration that breaks one fails here: secrets are stored only as ciphertext,
// and Runtime Agent liveness, queue positions and per-owner scheduler state do
// not become durable columns.
func TestPostgresSchemaKeepsSecretsEncryptedAndLivenessInMemory(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	columns := schemaColumns(t, ctx, pool)

	plaintext := regexp.MustCompile(`plaintext|password|passphrase|secret|api_key|private_key|^token$|_token$`)
	for table, types := range columns {
		for column := range types {
			if plaintext.MatchString(column) {
				t.Errorf("%s.%s looks like a plaintext secret column", table, column)
			}
		}
	}
	for _, table := range []string{"git_ssh_keys", "llm_credentials", "runtime_credentials"} {
		for column, want := range map[string]string{
			"encryption_schema_version": "text", "key_id": "text", "nonce": "bytea", "ciphertext": "bytea",
		} {
			if got := columns[table][column]; got != want {
				t.Errorf("%s.%s type = %q, want %q", table, column, got, want)
			}
		}
	}
	for table, fragments := range map[string][]string{
		"runtime_agent_principals": {"heartbeat", "lease_expires", "observed_state", "allocation_id", "control_url"},
		"owner_queue_controls":     {"run_id", "stage_execution_id", "priority", "position"},
		"scheduler_settings":       {"owner_id", "project_id"},
	} {
		if len(columns[table]) == 0 {
			t.Errorf("table %s is missing", table)
			continue
		}
		for column := range columns[table] {
			for _, fragment := range fragments {
				if strings.Contains(column, fragment) {
					t.Errorf("%s.%s stores state that belongs in memory", table, column)
				}
			}
		}
	}
}

// schemaColumns maps each table of the current schema to its column types.
func schemaColumns(t *testing.T, ctx context.Context, pool *pgxpool.Pool) map[string]map[string]string {
	t.Helper()
	rows, err := pool.Query(ctx, `
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = current_schema()`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	columns := map[string]map[string]string{}
	for rows.Next() {
		var table, column, dataType string
		if err := rows.Scan(&table, &column, &dataType); err != nil {
			t.Fatal(err)
		}
		if columns[table] == nil {
			columns[table] = map[string]string{}
		}
		columns[table][column] = dataType
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	return columns
}

// TestPostgresSchemaSeedsOneProtectedSchedulerSetting checks the row every
// database starts with: one Scheduler settings singleton that allows one Run
// at a time and that direct SQL can neither delete nor duplicate, so the
// Server never has to guess a process-local fallback.
func TestPostgresSchemaSeedsOneProtectedSchedulerSetting(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	var rows, maxConcurrentRuns int
	var revision string
	if err := pool.QueryRow(ctx, `
SELECT count(*), min(max_concurrent_runs), min(revision)::text FROM scheduler_settings WHERE singleton`,
	).Scan(&rows, &maxConcurrentRuns, &revision); err != nil {
		t.Fatal(err)
	}
	if rows != 1 || maxConcurrentRuns != 1 || revision != "1" {
		t.Fatalf("seeded Scheduler settings = %d rows, max %d, revision %s; want 1 row, max 1, revision 1",
			rows, maxConcurrentRuns, revision)
	}
	_, err := pool.Exec(ctx, `DELETE FROM scheduler_settings`)
	assertSQLState(t, err, SQLStateCheckViolation)
	_, err = pool.Exec(ctx, `INSERT INTO scheduler_settings (singleton, max_concurrent_runs, revision) VALUES (true, 2, 1)`)
	assertSQLState(t, err, SQLStateUniqueViolation)
	_, err = pool.Exec(ctx, `INSERT INTO scheduler_settings (singleton, max_concurrent_runs, revision) VALUES (false, 2, 1)`)
	assertSQLState(t, err, SQLStateCheckViolation)
}

// TestPostgresRunMetadataLabelsAreBoundedIndexedAndImmutable checks that the
// schema itself bounds Run metadata labels, indexes them for exact lookup and
// rejects any change after the Run is created.
func TestPostgresRunMetadataLabelsAreBoundedIndexedAndImmutable(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	insert := func(runID, labels string) error {
		_, err := pool.Exec(ctx, `
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    runtime_labels, runtime_config_snapshot, metadata_labels,
    state, state_reason_code
) VALUES ($1, 'user', 'workflow', '1', 'v1', '{}', '{}', '{}', $2::jsonb, $3::jsonb, 'initializing', 'test')`,
			runID, builtInRunRuntimeSnapshot, labels)
		return err
	}
	tooMany := make([]string, 33)
	for index := range tooMany {
		tooMany[index] = fmt.Sprintf(`"label%d":"value"`, index)
	}
	for name, labels := range map[string]string{
		"33 labels":        "{" + strings.Join(tooMany, ",") + "}",
		"upper-case key":   `{"Team":"red"}`,
		"reserved prefix":  `{"contractor.owner":"x"}`,
		"non-string value": `{"team":1}`,
		"long value":       `{"team":"` + strings.Repeat("x", 257) + `"}`,
	} {
		assertSQLState(t, insert("run-"+strings.ReplaceAll(name, " ", "-"), labels), SQLStateCheckViolation)
	}
	if err := insert("run-labelled", `{"team":"red","tier":"gold"}`); err != nil {
		t.Fatalf("insert labelled Run: %v", err)
	}
	_, err := pool.Exec(ctx, `UPDATE workflow_runs SET metadata_labels = '{}' WHERE run_id = 'run-labelled'`)
	assertSQLState(t, err, SQLStateCheckViolation)
	var definition string
	if err := pool.QueryRow(ctx, `
SELECT indexdef FROM pg_indexes
WHERE schemaname = current_schema() AND indexname = 'workflow_runs_metadata_labels_idx'`,
	).Scan(&definition); err != nil {
		t.Fatalf("metadata label index: %v", err)
	}
	if !strings.Contains(definition, "USING gin (metadata_labels jsonb_path_ops)") {
		t.Fatalf("metadata label index = %s, want a jsonb_path_ops GIN index", definition)
	}
}
