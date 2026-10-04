package postgres

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

func TestLifecycleForeignKeyProbeIndexes(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, databaseURL)
	if _, err := ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}

	// A deleted Run cascades through these tables. PostgreSQL probes each
	// referencing key, so its columns must lead a usable index. The two named
	// partial indexes exclude only null run_id rows, which a probe cannot match.
	rows, err := pool.Query(ctx, `
SELECT constraint_name, referencing_table FROM (
    SELECT con.conname AS constraint_name, con.conrelid::regclass::text AS referencing_table
    FROM pg_constraint AS con
    WHERE con.contype = 'f'
      AND con.confrelid IN (
          'workflow_runs'::regclass, 'workflow_run_events'::regclass,
          'planner_sessions'::regclass
      )
      -- These probes use an existing index on their first referencing key:
      -- stage_execution_id is unique, while output rows are scoped by run_id.
      AND con.conname NOT IN (
          'stage_execution_planner_session',
          'workflow_run_output_publications_run_id_project_id_fkey'
      )
      AND NOT EXISTS (
          SELECT 1 FROM pg_index AS idx
          JOIN pg_class AS index_class ON index_class.oid = idx.indexrelid
          WHERE idx.indrelid = con.conrelid
            AND idx.indisvalid AND idx.indisready AND idx.indislive
            AND idx.indnkeyatts >= array_length(con.conkey, 1)
            AND NOT EXISTS (
                SELECT 1 FROM generate_subscripts(con.conkey, 1) AS key_position(position)
                WHERE idx.indkey[key_position.position - 1] <> con.conkey[key_position.position]
            )
            AND (idx.indpred IS NULL OR index_class.relname IN (
                'artifact_scopes_run_idx'
            ))
      )
) AS missing
ORDER BY referencing_table, constraint_name`)
	if err != nil {
		t.Fatal(err)
	}
	var missing []string
	for rows.Next() {
		var constraint, table string
		if err := rows.Scan(&constraint, &table); err != nil {
			rows.Close()
			t.Fatal(err)
		}
		missing = append(missing, table+"."+constraint)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		t.Fatal(err)
	}
	rows.Close()
	if len(missing) > 0 {
		t.Fatalf("lifecycle foreign keys lack leading indexes: %s", strings.Join(missing, ", "))
	}
}
