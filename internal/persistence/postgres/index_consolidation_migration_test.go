package postgres

import (
	"context"
	"errors"
	"maps"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// consolidationIndexesSQL lists every index of the current schema with its
// definition, without the schema name.
const consolidationIndexesSQL = `
SELECT index.relname, replace(pg_get_indexdef(i.indexrelid), ' ON ' || quote_ident(current_schema()) || '.', ' ON ')
  FROM pg_index AS i
  JOIN pg_class AS index ON index.oid = i.indexrelid
 WHERE index.relnamespace = current_schema()::regnamespace`

// consolidationConstraintsSQL lists every primary key, UNIQUE constraint and
// foreign key of the current schema with its definition, the index it owns or
// references, its actions and its deferral and validation state.
const consolidationConstraintsSQL = `
SELECT conrelid::regclass::text || '.' || conname,
       concat_ws(' ', contype, pg_get_constraintdef(oid), conindid::regclass::text,
                 confdeltype, confupdtype, condeferrable, condeferred, convalidated)
  FROM pg_constraint
 WHERE connamespace = current_schema()::regnamespace AND contype IN ('p', 'u', 'f')`

// applyMigrationsThrough applies the embedded migrations that pool's ledger
// lacks, up to and including the one named suffix, so that a migration's
// catalog test sees its own change and none of a later migration.
func applyMigrationsThrough(t *testing.T, ctx context.Context, pool *pgxpool.Pool, suffix string) MigrationResult {
	t.Helper()
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	for i, item := range available {
		if strings.TrimSuffix(item.name[strings.IndexByte(item.name, '_')+1:], ".sql") == suffix {
			result, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:i+1])
			if err != nil {
				t.Fatal(err)
			}
			return result
		}
	}
	t.Fatalf("no embedded migration is named %s", suffix)
	return MigrationResult{}
}

// TestSharedPostgresIndexConsolidationKeepsGuarantees applies the V409-007
// migration to the schema before it and compares the catalog: exactly the
// consolidated indexes and the two duplicate UNIQUE constraints are gone, the
// deletion claim index has its claim-order definition, and every other
// index, every primary key and UNIQUE constraint and every foreign key, with
// the unique index it references and its actions, is unchanged. The
// guarantees still reject what they rejected: a Project-owner foreign key an
// owner mismatch and a Project deletion under an Audit, the correlation
// foreign key an Audit Project change while the Audit keeps a finding, and
// an Audit deletion still cascades to its findings.
func TestSharedPostgresIndexConsolidationKeepsGuarantees(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "index_consolidation")
	installMigrationPrefix(t, ctx, pool, previous)
	beforeIndexes := consolidationCatalog(t, ctx, pool, consolidationIndexesSQL)
	beforeConstraints := consolidationCatalog(t, ctx, pool, consolidationConstraintsSQL)
	applyMigrationsThrough(t, ctx, pool, "index_consolidation")
	afterIndexes := consolidationCatalog(t, ctx, pool, consolidationIndexesSQL)
	afterConstraints := consolidationCatalog(t, ctx, pool, consolidationConstraintsSQL)

	dropped := []string{
		"workflow_run_events_time_idx", "workflow_runs_runtime_labels_idx", "workflow_runs_publication_mode_idx",
		"stage_executions_recovery_idx", "eval_experiments_owner_page_idx",
		"projects_project_owner_key", "audits_project_owner_identity",
	}
	wantIndexes := maps.Clone(beforeIndexes)
	for _, name := range dropped {
		if _, present := wantIndexes[name]; !present {
			t.Fatalf("index %s is missing before the migration", name)
		}
		delete(wantIndexes, name)
	}
	wantIndexes["projects_deletion_claim_idx"] = "CREATE INDEX projects_deletion_claim_idx ON projects USING btree " +
		"(deletion_requested_at, project_id) WHERE (lifecycle_state = 'deleting'::text)"
	if !maps.Equal(afterIndexes, wantIndexes) {
		t.Errorf("indexes after the migration differ from the intended change:\n%s", consolidationDiff(wantIndexes, afterIndexes))
	}
	wantConstraints := maps.Clone(beforeConstraints)
	for _, name := range []string{"projects.projects_project_owner_key", "audits.audits_project_owner_identity"} {
		if _, present := wantConstraints[name]; !present {
			t.Fatalf("constraint %s is missing before the migration", name)
		}
		delete(wantConstraints, name)
	}
	if !maps.Equal(afterConstraints, wantConstraints) {
		t.Errorf("constraints after the migration differ from the intended change:\n%s", consolidationDiff(wantConstraints, afterConstraints))
	}
	// The kept unique indexes are what the foreign keys reference.
	for name, want := range map[string]string{
		"audits.audits_project_owner_fkey":                                       "projects_owner_identity r a",
		"eval_dataset_revisions.eval_dataset_revisions_owner_id_project_id_fkey": "projects_owner_identity r a",
		"eval_experiments.eval_experiments_owner_id_project_id_fkey":             "projects_owner_identity r a",
		"workflow_runs.workflow_runs_project_owner_fkey":                         "projects_owner_identity a a",
		"audit_findings.audit_findings_correlation_scope":                        "audits_correlation_scope c a",
		"workflow_run_events.workflow_run_events_event_id_key":                   "workflow_run_events_event_id_key",
	} {
		if definition := afterConstraints[name]; !strings.Contains(definition, want) {
			t.Errorf("%s = %q, want it to reference %q", name, definition, want)
		}
	}

	if _, err := pool.Exec(ctx, `
INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
VALUES ('consolidation-a', 'owner', 'project', 'A', 'create-a', 'sha256:' || repeat('a', 64)),
       ('consolidation-b', 'owner', 'project', 'B', 'create-b', 'sha256:' || repeat('a', 64)),
       ('consolidation-e', 'owner', 'evaluation', 'E', 'create-e', 'sha256:' || repeat('a', 64));
INSERT INTO audits (audit_id, owner_id, project_id, profile_name, profile_version, profile_digest, profile_snapshot,
    input_selection, max_rounds, batch_size, max_items_per_round, max_items_total, max_submitted_runs,
    max_item_run_attempts, max_evidence_bytes)
SELECT audit_id, 'owner', 'consolidation-a', 'profile', '1', 'sha256:' || repeat('a', 64), '{}', '{}', 1, 1, 10, 10, 10, 1, 1024
  FROM unnest(ARRAY['consolidation-audit', 'consolidation-empty']) AS audit_id;
INSERT INTO finding_proposal_receipts (receipt_id, proposal_id, allocation_id, runtime_agent_id, runtime_instance_id,
    stage_execution_id, logical_agent_name, invocation_id, submission_id, client_key, request_digest, run_id, owner_id,
    project_id, workflow_name, workflow_version, workflow_schema_version, workflow_configuration_ref,
    workflow_closure_digest, proposal_ref, proposal_digest, proposal_media_type, proposal_size_bytes, evidence,
    report_projection, anchor_projection)
VALUES ('consolidation-receipt', 'consolidation-receipt', 'allocation', 'runtime', 'instance', 'stage', 'worker',
    'invocation', 'submission', 'client', 'sha256:' || repeat('a', 64), 'run', 'owner', 'consolidation-a',
    'workflow', '1', 'contractor/v1alpha1', '{}', 'sha256:' || repeat('b', 64), '{}', 'sha256:' || repeat('c', 64),
    'application/json', 2, '[]', '{}',
    '{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"not_recorded"}');
INSERT INTO audit_findings (finding_id, audit_id, first_receipt_id, first_proposal_ref)
VALUES ('consolidation-finding', 'consolidation-audit', 'consolidation-receipt', '{}')`); err != nil {
		t.Fatal(err)
	}
	for _, rejected := range []struct {
		name, sql, constraint string
	}{
		{"Audit Project change with a retained finding",
			`UPDATE audits SET project_id = 'consolidation-b' WHERE audit_id = 'consolidation-audit'`,
			"audit_findings_correlation_scope"},
		{"experiment of another owner's Project",
			`INSERT INTO eval_experiments (experiment_id, owner_id, project_id, portable_id, control_mode, name, state,
    max_in_flight, wall_ms) VALUES ('consolidation-experiment', 'stranger', 'consolidation-e', 'portable', 'server', 'X', 'draft', 1, 1000)`,
			"eval_experiments_owner_id_project_id_fkey"},
		{"dataset of another owner's Project",
			`INSERT INTO eval_dataset_revisions (owner_id, project_id, dataset_id, revision, document, metadata)
VALUES ('stranger', 'consolidation-e', 'dataset', '1', convert_to('{}', 'UTF8'), '{}'::jsonb)`,
			"eval_dataset_revisions_owner_id_project_id_fkey"},
		{"deletion of a Project with an Audit",
			`DELETE FROM projects WHERE project_id = 'consolidation-a'`,
			"audits_project_owner_fkey"},
	} {
		_, err := pool.Exec(ctx, rejected.sql)
		var failure *pgconn.PgError
		if Classify(err) != ClassReferenceViolation || !errors.As(err, &failure) || failure.ConstraintName != rejected.constraint {
			t.Errorf("%s: error = %v, want a foreign-key violation of %s", rejected.name, err, rejected.constraint)
		}
	}
	// Without a finding the same change passes the correlation check, and
	// the Audit's deletion cascades to its finding.
	if _, err := pool.Exec(ctx, `UPDATE audits SET project_id = 'consolidation-b' WHERE audit_id = 'consolidation-empty'`); err != nil {
		t.Fatalf("Project change of an Audit without findings: %v", err)
	}
	var findings int
	err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `DELETE FROM audits WHERE audit_id = 'consolidation-audit'`); err != nil {
			return err
		}
		return tx.QueryRow(ctx, `SELECT count(*) FROM audit_findings WHERE finding_id = 'consolidation-finding'`).Scan(&findings)
	})
	if err != nil || findings != 0 {
		t.Fatalf("Audit deletion left %d findings: %v", findings, err)
	}
}

func consolidationCatalog(t *testing.T, ctx context.Context, pool *pgxpool.Pool, query string) map[string]string {
	t.Helper()
	rows, err := pool.Query(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	catalog := map[string]string{}
	for rows.Next() {
		var name, definition string
		if err := rows.Scan(&name, &definition); err != nil {
			t.Fatal(err)
		}
		catalog[name] = definition
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(catalog) == 0 {
		t.Fatal("the catalog query returned nothing")
	}
	return catalog
}

func consolidationDiff(want, got map[string]string) string {
	var lines []string
	for name, definition := range want {
		if got[name] != definition {
			lines = append(lines, "  want "+name+": "+definition+"\n   got "+name+": "+got[name])
		}
	}
	for name, definition := range got {
		if _, wanted := want[name]; !wanted {
			lines = append(lines, "  unexpected "+name+": "+definition)
		}
	}
	slices.Sort(lines)
	return strings.Join(lines, "\n")
}
