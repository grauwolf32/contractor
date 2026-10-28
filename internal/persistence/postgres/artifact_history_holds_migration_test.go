package postgres

import (
	"context"
	"errors"
	"maps"
	"os"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// holdKey names one revision as the hold assertions of these tests list it.
type holdKey struct {
	kind, id, namespace, name, revision string
}

// Migration artifact_history_retention counts, on every stored revision, the
// holders that name it exactly: both ends of lineage, the publications of a
// Run's outputs, StageExecution context pins, Planner session state, the
// Skill sources of Runs and Audit baselines, Audit inputs and finding
// proposal receipts. An Eval dataset's case inputs are availability-only
// references and protect nothing (DEL-R64). Afterwards each holder keeps the
// counts current, and only an unheld revision other than the current one may
// be removed, and only under the retention setting.
func TestSharedPostgresArtifactHistoryHoldsMigrationCountsEveryHolder(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "artifact_history_retention")
	installMigrationPrefix(t, ctx, pool, previous)
	seedHistoryHolders(t, ctx, pool, `{"artifacts": {"pin": {"artifact": {"namespace": "out", "name": "o", "revision": "o1"}}}}`)
	// Preserve this historical fixture before the incompatible required-conditions transition.
	applyMigrationsThrough(t, ctx, pool, "audit_admission_chain_transition")
	want := map[holdKey]int{
		{"user", "owner", "notes", "n", "n1"}:          1, // lineage source
		{"run", "run-h", "inputs", "f", "f1"}:          1, // lineage target
		{"user", "owner", "skills", "s", "s1"}:         2, // Run and Audit baseline Skill
		{"project", "proj", "inputs", "i", "i1"}:       1, // one Audit selecting it and naming it in its baseline
		{"run", "run-h", "out", "o", "o1"}:             1, // StageContext pin
		{"run", "run-h", "out", "o", "o2"}:             1, // receipt proposal
		{"run", "run-h", "out", "o", "o3"}:             2, // Planner session plan, receipt evidence
		{"project", "proj", "outputs", "report", "p1"}: 1, // output publication target
		{"run", "run-h", "outputs", "report", "q1"}:    1, // output publication source
	}
	if got := storedHolds(t, ctx, pool); !maps.Equal(got, want) {
		t.Fatalf("holds after the migration = %v\nwant %v", got, want)
	}

	// A summary continuation's summary and artifacts are pins of its
	// StageExecution like its context's.
	var continuation string
	if err := pool.QueryRow(ctx, `SELECT contractor_stage_execution_artifact_holds('run-h',
'{"continuation": {"summaryArtifact": {"namespace": "out", "name": "o", "revision": "o2"},
  "artifacts": {"slot": {"namespace": "out", "name": "o", "revision": "o5"}}}}', NULL, NULL)::text`).
		Scan(&continuation); err != nil {
		t.Fatal(err)
	}
	if continuation != `[["run", "run-h", "out", "o", "o2"], ["run", "run-h", "out", "o", "o5"]]` {
		t.Fatalf("continuation holds = %s", continuation)
	}

	// Holders keep the counts current.
	for _, statement := range []string{
		`UPDATE audits SET input_selection = '{"inputs": {"k": {"ref": {"namespace": "inputs", "name": "i", "revision": "i2"}}}}'
WHERE audit_id = 'audit-h'`,
		`UPDATE planner_sessions SET state = '{"scan": {"state": {"jobs": [{"report": {"namespace": "out", "name": "o", "revision": "o4"}}]}}}'
WHERE session_id = 'session-h'`,
	} {
		if _, err := pool.Exec(ctx, statement); err != nil {
			t.Fatal(err)
		}
	}
	want[holdKey{"project", "proj", "inputs", "i", "i2"}] = 1
	want[holdKey{"run", "run-h", "out", "o", "o3"}] = 1
	want[holdKey{"run", "run-h", "out", "o", "o4"}] = 1
	if got := storedHolds(t, ctx, pool); !maps.Equal(got, want) {
		t.Fatalf("holds after holder updates = %v\nwant %v", got, want)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM audits WHERE audit_id = 'audit-h'`); err != nil {
		t.Fatal(err)
	}
	delete(want, holdKey{"project", "proj", "inputs", "i", "i2"})
	want[holdKey{"user", "owner", "skills", "s", "s1"}] = 1
	delete(want, holdKey{"project", "proj", "inputs", "i", "i1"})
	if got := storedHolds(t, ctx, pool); !maps.Equal(got, want) {
		t.Fatalf("holds after the Audit's deletion = %v\nwant %v", got, want)
	}
	// A strict holder cannot record a missing revision.
	_, err := pool.Exec(ctx, `INSERT INTO artifact_lineage (
    target_scope_kind, target_scope_id, target_namespace, target_name, target_revision,
    source_scope_kind, source_scope_id, source_namespace, source_name, source_revision, lineage_kind)
VALUES ('run', 'run-h', 'out', 'o', 'o5', 'user', 'owner', 'notes', 'n', 'missing', 'input_fork')`)
	var rejection *pgconn.PgError
	if !errors.As(err, &rejection) || Classify(err) != ClassReferenceViolation {
		t.Fatalf("lineage to a missing revision = %v, want a reference violation", err)
	}
	_, err = pool.Exec(ctx, `SELECT contractor_artifact_hold_apply('[["run", "run-h", "out", "o", "missing"]]', '[]', true)`)
	if Classify(err) != ClassReferenceViolation || ConstraintName(err) != "artifact_exact_hold" {
		t.Fatalf("strict hold of a missing revision of a live scope = %v, want artifact_exact_hold", err)
	}
	// A reference into a scope that no longer exists holds nothing: that
	// scope's purge removed every revision it could name.
	if _, err := pool.Exec(ctx, `SELECT contractor_artifact_hold_apply('[["run", "run-purged", "out", "o", "o1"]]', '[]', true)`); err != nil {
		t.Fatalf("strict hold into a purged scope: %v", err)
	}

	// Revisions stay immutable outside the retention exception, which only
	// removes an unheld revision other than the binding's current one.
	removal := func(revision string, retention bool) error {
		return InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			if retention {
				if _, err := tx.Exec(ctx, `SELECT set_config('contractor.artifact_retention', 'evict', true)`); err != nil {
					return err
				}
			}
			_, err := tx.Exec(ctx, `DELETE FROM artifact_binding_revisions
WHERE scope_kind = 'user' AND scope_id = 'owner' AND namespace = 'notes' AND name = 'n' AND revision = $1`, revision)
			return err
		})
	}
	for _, refused := range []struct {
		revision  string
		retention bool
	}{{"n3", false}, {"n1", true}, {"n4", true}} {
		if err := removal(refused.revision, refused.retention); Classify(err) != ClassRejected {
			t.Fatalf("removal of %s (retention %v) = %v, want the immutability rejection", refused.revision, refused.retention, err)
		}
	}
	if err := removal("n3", true); err != nil {
		t.Fatalf("retention of an unheld historical revision: %v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE artifact_binding_revisions SET version_id = 'v-n1'
WHERE scope_kind = 'user' AND scope_id = 'owner' AND namespace = 'notes' AND name = 'n' AND revision = 'n2'`); Classify(err) != ClassRejected {
		t.Fatalf("revision update = %v, want the immutability rejection", err)
	}
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SELECT set_config('contractor.artifact_retention', 'evict', true)`); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `DELETE FROM artifact_versions WHERE version_id = 'v-n3'`)
		return err
	})
	if err != nil {
		t.Fatalf("retention of an unreferenced version: %v", err)
	}
}

// The migration stops on a StageExecution that pins a revision its Run's live
// scope no longer has: that reference was proven when it was recorded, so its
// absence is corrupt data, not an unprotected reference.
func TestSharedPostgresArtifactHistoryHoldsMigrationRejectsADanglingPin(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "artifact_history_retention")
	installMigrationPrefix(t, ctx, pool, previous)
	seedHistoryHolders(t, ctx, pool, `{"artifacts": {"pin": {"artifact": {"namespace": "out", "name": "o", "revision": "gone"}}}}`)
	_, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	var rejection *pgconn.PgError
	if !errors.As(err, &rejection) || rejection.ConstraintName != "artifact_exact_hold" {
		t.Fatalf("migration over a dangling pin = %v, want the artifact_exact_hold rejection", err)
	}
}

// seedHistoryHolders stores, on the schema before the migration, revisions of
// four bindings and one holder of every kind, the StageExecution with
// stageContext.
func seedHistoryHolders(t *testing.T, ctx context.Context, pool *pgxpool.Pool, stageContext string) {
	t.Helper()
	ref := func(namespace, name, revision string) string {
		return `{"namespace": "` + namespace + `", "name": "` + name + `", "revision": "` + revision + `"}`
	}
	err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, statement := range []struct {
			sql       string
			arguments []any
		}{
			{`INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
VALUES ('proj', 'owner', 'project', 'Holds', 'create-proj', 'sha256:' || repeat('a', 64))`, nil},
			{`INSERT INTO workflow_runs (
    run_id, owner_id, project_id, workflow_name, workflow_version, workflow_schema_version, workflow_snapshot,
    parameters, runtime_labels, runtime_config_snapshot, scheduler_lane_class, state, state_reason_code,
    started_at, skill_snapshot
) VALUES ('run-h', 'owner', 'proj', 'workflow', '1', 'v1', '{}', '{}', '{}', $1::jsonb, 'model', 'running', 'test',
          clock_timestamp(), $2::jsonb)`, []any{builtInRunRuntimeSnapshot, `[{"name": "skill", "source": ` + ref("skills", "s", "s1") + `}]`}},
			{`INSERT INTO artifact_scopes (scope_kind, scope_id)
VALUES ('user', 'owner'), ('project', 'proj'), ('run', 'run-h')`, nil},
			{`INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend)
VALUES (sha256('\x78'::bytea), '\x78'::bytea, 1, 'postgresql')`, nil},
			{`INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
SELECT 'v-' || revision, sha256('\x78'::bytea), 'text/plain'
FROM unnest(ARRAY['s1', 's2', 'n1', 'n2', 'n3', 'n4', 'i1', 'i2', 'o1', 'o2', 'o3', 'o4', 'o5', 'f1', 'p1', 'q1']) AS revision`, nil},
			{`INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
SELECT binding.kind, binding.id, binding.namespace, binding.name, revision, 'v-' || revision
FROM (VALUES ('user', 'owner', 'skills', 's', ARRAY['s1', 's2']),
             ('user', 'owner', 'notes', 'n', ARRAY['n1', 'n2', 'n3', 'n4']),
             ('project', 'proj', 'inputs', 'i', ARRAY['i1', 'i2']),
             ('project', 'proj', 'outputs', 'report', ARRAY['p1']),
             ('run', 'run-h', 'outputs', 'report', ARRAY['q1']),
             ('run', 'run-h', 'out', 'o', ARRAY['o1', 'o2', 'o3', 'o4', 'o5']),
             ('run', 'run-h', 'inputs', 'f', ARRAY['f1'])) AS binding(kind, id, namespace, name, revisions),
     unnest(binding.revisions) AS revision`, nil},
			{`INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision)
VALUES ('user', 'owner', 'skills', 's', 's2'), ('user', 'owner', 'notes', 'n', 'n4'),
       ('project', 'proj', 'inputs', 'i', 'i2'), ('project', 'proj', 'outputs', 'report', 'p1'),
       ('run', 'run-h', 'outputs', 'report', 'q1'),
       ('run', 'run-h', 'out', 'o', 'o5'), ('run', 'run-h', 'inputs', 'f', 'f1')`, nil},
			{`INSERT INTO artifact_lineage (
    target_scope_kind, target_scope_id, target_namespace, target_name, target_revision,
    source_scope_kind, source_scope_id, source_namespace, source_name, source_revision, lineage_kind)
VALUES ('run', 'run-h', 'inputs', 'f', 'f1', 'user', 'owner', 'notes', 'n', 'n1', 'input_fork')`, nil},
			{`INSERT INTO workflow_run_output_publications (
    run_id, project_id, output_name, status, source_namespace, source_name, source_revision,
    target_namespace, target_name, target_revision)
VALUES ('run-h', 'proj', 'report', 'published', 'outputs', 'report', 'q1', 'outputs', 'report', 'p1')`, nil},
			{`INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt, stage_spec_schema_version, stage_spec_snapshot,
    stage_context_schema_version, stage_context_snapshot, state, state_reason_code
) VALUES ('stage-h', 'run-h', 'stage', 1, 'v1', '{}', 'v1', $1::jsonb, 'preparing', 'test')`, []any{stageContext}},
			{`INSERT INTO planner_sessions (session_id, stage_execution_id, invocation_id, state_schema_version, state)
VALUES ('session-h', 'stage-h', 'invocation', 'v1', $1::jsonb)`,
				[]any{`{"scan": {"state": {"plan": ` + ref("out", "o", "o3") + `}}}`}},
			{`INSERT INTO audits (audit_id, owner_id, project_id, profile_name, profile_version, profile_digest,
    profile_snapshot, input_selection, baseline_snapshot, max_rounds, batch_size, max_items_per_round,
    max_items_total, max_submitted_runs, max_item_run_attempts, max_evidence_bytes)
VALUES ('audit-h', 'owner', 'proj', 'profile', '1', 'sha256:' || repeat('a', 64), '{}', $1::jsonb, $2::jsonb,
        1, 1, 10, 10, 10, 1, 1024)`, []any{
				`{"inputs": {"k": {"ref": ` + ref("inputs", "i", "i1") + `}}}`,
				`{"inputs": {"k": {"ref": ` + ref("inputs", "i", "i1") + `}}, "skills": [{"source": ` + ref("skills", "s", "s1") + `}]}`,
			}},
			{`INSERT INTO finding_proposal_receipts (receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name, invocation_id, submission_id, client_key,
    request_digest, run_id, owner_id, project_id, workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest, proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence, report_projection, anchor_projection)
VALUES ('receipt-h', 'receipt-h', 'allocation', 'runtime', 'instance', 'stage-h', 'worker', 'invocation',
        'submission', 'client', 'sha256:' || repeat('a', 64), 'run-h', 'owner', 'proj', 'workflow', '1',
        'contractor/v1alpha1', '{}', 'sha256:' || repeat('b', 64), $1::jsonb, 'sha256:' || repeat('c', 64),
        'application/json', 1, $2::jsonb, '{}', $3)`, []any{
				ref("out", "o", "o2"), `[{"ref": ` + ref("out", "o", "o3") + `}]`,
				`{"schema": "contractor.finding.anchors-projection.v1", "availability": "unavailable"}`,
			}},
			{`INSERT INTO eval_dataset_revisions (owner_id, project_id, dataset_id, revision, document, metadata)
VALUES ('owner', 'proj', 'dataset', '1', convert_to($1, 'UTF8'), '{}')`, []any{`{"cases": [{"inputs": {
  "note": {"scope": "user", "scopeId": "owner", "namespace": "notes", "name": "n", "revision": "n2"},
  "input": {"scope": "project", "scopeId": "proj", "namespace": "inputs", "name": "i", "revision": "i2"},
  "foreign": {"scope": "user", "scopeId": "someone-else", "namespace": "notes", "name": "n", "revision": "n3"},
  "unowned": {"scope": "project", "scopeId": "not-owned", "namespace": "inputs", "name": "i", "revision": "i1"}}}]}`}},
		} {
			if _, err := tx.Exec(ctx, statement.sql, statement.arguments...); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// storedHolds returns every revision that holds at least one hold.
func storedHolds(t *testing.T, ctx context.Context, pool *pgxpool.Pool) map[holdKey]int {
	t.Helper()
	rows, err := pool.Query(ctx, `SELECT scope_kind, scope_id, namespace, name, revision, hold_count
FROM artifact_binding_revisions WHERE hold_count <> 0`)
	if err != nil {
		t.Fatal(err)
	}
	holds := map[holdKey]int{}
	for rows.Next() {
		var key holdKey
		var count int
		if err := rows.Scan(&key.kind, &key.id, &key.namespace, &key.name, &key.revision, &count); err != nil {
			t.Fatal(err)
		}
		holds[key] = count
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	return holds
}
