//go:build integration

package restore_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// This is an offline recovery rehearsal: writers are quiescent during both
// snapshots. Filesystem objects are backed up with the database, not inferred
// from its metadata. Only databases created by this test are ever dropped.
func TestBackupRestorePreservesExactArtifactsAndCAS(t *testing.T) {
	t.Run("readable tool failure", checkReadableToolFailure)
	for _, backend := range []blob.Backend{blob.Postgres, blob.Filesystem} {
		t.Run(string(backend), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
			defer cancel()
			admin, config := restoreAdmin(t, ctx)
			source, sourceName := restoreDatabase(t, ctx, admin, config)
			target, targetName := restoreDatabase(t, ctx, admin, config)
			initial, err := postgres.ApplyMigrations(ctx, source, postgres.MigrationBudgets{})
			if err != nil || len(initial.AppliedVersions) == 0 {
				t.Fatalf("fresh migration: %+v, %v", initial, err)
			}
			sourceDir, backupDir, targetDir := t.TempDir(), filepath.Join(t.TempDir(), "blobs"), t.TempDir()
			sourceStorage := restoreBlobStorage(t, ctx, backend, sourceDir)
			projects := projectstore.NewPostgresStore(source)
			params := projectstore.CreateParams{
				ProjectID: "restore-project", OwnerID: "restore-owner", Kind: projectstore.KindProject,
				Name: "Restored project", IdempotencyKey: "restore-project-create",
				RequestDigest: "sha256:" + strings.Repeat("a", 64),
			}
			project, _, err := projects.Create(ctx, params)
			if err != nil {
				t.Fatal(err)
			}
			store, err := artifacts.NewService(sourceStorage.Repository(source)).Project(project.ProjectID)
			if err != nil {
				t.Fatal(err)
			}
			ref := artifacts.ArtifactRef{Namespace: "docs", Name: "recovery"}
			firstBody := []byte("retained first revision: Unicode Привет and <>&\n")
			first, err := store.Write(ctx, ref, artifacts.Payload{MediaType: "text/plain", Data: firstBody}, nil)
			if err != nil {
				t.Fatal(err)
			}
			latestBody := bytes.Repeat([]byte("binary\x00payload\xff\n"), 1024)
			latest, err := store.Write(ctx, ref, artifacts.Payload{MediaType: "application/octet-stream", Data: latestBody}, first.Ref.Revision)
			if err != nil {
				t.Fatal(err)
			}
			before, err := store.Metadata(ctx, latest.Ref)
			if err != nil {
				t.Fatal(err)
			}
			seedTableFunctionRows(t, ctx, source)
			functionRows := tableFunctionRows(t, ctx, source)
			for table, rows := range functionRows {
				if rows.count == 0 {
					t.Fatalf("%s calls a schema function but holds no row, so the restore would not run it; add a row to tableFunctionRowsSQL", table)
				}
			}
			archive := filepath.Join(t.TempDir(), "database.dump")
			runPostgresTool(t, ctx, config, sourceName, "pg_dump", "--format=custom", "--no-owner", "--file", archive)
			if backend == blob.Filesystem {
				if err := os.CopyFS(backupDir, os.DirFS(sourceDir)); err != nil {
					t.Fatal(err)
				}
				// Recovery must not accidentally continue reading the original store.
				if err := os.RemoveAll(sourceDir); err != nil {
					t.Fatal(err)
				}
			}
			source.Close()
			if _, err := admin.Exec(ctx, "DROP DATABASE "+pgx.Identifier{sourceName}.Sanitize()); err != nil {
				t.Fatal(err)
			}
			runPostgresTool(t, ctx, config, targetName, "pg_restore", "--exit-on-error", "--no-owner", "--dbname", targetName, archive)
			restored, err := postgres.ApplyMigrations(ctx, target, postgres.MigrationBudgets{})
			if err != nil || len(restored.AppliedVersions) != 0 || restored.CurrentVersion != initial.CurrentVersion {
				t.Fatalf("restored migration ledger: %+v, %v", restored, err)
			}
			if restoredRows := tableFunctionRows(t, ctx, target); !maps.Equal(restoredRows, functionRows) {
				t.Fatalf("restored rows of tables that call schema functions differ:\n%v\nwant\n%v", restoredRows, functionRows)
			}
			targetStorage := restoreBlobStorage(t, ctx, backend, targetDir)
			recoveredStore, err := artifacts.NewService(targetStorage.Repository(target)).Project(project.ProjectID)
			if err != nil {
				t.Fatal(err)
			}
			if backend == blob.Filesystem {
				if _, err := recoveredStore.Read(ctx, latest.Ref); err == nil {
					t.Fatal("database-only restore fabricated missing filesystem payload")
				}
				if err := os.CopyFS(targetDir, os.DirFS(backupDir)); err != nil {
					t.Fatal(err)
				}
			}
			for _, expected := range []struct {
				ref  artifacts.ArtifactRef
				body []byte
			}{{first.Ref, firstBody}, {latest.Ref, latestBody}, {ref, latestBody}} {
				read, err := recoveredStore.Read(ctx, expected.ref)
				if err != nil || !bytes.Equal(read.Payload.Data, expected.body) {
					t.Fatalf("restored exact/current content differs: %v", err)
				}
			}
			after, err := recoveredStore.Metadata(ctx, latest.Ref)
			if err != nil || after.Digest != before.Digest || after.Size != before.Size || after.MediaType != before.MediaType {
				t.Fatalf("restored metadata differs: %v", err)
			}
			replayed, created, err := projectstore.NewPostgresStore(target).Create(ctx, params)
			if err != nil || created || replayed.ProjectID != project.ProjectID || replayed.Revision != project.Revision {
				t.Fatalf("restored idempotency receipt: %+v, created=%v, %v", replayed, created, err)
			}
			if _, err := recoveredStore.Write(ctx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("stale")}, first.Ref.Revision); !errors.Is(err, artifacts.ErrArtifactConflict) {
				t.Fatalf("restored stale CAS: %v", err)
			}
			if _, err := recoveredStore.Write(ctx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("resumed")}, latest.Ref.Revision); err != nil {
				t.Fatalf("resumed write: %v", err)
			}
			t.Logf("restored schema %d, two exact revisions, idempotency, CAS and the rows of %d tables that call schema functions; backend=%s",
				restored.CurrentVersion, len(functionRows), backend)
		})
	}
}

func restoreAdmin(t *testing.T, ctx context.Context) (*pgxpool.Pool, *pgxpool.Config) {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL must point to disposable PostgreSQL")
	}
	config, err := pgxpool.ParseConfig(url)
	if err != nil {
		t.Fatal("invalid test database configuration")
	}
	config.MaxConns, config.MinConns, config.MinIdleConns = 2, 0, 0
	admin, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal("open isolated test database")
	}
	t.Cleanup(admin.Close)
	return admin, config
}

func restoreDatabase(t *testing.T, ctx context.Context, admin *pgxpool.Pool, config *pgxpool.Config) (*pgxpool.Pool, string) {
	t.Helper()
	var random [8]byte
	if _, err := rand.Read(random[:]); err != nil {
		t.Fatal(err)
	}
	name := "contractor_restore_" + hex.EncodeToString(random[:])
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+pgx.Identifier{name}.Sanitize()); err != nil {
		t.Fatal(err)
	}
	copy := config.Copy()
	copy.ConnConfig.Database = name
	pool, err := pgxpool.NewWithConfig(ctx, copy)
	if err != nil {
		t.Fatal("open fresh restore database")
	}
	t.Cleanup(func() {
		pool.Close()
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, "DROP DATABASE IF EXISTS "+pgx.Identifier{name}.Sanitize()); err != nil {
			t.Errorf("clean up owned restore database: %v", err)
		}
	})
	return pool, name
}

func restoreBlobStorage(t *testing.T, ctx context.Context, backend blob.Backend, directory string) *artifactpostgres.Storage {
	t.Helper()
	if backend != blob.Filesystem {
		return artifactpgtest.NewStorage()
	}
	store, err := filesystem.Open(ctx, directory)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return artifactpgtest.NewConfigured(blob.NewRuntime(store, nil), artifactpostgres.DefaultHistoryPolicy())
}

func runPostgresTool(t *testing.T, ctx context.Context, config *pgxpool.Config, database, tool string, args ...string) {
	t.Helper()
	if err := postgresTool(ctx, config, database, tool, args...); err != nil {
		t.Fatal(err)
	}
}

// postgresTool runs a PostgreSQL client tool against database and returns
// its output, with the configured password hidden, when it fails.
func postgresTool(ctx context.Context, config *pgxpool.Config, database, tool string, args ...string) error {
	command := exec.CommandContext(ctx, tool, args...)
	command.Env = append(os.Environ(),
		"PGHOST="+config.ConnConfig.Host,
		"PGPORT="+strconv.Itoa(int(config.ConnConfig.Port)),
		"PGUSER="+config.ConnConfig.User,
		"PGPASSWORD="+config.ConnConfig.Password,
		"PGDATABASE="+database,
		"PGSSLMODE=disable",
	)
	if output, err := command.CombinedOutput(); err != nil {
		return fmt.Errorf("%s failed: %w; %s", tool, err, redactPassword(string(output), config.ConnConfig.Password))
	}
	return nil
}

// redactPassword hides a configured password in tool output. An empty
// password (a local socket role) has nothing to hide; replacing it would put
// the marker between every character.
func redactPassword(output, password string) string {
	if password == "" {
		return output
	}
	return strings.ReplaceAll(output, password, "<redacted>")
}

// checkReadableToolFailure runs a failing pg_restore through the helper of
// the restore rehearsal: the failure keeps the tool's own words whatever the
// configured password, an empty one included, and redaction hides only a
// configured password.
func checkReadableToolFailure(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	_, config := restoreAdmin(t, ctx)
	err := postgresTool(ctx, config, config.ConnConfig.Database, "pg_restore", "--list", filepath.Join(t.TempDir(), "missing.dump"))
	if err == nil || !strings.Contains(err.Error(), "could not open input file") || !strings.Contains(err.Error(), "missing.dump") {
		t.Fatalf("pg_restore of a missing archive: %v", err)
	}
	for _, test := range []struct{ output, password, want string }{
		{"pg_restore: error: could not open input file", "", "pg_restore: error: could not open input file"},
		{"password authentication failed for secret-pass, not secret", "secret-pass", "password authentication failed for <redacted>, not secret"},
	} {
		if got := redactPassword(test.output, test.password); got != test.want {
			t.Errorf("redactPassword(%q, %q) = %q, want %q", test.output, test.password, got, test.want)
		}
	}
}

// tableFunctionRowsSQL stores, next to the Project "restore-project" of owner
// "restore-owner", a row in every table whose CHECK constraints or generated
// columns call a schema function, with the columns those functions read set,
// so pg_restore runs each function while it loads the rows (PER-R46). The
// Audit-managed Run needs its AuditExecution first and is then recorded on it,
// as the Audit dispatcher does.
var tableFunctionRowsSQL = []string{
	`INSERT INTO runtime_agent_principals(runtime_agent_id,labels,label_revision,created_by,created_at,updated_by,updated_at)
VALUES(repeat('e',64),'{gpu}',1,'restore-admin',clock_timestamp(),'restore-admin',clock_timestamp())`,
	`INSERT INTO model_selection_settings(default_model,role_models) VALUES('qwen3-32b','{"planner":"qwen3-80b"}')`,
	`INSERT INTO workflow_runs(run_id,owner_id,project_id,workflow_name,workflow_version,workflow_schema_version,
 workflow_snapshot,parameters,state,state_reason_code,finished_at,cancellation_schema_version,run_cancellation,
 runtime_labels,runtime_config_snapshot,scheduler_lane_class,metadata_labels,required_placement_labels)
VALUES('restore-run','restore-owner','restore-project','workflow','1','v1','{}','{}','cancelled','user_cancelled',
 clock_timestamp(),'v1','{"code":"user_cancelled","requestedAt":"2026-10-30T00:00:00Z","requestedBy":"restore-owner"}',
 '{gpu}',jsonb_build_object(
  'default',jsonb_build_object('label','default','explicit',false,'bindingRevision',1,
   'config',jsonb_build_object('name','contractor-empty','version','1','digest','sha256:'||repeat('1',64))),
  'labels',jsonb_build_array(jsonb_build_object('label','gpu','explicit',true,'bindingRevision',2,
   'config',jsonb_build_object('name','gpu','version','1.0','digest','sha256:'||repeat('2',64)))),
  'llmCredentialIds',jsonb_build_array('llm-a','llm-b'),'runtimeCredentialIds',jsonb_build_array('runtime-a')),
 'model','{"team":"blue"}','{cloud,cloud/eu}')`,
	`INSERT INTO stage_executions(stage_execution_id,run_id,stage_name,attempt,stage_spec_schema_version,stage_spec_snapshot,
 stage_context_schema_version,stage_context_snapshot,state,state_reason_code,termination_schema_version,stage_termination,
 abort_id,abort_deadline,terminal_at)
VALUES('restore-stage-cancelled','restore-run','review',1,'v1','{}','v1','{}','cancelled','user_cancelled','v1',
 '{"outcome":"cancelled","code":"user_cancelled","message":"Cancelled by the owner","retryable":false,"phase":"preparing","occurredAt":"2026-10-30T00:00:00Z"}',
 'restore-abort',clock_timestamp(),clock_timestamp())`,
	`INSERT INTO stage_executions(stage_execution_id,run_id,stage_name,attempt,stage_spec_schema_version,stage_spec_snapshot,
 stage_context_schema_version,stage_context_snapshot,state,state_reason_code,planner_session_id,planner_invocation_id,
 planner_started_at,candidate_result_schema_version,candidate_stage_result,accepted_result_schema_version,
 accepted_stage_result,finalization_id,finalization_deadline,terminal_at)
VALUES('restore-stage','restore-run','build',1,'v1','{}','v1','{}','succeeded','completed','restore-session',
 'restore-invocation',clock_timestamp(),'v1','{"apiVersion":"contractor/v1alpha1","outcome":"succeeded","summary":"Built","artifacts":{}}',
 'v1','{"apiVersion":"contractor/v1alpha1","outcome":"succeeded","summary":"Built","artifacts":{}}',
 'restore-finalization',clock_timestamp(),clock_timestamp())`,
	`INSERT INTO planner_sessions(session_id,stage_execution_id,invocation_id,state_schema_version,state)
VALUES('restore-session','restore-stage','restore-invocation','v1','{"scan":{"state":{"jobs":[]}}}')`,
	`INSERT INTO stage_allocations(allocation_id,stage_execution_id,logical_agent_name,namespace,agent_template_ref,
 worker_runtime_ref,runtime_agent_instance_id,runtime_agent_id,runtime_agent_label_revision,
 runtime_configuration_schema_version,runtime_configuration,performance_collection_policy,runtime_placement_labels)
VALUES('restore-allocation','restore-stage','worker','build','{}','{}','restore-instance',repeat('e',64),1,
 'contractor.runtime-config-provenance/v2',jsonb_build_object(
  'modelPolicy',jsonb_build_object('policyId','default','version','1','digest','sha256:'||repeat('3',64)),
  'origins','{}'::jsonb,'provenance',jsonb_build_object('agentLabels','[]'::jsonb,'default','{}'::jsonb,
   'llmGatewayConfig','{}'::jsonb,'runLabels','[]'::jsonb,'runtimeAdapters','[]'::jsonb,'runtimeCredentialRefs','[]'::jsonb)),
 'requested','{cloud,cloud/eu}')`,
	`INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,profile_snapshot,
 input_selection,baseline_snapshot,state,max_rounds,batch_size,max_items_per_round,max_items_total,
 max_submitted_runs,max_item_run_attempts,max_evidence_bytes,started_at)
VALUES('restore-audit','restore-owner','restore-project','restore-profile','1','sha256:'||repeat('b',64),
 '{"workflows":{"check":{"kind":"check"}}}','{"inputs":{}}','{"schema":"contractor.audit.baseline.v1","inputs":{},"skills":[]}',
 'active',1,1,1,1,1,1,67108864,clock_timestamp())`,
	`INSERT INTO audit_tasks(task_id,audit_id,source_key,source,document,batch_digest,ordinal,iteration)
VALUES('restore-task','restore-audit','sha256:'||repeat('d',64),'{}',
 '{"schema":"contractor.audit.task.v1","audit_id":"restore-audit","task_id":"restore-task","workflow":"check"}',
 'sha256:'||repeat('e',64),1,1)`,
	`INSERT INTO audit_executions(execution_id,audit_id,submission_key,request_digest,workflow_role,lane_class)
VALUES('restore-execution','restore-audit','restore-submission','sha256:'||repeat('a',64),'check','model')`,
	`INSERT INTO workflow_runs(run_id,owner_id,project_id,workflow_name,workflow_version,workflow_schema_version,
 workflow_snapshot,parameters,state,state_reason_code,runtime_labels,runtime_config_snapshot,scheduler_lane_class,
 publication_mode,audit_execution_id,audit_submission_key,audit_input_materialization)
VALUES('restore-audit-run','restore-owner','restore-project','check','1','v1','{}','{}','pending','submitted','{}',
 jsonb_build_object('default',jsonb_build_object('label','default','explicit',false,'bindingRevision',1,
   'config',jsonb_build_object('name','contractor-empty','version','1','digest','sha256:'||repeat('1',64))),
  'labels','[]'::jsonb,'llmCredentialIds','[]'::jsonb,'runtimeCredentialIds','[]'::jsonb),
 'model','audit-managed','restore-execution','restore-submission',
 json_build_object('identity','sha256:'||repeat('4',64),'revision',1,'complete',false,
  'deadlineAt','2026-10-30T00:00:00Z','checkpoint','{}'::json,'refs','[]'::json))`,
	`UPDATE audit_executions SET state='submitted',run_id='restore-audit-run',
 run_provenance='{"schema":"contractor.audit.run-provenance.v1","runId":"restore-audit-run"}'
WHERE execution_id='restore-execution'`,
	`INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance,display_ref)
VALUES('restore-audit','report','{"namespace":"audit-report","name":"report","revision":"r1"}','sha256:'||repeat('5',64),
 'application/json',2,'{"evidence":{"kind":"note","id":"evidence-1","artifact":{"namespace":"notes","name":"n","revision":"r1"}}}',
 'Report <&>')`,
	`INSERT INTO finding_proposal_receipts(receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,
 stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,request_digest,run_id,owner_id,project_id,
 workflow_name,workflow_version,workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,
 proposal_digest,proposal_media_type,proposal_size_bytes,evidence,report_projection,anchor_projection, checks_projection)
VALUES('restore-receipt','restore-proposal','restore-allocation',repeat('e',64),'restore-instance','restore-stage','worker',
 'restore-invocation','restore-submission','restore-client','sha256:'||repeat('a',64),'restore-run','restore-owner',
 'restore-project','workflow','1','contractor/v1alpha1','{}','sha256:'||repeat('b',64),
 '{"namespace":"findings","name":"proposal","revision":"r1"}','sha256:'||repeat('c',64),'application/json',2,'[]',
 '{"preconditions":[]}',
 '{"schema":"contractor.finding.anchors-projection.v1","availability":"available","keys":[{"issueKey":"'||repeat('6',64)||'"}]}', json_build_object('schema','contractor.audit.finding-checks-projection.v1','proposal_digest',('sha256:'||repeat('c',64))::text,'size_bytes',(2)::bigint,'discarded',false,'subject',NULL,'limitations',NULL,'checks','[]'::json))`,
	`INSERT INTO project_findings(owner_id,finding_id,project_id,first_receipt_id,first_proposal_ref,
 title,subject_kind,subject_key,subject_truncated,severity_suggestion,
 correlation_algorithm,correlation_value,correlation_group_key,correlation_reason)
VALUES('restore-owner','restore-finding','restore-project','restore-receipt',
 jsonb_build_object('ref',jsonb_build_object('namespace','findings','name','proposal','revision','r1'),
 'digest','sha256:'||repeat('c',64),'mediaType','application/json','sizeBytes',2),
 'Restored issue','','',false,'','contractor/snapshot-issue-anchor/v1',repeat('6',64),
 contractor_finding_group_key('restore-project','contractor/snapshot-issue-anchor/v1',repeat('6',64)),'')`,
}

// tableFunctionTablesSQL lists the tables of the current schema whose CHECK
// constraints or generated columns call a function outside pg_catalog: the
// tables whose rows make pg_restore run such a function.
const tableFunctionTablesSQL = `
SELECT DISTINCT relation.relname::text
  FROM pg_depend AS dependency
  JOIN pg_proc AS function ON function.oid = dependency.refobjid
  LEFT JOIN pg_constraint AS constraint_
    ON dependency.classid = 'pg_constraint'::regclass AND constraint_.oid = dependency.objid AND constraint_.contype = 'c'
  LEFT JOIN pg_attrdef AS attrdef ON dependency.classid = 'pg_attrdef'::regclass AND attrdef.oid = dependency.objid
  LEFT JOIN pg_attribute AS attribute
    ON attribute.attrelid = attrdef.adrelid AND attribute.attnum = attrdef.adnum AND attribute.attgenerated <> ''
  JOIN pg_class AS relation ON relation.oid = coalesce(constraint_.conrelid, attribute.attrelid)
 WHERE dependency.refclassid = 'pg_proc'::regclass
   AND function.pronamespace <> 'pg_catalog'::regnamespace
   AND relation.relnamespace = current_schema()::regnamespace
 ORDER BY 1`

// tableRows is the number of rows of a table and a digest of their text,
// generated columns included.
type tableRows struct {
	count  int64
	digest string
}

// seedTableFunctionRows stores tableFunctionRowsSQL in one transaction.
func seedTableFunctionRows(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	if err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		for _, statement := range tableFunctionRowsSQL {
			if _, err := tx.Exec(ctx, statement); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatalf("store rows that call table functions: %v", err)
	}
}

// tableFunctionRows returns the rows of every table whose CHECK constraints or
// generated columns call a schema function, by table name.
func tableFunctionRows(t *testing.T, ctx context.Context, pool *pgxpool.Pool) map[string]tableRows {
	t.Helper()
	rows, err := pool.Query(ctx, tableFunctionTablesSQL)
	if err != nil {
		t.Fatal(err)
	}
	tables, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatal(err)
	}
	if len(tables) == 0 {
		t.Fatal("no table calls a schema function; the catalog query is broken")
	}
	result := make(map[string]tableRows, len(tables))
	for _, table := range tables {
		var found tableRows
		if err := pool.QueryRow(ctx, `SELECT count(*), coalesce(md5(string_agg(row::text, E'\n' ORDER BY row::text)), '') FROM `+
			pgx.Identifier{table}.Sanitize()+` AS row`).Scan(&found.count, &found.digest); err != nil {
			t.Fatal(err)
		}
		result[table] = found
	}
	return result
}
