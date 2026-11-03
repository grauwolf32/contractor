package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5/pgxpool"
)

// injectCheckViolationSQL makes every statement that writes an Audit table
// fail as the CHECK constraint that contractor_test_inject_check names would:
// SQLSTATE 23514 naming that constraint. It reaches every writer's error path
// from any fixture state, since a statement trigger fires even when the
// statement's guard matches no row.
const injectCheckViolationSQL = `
CREATE TABLE contractor_test_inject_check (constraint_name text NOT NULL);
CREATE FUNCTION contractor_test_inject_check_violation() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    injected text;
BEGIN
    SELECT constraint_name INTO injected FROM contractor_test_inject_check LIMIT 1;
    IF injected IS NOT NULL THEN
        RAISE EXCEPTION 'injected check violation' USING ERRCODE = 'check_violation', CONSTRAINT = injected;
    END IF;
    RETURN NULL;
END $$;
CREATE TRIGGER contractor_test_inject_check_violation BEFORE INSERT OR UPDATE ON audits
FOR EACH STATEMENT EXECUTE FUNCTION contractor_test_inject_check_violation();
CREATE TRIGGER contractor_test_inject_check_violation BEFORE INSERT OR UPDATE ON audit_items
FOR EACH STATEMENT EXECUTE FUNCTION contractor_test_inject_check_violation();
CREATE TRIGGER contractor_test_inject_check_violation BEFORE INSERT OR UPDATE ON audit_executions
FOR EACH STATEMENT EXECUTE FUNCTION contractor_test_inject_check_violation();`

// Every Audit store writer of a jsonb column whose stored size bound can
// refuse values its Go validation accepts returns a named CHECK rejection as
// auditstore.ErrInvalid with the PostgreSQL cause kept (PER-R13). The public
// API answers that error 400 invalid_request, not the retryable 500 of a raw
// constraint error, and the importer and next-Round paths end on it instead
// of retrying the same write.
func TestSharedPostgresStoredConstraintRejectionsAreInvalid(t *testing.T) {
	f := newCollectingAuditFixture(t, "storedcheck", 1<<20, collectingAuditFixtureOptions{extraPending: true})
	if _, err := f.pool.Exec(f.ctx, injectCheckViolationSQL); err != nil {
		t.Fatal(err)
	}
	result := testExact("audit-storedcheck", "result", "result-r1")
	for _, path := range []struct {
		name, constraint string
		call             func(context.Context) error
	}{
		{"CreateDraft", "audits_profile_snapshot_check", func(ctx context.Context) error {
			_, _, err := f.store.CreateDraft(ctx, auditstore.CreateDraftParams{
				AuditID: "audit-storedcheck-draft", OwnerID: f.audit.OwnerID, ProjectID: f.project.ProjectID,
				Profile: f.audit.Profile, ProfileSnapshot: f.audit.ProfileSnapshot,
				InputSelection: json.RawMessage(`{}`), Limits: f.audit.Limits,
				IdempotencyKey: "storedcheck-create", RequestDigest: testDigest("1"),
			})
			return err
		}},

		{"Collect", "audit_executions_collection_retained_refs_check", func(ctx context.Context) error {
			_, _, err := f.store.Collect(ctx, storedCheckCollect(f, result))
			return err
		}},
	} {
		t.Run(path.name, func(t *testing.T) {
			if _, err := f.pool.Exec(f.ctx, `INSERT INTO contractor_test_inject_check VALUES ($1)`, path.constraint); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if _, err := f.pool.Exec(context.WithoutCancel(f.ctx), `DELETE FROM contractor_test_inject_check`); err != nil {
					t.Error(err)
				}
			})
			assertStoredConstraintInvalid(t, path.call(f.ctx), path.constraint)
		})
	}
}

// A real stored size CHECK, lowered below the values written and keeping its
// name and its measure on the jsonb text, refuses the public Start's Round
// and a collection as auditstore.ErrInvalid, and neither write leaves any
// effect.
func TestPostgresStoredSizeBoundsRefuseAuditWritesAsInvalid(t *testing.T) {
	f := newCollectingAuditFixture(t, "storedbound", 1<<20)
	result := testExact("audit-storedbound", "result", "result-r1")
	lowerStoredBound(t, f.ctx, f.pool, "audit_executions", "audit_executions_collection_retained_refs_check",
		"collection_retained_refs", 2)
	_, inserted, err := f.store.Collect(f.ctx, storedCheckCollect(f, result))
	assertStoredConstraintInvalid(t, err, "audit_executions_collection_retained_refs_check")
	var state string
	getErr := f.pool.QueryRow(f.ctx, `SELECT state FROM audit_executions WHERE execution_id = $1`,
		f.execution.ExecutionID).Scan(&state)
	if inserted || getErr != nil || state != string(f.execution.State) {
		t.Fatalf("execution after a refused collection = (%s, %t, %v), want %s", state, inserted, getErr, f.execution.State)
	}

	draft, _, err := f.store.CreateDraft(f.ctx, auditstore.CreateDraftParams{
		AuditID: "audit-storedbound-start", OwnerID: f.audit.OwnerID, ProjectID: f.project.ProjectID,
		Profile: f.audit.Profile, ProfileSnapshot: f.audit.ProfileSnapshot,
		InputSelection: json.RawMessage(`{}`), Limits: f.audit.Limits,
		IdempotencyKey: "storedbound-create", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	lowerStoredBound(t, f.ctx, f.pool, "audits", "audits_baseline_snapshot_check", "baseline_snapshot", 2)
	_, created, err := f.store.Activate(f.ctx, auditstore.ActivateParams{OwnerID: draft.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision, BaselineSnapshot: f.audit.BaselineSnapshot, DeadlineAt: time.Now().Add(time.Hour), IdempotencyKey: "bounded-start", RequestDigest: testDigest("2")})
	assertStoredConstraintInvalid(t, err, "audits_baseline_snapshot_check")
	after, getErr := f.store.Get(f.ctx, draft.OwnerID, draft.AuditID)
	if created || getErr != nil || after.Revision != draft.Revision || after.State != draft.State || after.CurrentRoundID != nil {
		t.Fatalf("Audit after a refused start = (%+v, %t, %v), want the unchanged draft", after, created, getErr)
	}
}

func storedCheckCollect(f collectingAuditFixture, result auditstore.ExactArtifact) auditstore.CollectParams {
	var task auditstore.Task
	if rows, err := f.store.ListExecutionTasks(f.ctx, f.audit.AuditID, f.execution.ExecutionID); err == nil && len(rows) == 1 {
		task = rows[0]
	}
	material := task.Document.Inputs["source"]
	return auditstore.CollectParams{Claim: f.claim, ReceiptID: "receipt-" + f.execution.ExecutionID, ExecutionID: f.execution.ExecutionID, Disposition: auditstore.CollectionAccepted, RequestDigest: testDigest("7"),
		SourceOutput: &auditstore.ExactArtifact{Ref: material.Ref, Digest: material.Digest, MediaType: material.MediaType, SizeBytes: material.SizeBytes},
		Retained:     []auditstore.ArtifactLink{{Scope: "audit", LogicalKey: "result/" + f.memberID, Artifact: auditstore.ExactArtifact{Ref: material.Ref, Digest: material.Digest, MediaType: material.MediaType, SizeBytes: material.SizeBytes}, SourceProvenance: json.RawMessage("{}")}},
		Tasks:        []auditstore.TaskCollection{{ExecutionItemID: f.memberID, TaskID: f.itemID, Result: &material}}}
}

func lowerStoredBound(t *testing.T, ctx context.Context, pool *pgxpool.Pool, table, constraint, column string, limit int) {
	t.Helper()
	if _, err := pool.Exec(ctx, fmt.Sprintf(
		`ALTER TABLE %s DROP CONSTRAINT %s, ADD CONSTRAINT %s CHECK (octet_length(%s::text) <= %d) NOT VALID`,
		table, constraint, constraint, column, limit,
	)); err != nil {
		t.Fatal(err)
	}
}

func assertStoredConstraintInvalid(t *testing.T, err error, constraint string) {
	t.Helper()
	classified, ok := apierror.Of(err)
	if !errors.Is(err, auditstore.ErrInvalid) || persistencepostgres.Classify(err) != persistencepostgres.ClassInvalid ||
		persistencepostgres.ConstraintName(err) != constraint || !ok || classified.ErrorKind() != apierror.InvalidRequest {
		t.Fatalf("refused write = %v, want auditstore.ErrInvalid wrapping %s", err, constraint)
	}
}
