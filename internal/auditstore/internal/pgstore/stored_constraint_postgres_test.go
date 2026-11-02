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
	"github.com/grauwolf32/contractor/internal/config/resolved"
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
	if f.audit.CurrentRoundID == nil {
		t.Fatal("fixture Audit has no current Round")
	}
	roundID := *f.audit.CurrentRoundID
	item := func(name string) auditstore.MaterializedItem {
		return auditstore.MaterializedItem{
			ItemID: "item-" + name, ItemKey: "check-" + name, Ordinal: 0, Kind: "checklist",
			SubjectKey: "subject-" + name, Task: testExact("audits", name+"-task", "task-r1"),
			Origin: testOrigin("check-" + name), WorkflowRole: "check", InitialState: auditstore.ItemReady,
			Coverage: emptyCoverage(),
		}
	}
	next := item("storedcheck-next")
	next.ProposalSources = []auditstore.ProposalItemSource{{
		ReceiptID: "receipt-storedcheck-proposal", ProposedCheckOrdinal: 0,
		Proposal: testExact("audits", "storedcheck-proposal", "proposal-r1"),
	}}
	result := testExact("audit-storedcheck", "result", "result-r1")
	profile, snapshot := preparationPageProfile(t)
	preparing, _, err := f.store.CreateDraft(f.ctx, auditstore.CreateDraftParams{
		AuditID: "audit-storedcheck-preparing", OwnerID: f.audit.OwnerID, ProjectID: f.project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest},
		ProfileSnapshot: snapshot, InputSelection: json.RawMessage(`{}`), Limits: f.audit.Limits,
		IdempotencyKey: "storedcheck-preparing", RequestDigest: testDigest("9"),
	})
	if err != nil {
		t.Fatal(err)
	}
	settings := testExact("inputs", "settings", "settings-r1")
	settings.MediaType = "application/json"
	baseline, err := json.Marshal(map[string]any{
		"schema": "contractor.audit.baseline.v1",
		"inputs": map[string]auditstore.ExactArtifact{
			"source": testExact("inputs", "source", "source-r1"), "settings": settings,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
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
		{"MaterializeRound", "audits_baseline_snapshot_check", func(ctx context.Context) error {
			_, _, err := f.store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
				OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, ExpectedRevision: f.audit.Revision,
				RoundID: "round-storedcheck-start", RoundOrdinal: 1,
				Manifest:         testExact("audits", "storedcheck-start-worklist", "worklist-r1"),
				BaselineSnapshot: json.RawMessage(`{"inputs":[],"skills":[]}`),
				DeadlineAt:       time.Now().Add(time.Hour), Items: []auditstore.MaterializedItem{item("storedcheck-start")},
				IdempotencyKey: "storedcheck-start", RequestDigest: testDigest("2"),
			})
			return err
		}},
		{"StartPreparation", "audits_baseline_snapshot_check", func(ctx context.Context) error {
			_, _, err := f.store.StartPreparation(ctx, auditstore.StartPreparationParams{
				OwnerID: preparing.OwnerID, AuditID: preparing.AuditID, ExpectedRevision: preparing.Revision,
				BaselineSnapshot: baseline, DeadlineAt: time.Now().Add(time.Hour),
				IdempotencyKey: "storedcheck-prepare", RequestDigest: testDigest("3"),
			})
			return err
		}},
		{"CreateExecutionIntent", "audit_executions_preparation_shape", func(ctx context.Context) error {
			_, _, err := f.store.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{
				Claim: f.claim, ExecutionID: "execution-storedcheck-pending", RoundID: &roundID,
				Role: auditstore.ExecutionCheck, WorkflowRole: "check", LaneClass: resolved.LaneClassModel,
				Manifest:      testExact("audits", "storedcheck-pending-execution", "manifest-r1"),
				SubmissionKey: "submission-storedcheck-pending", RequestDigest: testDigest("4"),
				Members: []auditstore.ExecutionMemberIntent{{
					ExecutionItemID: "member-storedcheck-pending", ItemID: "pending-item-storedcheck",
					BatchOrdinal: 0, ItemAttempt: 1, Inputs: []auditstore.ExactArtifact{},
					Task: testExact("audits", "storedcheck-pending-task", "task-r1"),
				}},
			})
			return err
		}},
		{"Collect", "audit_executions_collection_retained_refs_check", func(ctx context.Context) error {
			_, _, err := f.store.Collect(ctx, storedCheckCollect(f, result))
			return err
		}},
		{"AcceptInitialRound", "audit_items_coverage_gaps_check", func(ctx context.Context) error {
			_, _, err := f.store.AcceptInitialRound(ctx, auditstore.AcceptInitialRoundParams{
				Claim: f.claim, ExpectedAuditRevision: f.audit.Revision, RoundID: "round-storedcheck-initial",
				Manifest: testExact("audits", "storedcheck-initial-worklist", "worklist-r1"),
				Items:    []auditstore.MaterializedItem{item("storedcheck-initial")},
				Inventory: auditstore.ArtifactLink{Scope: "project", LogicalKey: auditstore.InitialInventoryLogicalKey, SourceProvenance: json.RawMessage(`{}`),
					Artifact: testExact("audits", "storedcheck-inventory", "inventory-r1"),
				},
			})
			return err
		}},
		{"AcceptNextRound", "audit_items_coverage_gaps_check", func(ctx context.Context) error {
			_, _, err := f.store.AcceptNextRound(ctx, auditstore.AcceptRoundParams{
				Claim: f.claim, ExpectedAuditRevision: f.audit.Revision, PreviousRoundID: roundID,
				RoundID: "round-storedcheck-next", RoundOrdinal: 2,
				Manifest: testExact("audits", "storedcheck-next-worklist", "worklist-r1"),
				Items:    []auditstore.MaterializedItem{next},
			})
			return err
		}},
		{"SettleUndispatched", "audit_items_coverage_gaps_check", func(ctx context.Context) error {
			_, err := f.store.SettleUndispatched(ctx, f.claim, 1)
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
	lowerStoredBound(t, f.ctx, f.pool, "audit_items", "audit_items_coverage_gaps_check", "coverage_gaps", 2)
	gapped := auditstore.MaterializedItem{
		ItemID: "item-storedbound-start", ItemKey: "check-storedbound-start", Ordinal: 0, Kind: "checklist",
		SubjectKey: "subject-storedbound-start", Task: testExact("audits", "storedbound-start-task", "task-r1"),
		Origin: testOrigin("check-storedbound-start"), WorkflowRole: "check", InitialState: auditstore.ItemReady,
		Coverage: auditstore.Coverage{
			Status: auditstore.CoverageNotTested, Requested: []string{}, Completed: []string{},
			Gaps: []string{"callback-unsupported"},
		},
	}
	_, created, err := f.store.MaterializeRound(f.ctx, auditstore.MaterializeRoundParams{
		OwnerID: draft.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		RoundID: "round-storedbound-start", RoundOrdinal: 1,
		Manifest:         testExact("audits", "storedbound-start-worklist", "worklist-r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":[],"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), Items: []auditstore.MaterializedItem{gapped},
		IdempotencyKey: "storedbound-start", RequestDigest: testDigest("2"),
	})
	assertStoredConstraintInvalid(t, err, "audit_items_coverage_gaps_check")
	after, getErr := f.store.Get(f.ctx, draft.OwnerID, draft.AuditID)
	if created || getErr != nil || after.Revision != draft.Revision || after.State != draft.State || after.CurrentRoundID != nil {
		t.Fatalf("Audit after a refused start = (%+v, %t, %v), want the unchanged draft", after, created, getErr)
	}
}

func storedCheckCollect(f collectingAuditFixture, result auditstore.ExactArtifact) auditstore.CollectParams {
	return auditstore.CollectParams{
		Claim: f.claim, ReceiptID: "receipt-" + f.execution.ExecutionID, ExecutionID: f.execution.ExecutionID,
		Disposition: auditstore.CollectionAccepted, SourceOutput: &result, RequestDigest: testDigest("7"),
		Retained: []auditstore.ArtifactLink{{Scope: "project", LogicalKey: "result/" + f.memberID, Artifact: result,
			SourceProvenance: json.RawMessage(`{"runId":"run-storedcheck"}`),
		}},
		Items: []auditstore.CollectionItem{{
			ExecutionItemID: f.memberID, Disposition: auditstore.CollectionAccepted,
			FinalDisposition: auditstore.FinalAccepted, Result: &result,
			Coverage: auditstore.Coverage{Status: auditstore.CoverageSatisfied, Requested: []string{}, Completed: []string{}, Gaps: []string{}},
		}},
	}
}

// lowerStoredBound replaces constraint on table with a CHECK of the same name
// and measure, octet_length(column::text) <= limit. NOT VALID keeps the rows
// already stored, so only later writes meet the lowered bound.
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
