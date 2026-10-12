package artifacts_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
)

// TestSharedPostgresAuditPreparationStartRecoversInsideACallerTransaction drives
// StartPreparation into the recovery behind its failed statement on a store
// backed by the caller's transaction: the recovery reads must run and return
// the method's own outcome, and the transaction must stay usable (V368-029).
func TestSharedPostgresAuditPreparationStartRecoversInsideACallerTransaction(t *testing.T) {
	f := newPreparationFixture(t, false)
	snapshot, err := json.Marshal(f.profile)
	if err != nil {
		t.Fatal(err)
	}
	draft, _, err := f.store.CreateDraft(f.ctx, auditstore.CreateDraftParams{
		AuditID: "preparation-audit-transaction", OwnerID: f.audit.OwnerID, ProjectID: f.audit.ProjectID,
		Profile: f.audit.Profile, ProfileSnapshot: snapshot, InputSelection: json.RawMessage(`{}`),
		Limits: f.audit.Limits, IdempotencyKey: "create-audit-transaction", RequestDigest: prepDigest("audit-transaction"),
	})
	if err != nil {
		t.Fatal(err)
	}
	// Every statement that writes audits fails with unique_violation in a
	// session that asks for it.
	if _, err := f.pool.Exec(f.ctx, `
CREATE FUNCTION contractor_test_inject_unique_violation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF current_setting('contractor_test.inject', true) = 'unique_violation' THEN
        RAISE EXCEPTION 'injected unique violation' USING ERRCODE = 'unique_violation';
    END IF;
    RETURN NULL;
END $$;
CREATE TRIGGER contractor_test_inject_unique_violation BEFORE INSERT OR UPDATE ON audits
FOR EACH STATEMENT EXECUTE FUNCTION contractor_test_inject_unique_violation()`); err != nil {
		t.Fatal(err)
	}
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) }()
	if _, err := tx.Exec(f.ctx, `SET LOCAL contractor_test.inject = 'unique_violation'`); err != nil {
		t.Fatal(err)
	}
	start := f.start
	start.AuditID, start.ExpectedRevision = draft.AuditID, draft.Revision
	start.IdempotencyKey, start.RequestDigest = "start-transaction", prepDigest("start-transaction")
	if _, _, err := auditstoretest.NewStore(tx).StartPreparation(f.ctx, start); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("StartPreparation after a failed statement in a transaction = %v, want precondition", err)
	}
	var usable int
	if err := tx.QueryRow(f.ctx, `SELECT 1`).Scan(&usable); err != nil || usable != 1 {
		t.Fatalf("StartPreparation left the caller's transaction unusable: %v", err)
	}
}
