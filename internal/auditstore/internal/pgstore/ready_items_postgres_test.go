package pgstore

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// The reconcile snapshot reads the first MaxReconcileRows ready items of each
// Workflow role, so a role whose items fill more than the read bound ahead of
// another role's cannot hide them; scan and check roles here stand for the two
// lane classes (AUX-R3). Within a role the lowest ordinals come first
// (AUX-R13), and items not ready are never read.
func TestPostgresReconcileSnapshotReadsReadyItemsOfEveryRole(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-ready-roles", OwnerID: "owner-ready-roles", Kind: projectstore.KindProject,
		Name: "Ready roles", IdempotencyKey: "project-ready-roles", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	// A full Round (LIM-AUD-26): the scans alone are far beyond the read bound.
	const scans, checks = 9_990, 9
	store := newTestStore(pool)
	audit, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-ready-roles", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"profile"}`),
		InputSelection:  json.RawMessage(`{"inputs":{}}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: scans + checks + 1, MaxItemsTotal: scans + checks + 1,
			MaxSubmittedRuns: scans + checks + 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
		},
		IdempotencyKey: "audit-ready-roles", RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	// Every scan item precedes every check item, except one check item that
	// awaits review ahead of all of them.
	items := make([]auditstore.MaterializedItem, 0, scans+checks+1)
	add := func(role string, state auditstore.ItemState) {
		key := fmt.Sprintf("%s-%05d", role, len(items))
		item := auditstore.MaterializedItem{
			ItemID: "item-" + key, ItemKey: key, Ordinal: len(items), Kind: "checklist", SubjectKey: key,
			Task: testExact("audits", "task-"+key, "r1"), Origin: testOrigin(key), WorkflowRole: role,
			InitialState: state, Coverage: emptyCoverage(),
		}
		if state == auditstore.ItemAwaitingReview {
			item.ApprovalKind = auditstore.ItemApprovalActiveCheck
			item.ApprovalDigest = testDigest("5")
		}
		items = append(items, item)
	}
	add("check", auditstore.ItemAwaitingReview)
	for range scans {
		add("scan", auditstore.ItemReady)
	}
	for range checks {
		add("check", auditstore.ItemReady)
	}
	audit, _, err = store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: project.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: "round-ready-roles", RoundOrdinal: 1, Manifest: testExact("audits", "round-ready-roles", "r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":{},"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), Items: items,
		IdempotencyKey: "start-ready-roles", RequestDigest: testDigest("4"),
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "controller-ready-roles", Lease: 20 * time.Second, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	snapshot, err := store.GetReconcileSnapshot(ctx, claims[0])
	if err != nil {
		t.Fatal(err)
	}
	ready := map[string][]int{}
	previous := -1
	for _, item := range snapshot.ReadyItems {
		if item.State != auditstore.ItemReady || item.Ordinal <= previous {
			t.Fatalf("ready item %s (state %s, ordinal %d) after ordinal %d", item.ItemID, item.State, item.Ordinal, previous)
		}
		previous = item.Ordinal
		ready[item.WorkflowRole] = append(ready[item.WorkflowRole], item.Ordinal)
	}
	if got := ready["scan"]; len(got) != auditstore.MaxReconcileRows || got[0] != 1 || got[len(got)-1] != auditstore.MaxReconcileRows {
		t.Fatalf("scan items read = %d from %v, want the first %d", len(got), got[:min(len(got), 3)], auditstore.MaxReconcileRows)
	}
	if got := ready["check"]; len(got) != checks || got[0] != scans+1 {
		t.Fatalf("check items read = %v, want the %d ready check items behind the scans", got, checks)
	}

	// At this size the first role, each next role and each role's items are
	// index probes, not a scan of the Round; an executed plan shows all three.
	if _, err := pool.Exec(ctx, `ANALYZE audit_items`); err != nil {
		t.Fatal(err)
	}
	rows, err := pool.Query(ctx, "EXPLAIN (ANALYZE, COSTS OFF) "+listReadyItemsSQL, audit.AuditID, "round-ready-roles", auditstore.MaxReconcileRows)
	if err != nil {
		t.Fatal(err)
	}
	var plan strings.Builder
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err != nil {
			t.Fatal(err)
		}
		plan.WriteString(line + "\n")
	}
	if rows.Close(); rows.Err() != nil {
		t.Fatal(rows.Err())
	}
	if strings.Count(plan.String(), "using audit_items_ready_role_idx") != 3 || strings.Contains(plan.String(), "Seq Scan") ||
		strings.Contains(plan.String(), "Bitmap") {
		t.Fatalf("ready read does not probe audit_items_ready_role_idx:\n%s", plan.String())
	}
}
