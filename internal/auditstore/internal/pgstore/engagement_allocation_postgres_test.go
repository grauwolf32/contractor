package pgstore

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresAllocationBaselineChecksImmutableOwnership(t *testing.T) {
	f := newCollectingAuditFixture(t, "allocation-baseline", 1024)
	store := NewQueryStore(f.pool, artifactpgtest.NewStorage())
	owner, project, execution, run := f.audit.OwnerID, f.audit.ProjectID, f.execution.ExecutionID, *f.execution.RunID
	id, baseline, err := store.ReadAllocationBaseline(f.ctx, owner, project, execution, run)
	if err != nil || id != f.audit.AuditID || len(baseline) == 0 {
		t.Fatalf("pinned baseline = %q %s, %v", id, baseline, err)
	}
	for _, args := range [][4]string{
		{"foreign", project, execution, run},
		{owner, "foreign", execution, run},
		{owner, project, "foreign", run},
		{owner, project, execution, "foreign"},
	} {
		if _, _, err := store.ReadAllocationBaseline(f.ctx, args[0], args[1], args[2], args[3]); !errors.Is(err, auditstore.ErrNotFound) {
			t.Fatalf("foreign ownership %q accepted: %v", args, err)
		}
	}
}

func TestPostgresAllocationBaselinePreservesPinsWhileOwnedRunsDrain(t *testing.T) {
	for _, engagement := range []bool{false, true} {
		for _, operation := range []string{"pause", "cancel", "audit-delete", "project-delete"} {
			t.Run(fmt.Sprintf("engagement-%t/%s", engagement, operation), func(t *testing.T) {
				f, next := allocationBaselineDrainFixture(t, engagement)
				read := func() (string, json.RawMessage, error) {
					return f.store.ReadAllocationBaseline(f.ctx, f.audit.OwnerID, f.audit.ProjectID, f.execution.ExecutionID, *f.execution.RunID)
				}
				id, before, err := read()
				if err != nil || id != f.audit.AuditID {
					t.Fatalf("active baseline = %q, %v", id, err)
				}
				switch operation {
				case "pause", "cancel":
					target := auditstore.AuditPaused
					if operation == "cancel" {
						target = auditstore.AuditCancelling
					}
					_, _, err = f.store.Transition(f.ctx, auditstore.TransitionParams{
						OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, ExpectedRevision: f.audit.Revision,
						ExpectedState: auditstore.AuditActive, TargetState: target,
						Reason:         &auditstore.StopReason{Code: "owner_" + operation, Message: "Allocation lifecycle test"},
						IdempotencyKey: operation, RequestDigest: testDigest("b"),
					})
				case "audit-delete":
					_, _, err = f.store.RequestDelete(f.ctx, auditstore.DeleteParams{
						OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, ExpectedRevision: f.audit.Revision,
						IdempotencyKey: operation, RequestDigest: testDigest("b"),
					})
				case "project-delete":
					_, _, err = projectstore.NewPostgresStore(f.pool).BeginDeletion(f.ctx, projectstore.BeginDeletionParams{
						OwnerID: f.project.OwnerID, ProjectID: f.project.ProjectID, ExpectedRevision: f.project.Revision,
					})
				}
				if err != nil {
					t.Fatal(err)
				}
				id, after, err := read()
				if err != nil || id != f.audit.AuditID || !bytes.Equal(before, after) {
					t.Fatalf("%s discarded or changed bound Run pins: %q, %v", operation, id, err)
				}
				if _, _, err := f.store.ReadAllocationBaseline(f.ctx, "foreign-owner", f.audit.ProjectID, f.execution.ExecutionID, *f.execution.RunID); !errors.Is(err, auditstore.ErrNotFound) {
					t.Fatalf("%s weakened owner fence: %v", operation, err)
				}
				// Reading immutable data does not reopen the existing dispatch gate.
				_, inserted, err := f.store.SelectTasks(f.ctx, next)
				if operation == "project-delete" {
					if inserted || !errors.Is(err, auditstore.ErrProjectDeleting) {
						t.Fatalf("new dispatch crossed Project deletion fence: inserted=%t, %v", inserted, err)
					}
				} else if inserted || (!errors.Is(err, auditstore.ErrPrecondition) && !errors.Is(err, auditstore.ErrProjectDeleting) && !errors.Is(err, auditstore.ErrClaimLost)) {
					t.Fatalf("new dispatch crossed %s fence: inserted=%t, %v", operation, inserted, err)
				}
			})
		}
	}
}

// Reuse the store fixture's isolated Project, but leave this Audit's own Run
// submitted and nonterminal so the lifecycle cases exercise draining work.
func allocationBaselineDrainFixture(t *testing.T, engagement bool) (collectingAuditFixture, auditstore.SelectTasksParams) {
	t.Helper()
	baseline := map[string]any{}
	if engagement {
		origin := "https://target.example"
		doc := engagementcontract.Document{
			Schema: engagementcontract.Schema, AccessMode: "blackbox", Target: engagementcontract.Target{Origins: []string{origin}},
			Scope:      engagementcontract.Scope{Routes: []engagementcontract.Route{{Origin: origin, PathPrefixes: []string{"/"}, Methods: []string{"GET"}, AddressRanges: []string{"192.0.2.0/24"}}}, Exclusions: []engagementcontract.Route{}},
			Identities: []engagementcontract.Identity{{ID: "anonymous", Kind: "anonymous", Origins: []string{origin}}}, Inputs: []engagementcontract.Input{}, Policy: engagementcontract.Policy{Version: "1", ActiveChecks: "automatic", RequiredCapabilities: []string{engagementcontract.Capability}, OptionalCapabilities: []string{}},
			Budgets: engagementcontract.Budgets{ActiveSeconds: 1, Rounds: 1, SubmittedRuns: 1, EvidenceBytes: 1},
		}
		if err := doc.Validate(); err != nil {
			t.Fatal(err)
		}
		digest, err := doc.Policy.Digest()
		if err != nil {
			t.Fatal(err)
		}
		baseline["engagement"] = map[string]any{
			"schema": "contractor.audit.engagement-baseline.v1", "artifact": testExact("inputs", "engagement", "r1"),
			"document": doc, "policyDigest": digest, "authorizedBy": "owner-baseline-drain",
			"authorizedAt": time.Date(2026, 10, 28, 12, 0, 0, 0, time.UTC),
		}
	}
	f := newCollectingAuditFixture(t, "baseline-drain", 1<<20, collectingAuditFixtureOptions{extraPending: true, nonterminal: true, baselineExtras: baseline})
	page, err := f.store.ListTasks(f.ctx, auditstore.ListTasksParams{OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, Limit: 3})
	if err != nil || len(page.Tasks) != 2 {
		t.Fatalf("pending Tasks: %+v %v", page, err)
	}
	return f, auditstore.SelectTasksParams{Claim: f.claim, TaskIDs: []string{page.Tasks[1].Document.TaskID}}
}
