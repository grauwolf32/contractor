package public

import (
	"context"
	"encoding/json"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// attemptsWithoutEmptyEntries stands for a store that leaves out the items
// without attempts, which ListItemAttempts promises not to do, so the item
// list is shown to keep its own empty list rather than render null.
type attemptsWithoutEmptyEntries struct{ *fakeAuditManagement }

func (a attemptsWithoutEmptyEntries) ListItemAttempts(
	ctx context.Context, ownerID, auditID string, itemIDs []string,
) (map[string][]auditstore.ItemAttempt, error) {
	attempts, err := a.fakeAuditManagement.ListItemAttempts(ctx, ownerID, auditID, itemIDs)
	for itemID, itemAttempts := range attempts {
		if len(itemAttempts) == 0 {
			delete(attempts, itemID)
		}
	}
	return attempts, err
}

func auditItemAttemptsFixture() (*fakeAuditManagement, auditstore.Item, auditstore.Item, auditstore.ItemAttempt) {
	now := time.Date(2026, 10, 30, 12, 0, 0, 0, time.UTC)
	revision := "revision-1"
	task := auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "inputs", Name: "checklist", Revision: &revision},
		Digest: auditHandlerDigest("checklist"), MediaType: "application/json", SizeBytes: 64,
	}
	item := func(id string, ordinal int) auditstore.Item {
		return auditstore.Item{
			ItemID: id, AuditID: "audit-one", RoundID: "round-one", ItemKey: "check-" + id, Ordinal: ordinal,
			Kind: "checklist", SubjectKey: "check-" + id + "@1", Task: task,
			Origin: auditstore.ItemOrigin{
				Schema: auditstore.ItemOriginSchema, SourceRef: &task.Ref, SourceContentDigest: task.Digest,
				SourceMediaType: task.MediaType, CanonicalInventoryDigest: auditHandlerDigest("inventory"),
				EntryKey: "check-" + id,
			},
			WorkflowRole: "check", State: auditstore.ItemReady,
			CreatedAt: now, UpdatedAt: now,
		}
	}
	attempted, waiting := item("item-attempted", 0), item("item-waiting", 1)
	attempted.State = auditstore.ItemSubmitted
	runID := "run-one"
	attempt := auditstore.ItemAttempt{
		ExecutionItemID: "execution-item-one", ExecutionID: "execution-one", ItemID: attempted.ItemID,
		ItemAttempt: 1, Role: auditstore.ExecutionCheck, State: auditstore.ItemSubmitted,
		RunID: &runID, CreatedAt: now,
	}
	double := &fakeAuditManagement{
		started:  auditservice.StartedAudit{Items: []auditstore.Item{attempted, waiting}},
		attempts: map[string][]auditstore.ItemAttempt{attempted.ItemID: {attempt}},
	}
	return double, attempted, waiting, attempt
}

// TestAuditItemPageRendersEmptyAttemptsForAnItemWithout serves an item page
// whose second item has no attempts and validates the answer against the
// public contract: the item renders attempts as an empty array, both when the
// attempt read answers it with an empty list, as the store does, and when the
// read leaves it out.
func TestAuditItemPageRendersEmptyAttemptsForAnItemWithout(t *testing.T) {
	double, attempted, waiting, _ := auditItemAttemptsFixture()
	router := publicContractRouter(t)
	for name, audits := range map[string]AuditManagement{
		"store-like read":       double,
		"read without an entry": attemptsWithoutEmptyEntries{double},
	} {
		t.Run(name, func(t *testing.T) {
			fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
				func(dependencies *testServices) { dependencies.Audits = audits })
			response := contractCall(t, router, fixture.handler,
				jsonContractRequest(http.MethodGet, "/v1/audits/audit-one/items", ""), http.StatusOK)
			var page struct {
				Items []struct {
					ItemID   string            `json:"itemId"`
					Attempts []json.RawMessage `json:"attempts"`
				} `json:"items"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil {
				t.Fatal(err)
			}
			if len(page.Items) != 2 || page.Items[0].ItemID != attempted.ItemID || len(page.Items[0].Attempts) != 1 ||
				page.Items[1].ItemID != waiting.ItemID || page.Items[1].Attempts == nil || len(page.Items[1].Attempts) != 0 {
				t.Fatalf("item page = %s", response.Body.String())
			}
		})
	}
}

// TestAuditItemAttemptDoubleAnswersLikeTheStore pins the double to the
// store's answer (pgstore TestListItemAttemptsAnswersEveryRequestedItem): one
// entry per requested item, an empty list for an item without attempts.
func TestAuditItemAttemptDoubleAnswersLikeTheStore(t *testing.T) {
	double, attempted, waiting, attempt := auditItemAttemptsFixture()
	attempts, err := double.ListItemAttempts(t.Context(), "user-1", "audit-one", []string{attempted.ItemID, waiting.ItemID})
	want := map[string][]auditstore.ItemAttempt{attempted.ItemID: {attempt}, waiting.ItemID: {}}
	if err != nil || !reflect.DeepEqual(attempts, want) {
		t.Fatalf("double attempts = (%#v, %v), want %#v", attempts, err, want)
	}
}
