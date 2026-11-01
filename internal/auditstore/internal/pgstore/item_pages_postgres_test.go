package pgstore

import (
	"errors"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
)

// itemPageCase is one filter and cursor of the public item page.
type itemPageCase struct {
	name    string
	round   *string
	state   *auditstore.ItemState
	subject *string
	// after is the reference index of the cursor item, or -1 for the first
	// page.
	after int
	// budget is the most audit_items rows one page may read in either plan.
	budget int64
}

// TestPostgresItemPagesBoundCandidatesPerRound pages an Audit at the item
// limits (32 rounds, 100,000 items) and requires each public item page to
// read a bounded candidate set in both the custom and the generic plan.
// Unfiltered pages, from the first to the last, read at most 32 times the
// fetch limit, where the single sorted statement read every item of the
// Audit once a cursor moved past the first round or the plan went generic.
// The measured filtered cases keep explicit budgets: a round reads one
// fetch, an unsettled state reads its own items through the dispatch index,
// settled additionally reads the current round it rejects, and a subject
// filter, which no index serves, reads every item of the Audit. Every page
// equals the matching items of the full ordered read after the cursor, so
// filters apply before the limit and the round, item ordinal, item ID order
// and owner fence hold.
func TestPostgresItemPagesBoundCandidatesPerRound(t *testing.T) {
	f := newItemScaleFixture(t)
	reference, err := f.store.ListItems(f.ctx, f.auditID)
	if err != nil || len(reference) != itemScaleItems {
		t.Fatalf("reference read = %d items, %v", len(reference), err)
	}
	roundOrdinals := map[string]int{}
	rounds, err := f.store.ListRounds(f.ctx, f.auditID)
	if err != nil {
		t.Fatal(err)
	}
	for _, round := range rounds {
		roundOrdinals[round.RoundID] = round.Ordinal
	}
	fetch := pagelimit.FetchRows(pagelimit.MaxRows)
	unfiltered := int64(itemScaleRounds * fetch)
	at := func(round, ordinal int) int { return (round-1)*itemScaleRoundItems + ordinal }
	text := func(value string) *string { return &value }
	state := func(value auditstore.ItemState) *auditstore.ItemState { return &value }
	round16 := f.roundIDs[15]
	cases := []itemPageCase{
		{name: "first page", after: -1, budget: unfiltered},
		{name: "first round cursor", after: at(1, 1_500), budget: unfiltered},
		{name: "middle round cursor", after: at(16, 1_500), budget: unfiltered},
		{name: "late cursor", after: at(32, 3_100), budget: unfiltered},
		{name: "last item cursor", after: itemScaleItems - 1, budget: unfiltered},
		{name: "round", round: &round16, after: -1, budget: 2 * int64(fetch)},
		{name: "round late cursor", round: &round16, after: at(16, 3_000), budget: 2 * int64(fetch)},
		{name: "awaiting review", state: state(auditstore.ItemAwaitingReview), after: -1, budget: int64(fetch)},
		{name: "awaiting review cursor", state: state(auditstore.ItemAwaitingReview), after: at(1, 1_499), budget: int64(fetch)},
		{name: "ready", state: state(auditstore.ItemReady), after: -1, budget: unfiltered},
		{name: "ready late cursor", state: state(auditstore.ItemReady), after: at(1, 3_000), budget: unfiltered},
		{name: "settled", state: state(auditstore.ItemSettled), after: -1, budget: itemScaleRoundItems + unfiltered},
		{name: "settled cursor", state: state(auditstore.ItemSettled), after: at(20, 10), budget: unfiltered},
		{name: "subject", subject: text("subject-01000"), after: -1, budget: itemScaleItems + unfiltered},
		{name: "subject cursor", subject: text("subject-01000"), after: at(20, 1_000), budget: itemScaleItems + unfiltered},
	}
	conn, err := f.pool.Acquire(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Release()
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			params := auditstore.ListItemsParams{
				OwnerID: f.owner, AuditID: f.auditID, RoundID: test.round, State: test.state,
				SubjectKey: test.subject, Limit: fetch,
			}
			start := 0
			if test.after >= 0 {
				cursor := reference[test.after]
				ordinal := roundOrdinals[cursor.RoundID]
				params.AfterRoundOrdinal, params.AfterItemOrdinal, params.AfterItemID = &ordinal, &cursor.Ordinal, cursor.ItemID
				start = test.after + 1
			}
			var want []string
			for _, item := range reference[start:] {
				if len(want) == fetch {
					break
				}
				if (test.round == nil || item.RoundID == *test.round) && (test.state == nil || item.State == *test.state) &&
					(test.subject == nil || item.SubjectKey == *test.subject) {
					want = append(want, item.ItemID)
				}
			}
			for _, mode := range explaintest.Modes {
				if _, err := conn.Exec(f.ctx, "SET plan_cache_mode = "+string(mode)); err != nil {
					t.Fatal(err)
				}
				page, err := NewQueryStore(conn).ListItemsPage(f.ctx, params)
				if err != nil {
					t.Fatal(err)
				}
				got := make([]string, len(page))
				for index, item := range page {
					got[index] = item.ItemID
				}
				if !slices.Equal(got, want) {
					t.Fatalf("%s: page holds %d items %v, want %d %v", mode, len(got), got[:min(3, len(got))], len(want), want[:min(3, len(want))])
				}
				statement, args := itemPageQuery(params)
				plan := explainPrepared(t, f.ctx, f.pool, mode, statement, args...)
				if read := plan.Reads.Of("audit_items"); read > test.budget {
					t.Fatalf("%s: page read %d items, budget %d\n%s", mode, read, test.budget, plan.Plan)
				}
				t.Logf("%s: page of %d read %d items (budget %d)", mode, len(got), plan.Reads.Of("audit_items"), test.budget)
			}
			if _, err := conn.Exec(f.ctx, "RESET plan_cache_mode"); err != nil {
				t.Fatal(err)
			}
		})
	}

	// The owner fence: another owner's Audit and an Audit of the owner read
	// as not found and empty, whatever the filter.
	for _, params := range []auditstore.ListItemsParams{
		{OwnerID: f.owner, AuditID: f.foreign, Limit: fetch},
		{OwnerID: "owner-item-scale-foreign", AuditID: f.auditID, Limit: fetch},
		{OwnerID: "owner-item-scale-foreign", AuditID: f.auditID, State: state(auditstore.ItemReady), Limit: fetch},
	} {
		if page, err := f.store.ListItemsPage(f.ctx, params); !errors.Is(err, auditstore.ErrNotFound) || len(page) != 0 {
			t.Fatalf("foreign item page %+v = %d items, %v", params, len(page), err)
		}
	}
	walkSmallAuditItems(t, f)
}

// walkSmallAuditItems walks the small Audit's items two at a time under
// every filter combination the public API accepts and requires the pages to
// hold exactly the matching items of the full ordered read, in order, across
// round boundaries.
func walkSmallAuditItems(t *testing.T, f itemScaleFixture) {
	t.Helper()
	reference, err := f.store.ListItems(f.ctx, f.smallID)
	if err != nil || len(reference) != 21 {
		t.Fatalf("reference read = %d items, %v", len(reference), err)
	}
	rounds, err := f.store.ListRounds(f.ctx, f.smallID)
	if err != nil {
		t.Fatal(err)
	}
	ordinals := map[string]int{}
	for _, round := range rounds {
		ordinals[round.RoundID] = round.Ordinal
	}
	roundIDs := []*string{nil, &rounds[0].RoundID, &rounds[2].RoundID}
	states := []*auditstore.ItemState{nil}
	for _, value := range []auditstore.ItemState{auditstore.ItemReady, auditstore.ItemAwaitingReview, auditstore.ItemSettled, auditstore.ItemPending} {
		states = append(states, &value)
	}
	subject := "subject-00002"
	for _, roundID := range roundIDs {
		for _, state := range states {
			for _, subjectKey := range []*string{nil, &subject} {
				params := auditstore.ListItemsParams{
					OwnerID: f.owner, AuditID: f.smallID, RoundID: roundID, State: state, SubjectKey: subjectKey, Limit: 2,
				}
				var want, got []string
				for _, item := range reference {
					if (roundID == nil || item.RoundID == *roundID) && (state == nil || item.State == *state) &&
						(subjectKey == nil || item.SubjectKey == *subjectKey) {
						want = append(want, item.ItemID)
					}
				}
				for {
					page, err := f.store.ListItemsPage(f.ctx, params)
					if err != nil {
						t.Fatal(err)
					}
					for _, item := range page {
						got = append(got, item.ItemID)
					}
					if len(page) < params.Limit {
						break
					}
					last := page[len(page)-1]
					ordinal := ordinals[last.RoundID]
					params.AfterRoundOrdinal, params.AfterItemOrdinal, params.AfterItemID = &ordinal, &last.Ordinal, last.ItemID
				}
				if !slices.Equal(got, want) {
					t.Fatalf("round %v state %v subject %v: walked %v, want %v", deref(roundID), deref(state), deref(subjectKey), got, want)
				}
			}
		}
	}
}

func deref[T any](value *T) any {
	if value == nil {
		return nil
	}
	return *value
}
