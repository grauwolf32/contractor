package evalservice

import (
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evalstore"
)

// liveMember returns an accepted member of a started workflow experiment whose
// Run is still active, with a controller claim on the experiment.
func liveMember(t *testing.T) (*serviceHarness, evalstore.Experiment, evalstore.Claim, string) {
	t.Helper()
	h := newHarness(t)
	e := h.create(t, "workflow")
	h.command(t, e, "prepare")
	tick(t, h.coordinator(t, "prepare"))
	h.command(t, h.get(t, e.ID), "start")
	c := h.coordinator(t, "controller")
	tick(t, c)
	tick(t, c)
	tick(t, c)

	store := evalstore.NewPostgresStore(h.pool)
	claims, err := store.Claim(t.Context(), "observe", time.Minute, 1)
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim: %v %v", claims, err)
	}
	members, err := store.Outstanding(t.Context(), e.OwnerID, e.ID, 100)
	if err != nil || len(members) == 0 {
		t.Fatalf("outstanding: %v %v", members, err)
	}
	return h, e, claims[0], members[0]
}

func TestPostgresEvalUsageTicksKeepListCursors(t *testing.T) {
	h, e, claim, member := liveMember(t)
	store := evalstore.NewPostgresStore(h.pool)
	page, err := store.List(t.Context(), evalstore.ListParams{OwnerID: h.scope.OwnerID, ProjectID: h.scope.ProjectID, Limit: 1})
	if err != nil {
		t.Fatal(err)
	}
	owned, err := store.List(t.Context(), evalstore.ListParams{OwnerID: h.scope.OwnerID, Limit: 1})
	if err != nil {
		t.Fatal(err)
	}
	before := h.get(t, e.ID).Revision
	for _, observed := range []int64{10, 20} {
		if err = h.service.tx(t.Context(), func(s *evalstore.Store) error {
			return s.ObserveTokens(t.Context(), h.scope, e.ID, member, claim, observed)
		}); err != nil {
			t.Fatal(err)
		}
	}
	if h.get(t, e.ID).Revision == before {
		t.Fatal("fixture did not advance the experiment")
	}
	if _, err = store.List(t.Context(), evalstore.ListParams{OwnerID: h.scope.OwnerID, ProjectID: h.scope.ProjectID, Limit: 1, Revision: &page.Revision}); err != nil {
		t.Fatal("usage tick invalidated list cursor", err)
	}
	if _, err = store.List(t.Context(), evalstore.ListParams{OwnerID: h.scope.OwnerID, Limit: 1, Revision: &owned.Revision}); err != nil {
		t.Fatal("usage tick invalidated owner list cursor", err)
	}
}
