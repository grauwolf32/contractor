package evalservice

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

// Every visible change during a lifecycle publishes a new generation, but no
// reader can reach a replaced one: retained view rows stay O(members).
func TestPostgresNativeLifecycleRetainsOnlyCurrentViewGeneration(t *testing.T) {
	h := newHarness(t)
	e := h.prepared(t, "workflow")
	h.command(t, e, "start")
	c := h.coordinator(t, "single")
	var view *evalstore.View
	for n := 0; n < 80; n++ {
		tick(t, c)
		h.finishRuns(t)
		e = h.get(t, e.ID)
		var err error
		if view, err = evalstore.NewPostgresStore(h.pool).LatestView(t.Context(), e.OwnerID, e.ID); err != nil {
			t.Fatal(err)
		}
		if e.State == evaldomain.StateFinished && view != nil && view.Freshness == "current" {
			break
		}
	}
	if e.State != evaldomain.StateFinished || view == nil || view.Freshness != "current" {
		t.Fatalf("lifecycle did not settle: state=%s view=%+v", e.State, view)
	}
	var generations, members, pairs int
	if err := h.pool.QueryRow(t.Context(), `
SELECT (SELECT count(*) FROM eval_projection_queue WHERE experiment_id = $1 AND generation IS NOT NULL),
    (SELECT 2 * count(*) FROM eval_view_pairs WHERE experiment_id = $1),
    (SELECT count(*) FROM eval_view_pairs WHERE experiment_id = $1)
`, e.ID).Scan(&generations, &members, &pairs); err != nil {
		t.Fatal(err)
	}
	if view.Generation < 3 {
		t.Fatalf("lifecycle published only %d generations", view.Generation)
	}
	if generations != 1 || members != e.Expected || pairs != e.Expected/2 {
		t.Fatalf("after %d publications retained %d generations, %d member and %d pair rows", view.Generation, generations, members, pairs)
	}
}
