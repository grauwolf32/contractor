package evalservice

import (
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/runstore"
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
	page, err := store.SummaryPage(t.Context(), evalstore.SummaryPageParams{ListParams: evalstore.ListParams{
		OwnerID: h.scope.OwnerID, ProjectID: h.scope.ProjectID, Limit: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	owned, err := store.SummaryPage(t.Context(), evalstore.SummaryPageParams{ListParams: evalstore.ListParams{
		OwnerID: h.scope.OwnerID, Limit: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	before := h.get(t, e.ID)
	for _, observed := range []int64{10, 20} {
		if err = h.service.tx(t.Context(), func(s *evalstore.Store) error {
			return s.ObserveTokens(t.Context(), h.scope, e.ID, member, claim, observed)
		}); err != nil {
			t.Fatal(err)
		}
	}
	after := h.get(t, e.ID)
	if after.Revision != before.Revision || after.ObservedTokens != 20 || !after.UpdatedAt.After(before.UpdatedAt) {
		t.Fatalf("usage tick changed authority CAS or lost progress: before=%+v after=%+v", before, after)
	}
	if _, err = store.SummaryPage(t.Context(), evalstore.SummaryPageParams{ListParams: evalstore.ListParams{
		OwnerID: h.scope.OwnerID, ProjectID: h.scope.ProjectID, Limit: 1, Revision: &page.Revision,
	}}); err != nil {
		t.Fatal("usage tick invalidated list cursor", err)
	}
	if _, err = store.SummaryPage(t.Context(), evalstore.SummaryPageParams{ListParams: evalstore.ListParams{
		OwnerID: h.scope.OwnerID, Limit: 1, Revision: &owned.Revision,
	}}); err != nil {
		t.Fatal("usage tick invalidated owner list cursor", err)
	}
}

func TestPostgresEvalIdleCoordinatorTicksKeepSelectedView(t *testing.T) {
	h := newHarness(t)
	e := h.create(t, "workflow")
	var draft evaldomain.Draft
	if err := json.Unmarshal(e.Draft.Bytes(), &draft); err != nil {
		t.Fatal(err)
	}
	draft.Budgets.MaxInFlight = 2
	update := frozen(t, "DraftUpdate", evaldomain.DraftUpdate{Name: e.Name, Draft: draft})
	if err := h.service.tx(t.Context(), func(s *evalstore.Store) error {
		_, err := s.UpdateDraft(t.Context(), h.scope, e.ID, update, identity(t, "in-flight", e.Revision, update))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	h.command(t, h.get(t, e.ID), "prepare")
	c := h.coordinator(t, "idle-controller")
	tick(t, c)
	h.command(t, h.get(t, e.ID), "start")
	var accepted int
	for attempt := 0; attempt < 20; attempt++ {
		tick(t, c)
		if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_submissions WHERE experiment_id=$1 AND state='accepted'`, e.ID).Scan(&accepted); err != nil {
			t.Fatal(err)
		}
		if accepted == 2 {
			break
		}
	}
	if accepted != 2 {
		t.Fatalf("fixture has %d accepted Runs, want 2", accepted)
	}

	store := evalstore.NewPostgresStore(h.pool)
	outstanding, err := store.Outstanding(t.Context(), e.OwnerID, e.ID, 2)
	if err != nil || len(outstanding) != 2 {
		t.Fatalf("outstanding: %v %v", outstanding, err)
	}
	claims, err := store.Claim(t.Context(), "usage-observer", time.Minute, 1)
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim: %v %v", claims, err)
	}
	if err := h.service.tx(t.Context(), func(s *evalstore.Store) error {
		return s.ObserveTokens(t.Context(), h.scope, e.ID, outstanding[0], claims[0], 17)
	}); err != nil {
		t.Fatal(err)
	}
	rotated, err := store.Outstanding(t.Context(), e.OwnerID, e.ID, 2)
	if err != nil || len(rotated) != 2 || rotated[0] != outstanding[1] {
		t.Fatalf("usage poll did not rotate Outstanding: before=%v after=%v err=%v", outstanding, rotated, err)
	}
	if got := h.get(t, e.ID).ObservedTokens; got != 17 {
		t.Fatalf("observed tokens = %d, want 17", got)
	}
	if err := store.ReleaseClaim(t.Context(), claims[0]); err != nil {
		t.Fatal(err)
	}

	type viewState struct {
		generations, members, pairs, charts, progress int64
		revision, published                           int64
		snapshot                                      string
	}
	state := func() viewState {
		t.Helper()
		var v viewState
		err := h.pool.QueryRow(t.Context(), `
SELECT (SELECT count(*) FROM eval_view_generations WHERE experiment_id=$1),
       (SELECT count(*) FROM eval_view_members WHERE experiment_id=$1),
       (SELECT count(*) FROM eval_view_pairs WHERE experiment_id=$1),
       (SELECT count(*) FROM eval_view_charts WHERE experiment_id=$1),
       (SELECT count(*) FROM eval_progress_observations WHERE experiment_id=$1),
       revision, published_revision, snapshot_id
FROM eval_projection_queue WHERE experiment_id=$1`, e.ID).Scan(
			&v.generations, &v.members, &v.pairs, &v.charts, &v.progress,
			&v.revision, &v.published, &v.snapshot,
		)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	before := state()
	if before.generations == 0 || before.revision != before.published {
		t.Fatalf("fixture view is not current: %+v", before)
	}
	var polledAt time.Time
	if err := h.pool.QueryRow(t.Context(), `SELECT updated_at FROM eval_submissions WHERE experiment_id=$1 AND member_id=$2`, e.ID, outstanding[0]).Scan(&polledAt); err != nil {
		t.Fatal(err)
	}
	for range 10 {
		tick(t, c)
	}
	after := state()
	if after != before {
		t.Fatalf("idle ticks republished selected view: before=%+v after=%+v", before, after)
	}
	var lastPolledAt time.Time
	if err := h.pool.QueryRow(t.Context(), `SELECT updated_at FROM eval_submissions WHERE experiment_id=$1 AND member_id=$2`, e.ID, outstanding[0]).Scan(&lastPolledAt); err != nil {
		t.Fatal(err)
	}
	if !lastPolledAt.After(polledAt) || h.get(t, e.ID).ObservedTokens != 17 {
		t.Fatal("idle coordinator stopped polling or lost observed-token high-water mark")
	}

	var runID, runState string
	if err := h.pool.QueryRow(t.Context(), `SELECT r.run_id,r.state FROM workflow_runs r JOIN eval_submissions s ON s.execution_id=r.run_id WHERE s.experiment_id=$1 AND s.member_id=$2`, e.ID, outstanding[0]).Scan(&runID, &runState); err != nil {
		t.Fatal(err)
	}
	if runState != string(runstore.RunPending) {
		t.Fatalf("fixture Run state = %s, want pending", runState)
	}
	if _, err := runstore.NewPostgresStore(h.pool).TransitionRun(t.Context(), runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "fixture_admitted"}); err != nil {
		t.Fatal(err)
	}
	tick(t, c)
	changed := state()
	// Publication prunes the replaced generation, so only the new one remains.
	if changed.generations != 1 || changed.snapshot == after.snapshot || changed.revision != changed.published {
		t.Fatalf("Run state change did not publish a current view: before=%+v after=%+v", after, changed)
	}
}

func TestPostgresEvalSettleSurfacesTombstoneReadFailures(t *testing.T) {
	h, e, claim, member := liveMember(t)
	// A live execution without a tombstone drains; a failed tombstone read
	// must abort instead of being committed as a drain.
	if err := h.service.tx(t.Context(), func(s *evalstore.Store) error {
		return s.Settle(t.Context(), h.scope, e.ID, member, claim)
	}); !errors.Is(err, evalstore.ErrDrain) {
		t.Fatal("live execution did not drain", err)
	}
	if _, err := h.pool.Exec(t.Context(), `ALTER TABLE eval_execution_tombstones RENAME TO eval_execution_tombstones_unavailable`); err != nil {
		t.Fatal(err)
	}
	err := h.service.tx(t.Context(), func(s *evalstore.Store) error {
		return s.Settle(t.Context(), h.scope, e.ID, member, claim)
	})
	if err == nil || errors.Is(err, evalstore.ErrDrain) {
		t.Fatal("tombstone read failure was treated as drain", err)
	}
}
