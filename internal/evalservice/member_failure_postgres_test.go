package evalservice

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evalcoordinator"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/runservice"
)

var errMemberCreate = errors.New("fixture: unclassified Run creation failure")

// failingMemberRuns fails every Run creation of one member with an unclassified
// error, so that member keeps a replayable creation intent on every tick.
type failingMemberRuns struct {
	RunCreator
	key string
}

func (f failingMemberRuns) CreatePublic(ctx context.Context, p runservice.PublicCreateParams) (runservice.CreateResult, error) {
	if p.IdempotencyKey == f.key {
		return runservice.CreateResult{}, errMemberCreate
	}
	return f.RunCreator.CreatePublic(ctx, p)
}

// startedWithFailingMember starts eight Workflow members with capacity two;
// Run creation of the first member in frozen order always fails.
func startedWithFailingMember(t *testing.T) (*serviceHarness, evalstore.Experiment, string) {
	t.Helper()
	h := newHarness(t)
	e := h.preparedWithCapacity(t, "workflow", 2)
	var member, key string
	if err := h.pool.QueryRow(t.Context(), `
SELECT member_id, submission_key FROM eval_members WHERE experiment_id = $1 ORDER BY ordinal LIMIT 1`, e.ID).Scan(&member, &key); err != nil {
		t.Fatal(err)
	}
	h.driver.Runs = failingMemberRuns{RunCreator: h.run.RunCreator, key: key + "-run-create"}
	h.command(t, e, "start")
	return h, e, member
}

// tickWithMemberFailure tolerates only the injected member failure and reports
// whether the coordinator received it.
func tickWithMemberFailure(t *testing.T, c *evalcoordinator.Coordinator) bool {
	t.Helper()
	_, err := c.RunOnce(t.Context())
	if err != nil && !errors.Is(err, errMemberCreate) {
		t.Fatal(err)
	}
	return err != nil
}

func terminalSubmissions(t *testing.T, h *serviceHarness, id string) int {
	t.Helper()
	var n int
	if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_submissions WHERE experiment_id = $1 AND state = 'terminal'`, id).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// publishedTerminal returns the terminal members of the current published view.
func publishedTerminal(t *testing.T, h *serviceHarness, e evalstore.Experiment) int {
	t.Helper()
	view, err := evalstore.NewPostgresStore(h.pool).LatestView(t.Context(), e.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	if view == nil || view.Freshness != "current" {
		t.Fatalf("published view is not current: %+v", view)
	}
	n := 0
	for _, counts := range view.Summary.Counts {
		n += counts.Terminal
	}
	return n
}

func TestPostgresFailingMemberDoesNotBlockAdmissionOrPublication(t *testing.T) {
	h, e, failing := startedWithFailingMember(t)
	c := h.coordinator(t, "controller")
	for n := 0; n < 40 && terminalSubmissions(t, h, e.ID) < 7; n++ {
		tickWithMemberFailure(t, c)
		h.finishRuns(t)
		if e = h.get(t, e.ID); e.Outstanding > e.MaxInFlight {
			t.Fatalf("tick %d has %d outstanding members over capacity %d", n, e.Outstanding, e.MaxInFlight)
		}
	}
	if admitted, terminal := count(t, h.pool, "eval_submissions"), terminalSubmissions(t, h, e.ID); admitted != 8 || terminal != 7 {
		t.Fatalf("failing member blocked the rest: admitted=%d terminal=%d", admitted, terminal)
	}
	if !tickWithMemberFailure(t, c) {
		t.Fatal("persistent member failure was not reported to the coordinator")
	}
	e = h.get(t, e.ID)
	if e.State != evaldomain.StateSettling || e.Outstanding != 1 {
		t.Fatalf("lifecycle did not progress around the failing member: state=%s outstanding=%d", e.State, e.Outstanding)
	}
	var state string
	if err := h.pool.QueryRow(t.Context(), `SELECT state FROM eval_submissions WHERE experiment_id = $1 AND member_id = $2`, e.ID, failing).Scan(&state); err != nil || state != "intent" {
		t.Fatalf("failing member submission = %q, %v; want a retained intent", state, err)
	}
	if got := publishedTerminal(t, h, e); got != 7 {
		t.Fatalf("published view has %d terminal members, want 7", got)
	}
}

func TestPostgresFailingMemberStillDrainsAtWallDeadline(t *testing.T) {
	h, e, _ := startedWithFailingMember(t)
	c := h.coordinator(t, "controller")
	for n := 0; n < 3; n++ {
		tickWithMemberFailure(t, c)
		h.finishRuns(t)
	}
	e = h.get(t, e.ID)
	h.service.now = func() time.Time { return e.DeadlineAt.Add(time.Second) }
	tickWithMemberFailure(t, c)
	if e = h.get(t, e.ID); e.State != evaldomain.StateCancelling {
		t.Fatalf("wall deadline did not stop the experiment: %s", e.State)
	}
	admitted := count(t, h.pool, "eval_submissions")
	for n := 0; n < 10; n++ {
		tickWithMemberFailure(t, c)
		h.finishRuns(t)
	}
	e = h.get(t, e.ID)
	if got := count(t, h.pool, "eval_submissions"); got != admitted {
		t.Fatalf("cancelling admitted %d more members", got-admitted)
	}
	if terminal := terminalSubmissions(t, h, e.ID); e.State != evaldomain.StateCancelling || e.Outstanding != 1 || terminal != admitted-1 {
		t.Fatalf("other members did not drain: state=%s outstanding=%d terminal=%d of %d", e.State, e.Outstanding, terminal, admitted)
	}
	if got := publishedTerminal(t, h, e); got != admitted-1 {
		t.Fatalf("published view has %d terminal members, want %d", got, admitted-1)
	}
	// The uncertain intent replays once creation recovers, then drains too.
	h.driver.Runs = h.run.RunCreator
	for n := 0; n < 10 && e.State != evaldomain.StateCancelled; n++ {
		tick(t, c)
		h.finishRuns(t)
		e = h.get(t, e.ID)
	}
	if e.State != evaldomain.StateCancelled || e.Outstanding != 0 || count(t, h.pool, "eval_submissions") != admitted {
		t.Fatalf("deadline did not drain: state=%s outstanding=%d", e.State, e.Outstanding)
	}
}
