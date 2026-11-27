package gatewayrecovery

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

type fixture struct {
	s     *Service
	now   time.Time
	route Route
}

func memoryFixture(t *testing.T) *fixture {
	t.Helper()
	s, err := New(DefaultPolicy(), time.Minute, 30*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	f := &fixture{s: s, now: time.Date(2026, 11, 6, 0, 0, 0, 0, time.UTC), route: Route{OwnerID: "owner", GatewayDigest: "gateway", Model: "model"}}
	s.now = func() time.Time { return f.now }
	return f
}
func (f *fixture) participant(t *testing.T, id string, route Route) *allocationParticipant {
	t.Helper()
	if allowed, err := f.s.Admit(t.Context(), id, "stage-"+id, []Route{route}); err != nil || !allowed {
		t.Fatalf("admit %s: %v %v", id, allowed, err)
	}
	f.s.Activate(id, "stage-"+id)
	return newAllocationParticipant(t, f.s, id, "stage-"+id, route)
}
func update(t *testing.T, p *allocationParticipant, id, action string) Decision {
	t.Helper()
	r := Request{RequestID: id, Action: action}
	if action == "failed" {
		r.Code = "model_unavailable"
	}
	d, err := p.Update(t.Context(), r)
	if err != nil {
		t.Fatal(err)
	}
	return d
}
func TestMemoryRecoverySharedProbeQueuePriorityAndIndependentRoutes(t *testing.T) {
	f := memoryFixture(t)
	a, b := f.participant(t, "a", f.route), f.participant(t, "b", f.route)
	update(t, a, "outage", "failed")
	update(t, b, "wait", "acquire")
	f.now = f.now.Add(time.Second)
	if ok, err := f.s.Admit(t.Context(), "queued", "stage-queued", []Route{f.route}); err != nil || ok {
		t.Fatalf("queued bypassed waiters: %v %v", ok, err)
	}
	other := f.route
	other.Model = "other"
	if !update(t, f.participant(t, "independent", other), "healthy", "acquire").Allowed {
		t.Fatal("independent blocked")
	}
	var wg sync.WaitGroup
	decisions := make([]Decision, 2)
	errs := make([]error, 2)
	ps := []*allocationParticipant{a, b}
	for i, p := range ps {
		wg.Go(func() {
			decisions[i], errs[i] = p.Update(t.Context(), Request{RequestID: fmt.Sprint("probe", i), Action: "acquire"})
		})
	}
	wg.Wait()
	winner := -1
	for i, d := range decisions {
		if errs[i] != nil {
			t.Fatal(errs[i])
		}
		if d.Allowed {
			if winner != -1 {
				t.Fatal("two probes")
			}
			winner = i
		}
	}
	if winner == -1 {
		t.Fatal("no probe")
	}
	update(t, a, "outage", "succeeded")
	if !f.s.routes[f.route.Key()].blocked {
		t.Fatal("stale success reopened gate")
	}
	update(t, ps[winner], fmt.Sprint("probe", winner), "succeeded")
	if ok, err := f.s.Admit(t.Context(), "queued", "stage-queued", []Route{f.route}); err != nil || !ok {
		t.Fatalf("queue remained blocked: %v %v", ok, err)
	}
}
func TestMemoryRecoveryFailureRedeliveryAndStageCleanup(t *testing.T) {
	f := memoryFixture(t)
	a, b := f.participant(t, "a", f.route), f.participant(t, "b", f.route)
	update(t, a, "failure-a", "failed")
	update(t, b, "failure-b", "failed")
	state := f.s.routes[f.route.Key()]
	next := state.next
	update(t, a, "failure-a", "failed")
	if state.failures != 2 || state.next != next || f.s.failureReceipts != 2 {
		t.Fatal("redelivery counted twice")
	}
	f.s.EndStage("a", "stage-a")
	f.s.EndStage("b", "stage-b")
	if len(f.s.runs) != 0 || len(f.s.bindings) != 0 || f.s.failureReceipts != 0 || state.waiters != 0 || state.failures != 2 || !state.blocked {
		t.Fatal("cleanup lost gate or leaked state")
	}
	for _, action := range []string{"acquire", "failed"} {
		_, err := a.Update(t.Context(), Request{RequestID: "late", Action: action, Code: "model_unavailable"})
		if !IsUnavailable(err) {
			t.Fatalf("late %s: %v", action, err)
		}
	}
	for _, action := range []string{"succeeded", "finished", "released"} {
		update(t, a, "late", action)
	}
}
func TestMemoryRecoveryReleaseLeaseAndManualRetry(t *testing.T) {
	f := memoryFixture(t)
	a, b := f.participant(t, "a", f.route), f.participant(t, "b", f.route)
	update(t, a, "failure", "failed")
	f.now = f.now.Add(time.Second)
	if !update(t, a, "probe", "acquire").Allowed {
		t.Fatal("probe denied")
	}
	state := f.s.routes[f.route.Key()]
	if state.probeUntil.Sub(f.now) != 30*time.Minute+time.Second {
		t.Fatal("wrong Planner lease")
	}
	update(t, b, "stale", "released")
	if state.probeID != "probe" {
		t.Fatal("stale release")
	}
	update(t, a, "probe", "released")
	if !state.blocked || state.probeID != "" {
		t.Fatal("release reopened gate")
	}
	if !update(t, b, "replacement", "acquire").Allowed {
		t.Fatal("replacement denied")
	}
	update(t, a, "probe", "succeeded")
	if !state.blocked {
		t.Fatal("stale answer")
	}
	update(t, b, "replacement", "finished")
	if state.blocked {
		t.Fatal("permanent response did not reopen transport")
	}
	update(t, a, "again", "failed")
	f.now = f.now.Add(6 * time.Minute)
	before := state.automaticUntil
	if err := f.s.Retry(t.Context(), "intruder", "a"); !IsUnavailable(err) || state.automaticUntil != before {
		t.Fatal("Retry owner check")
	}
	if !update(t, a, "expired", "acquire").RequiresRetry {
		t.Fatal("window did not expire")
	}
	if err := f.s.Retry(t.Context(), "owner", "a"); err != nil {
		t.Fatal(err)
	}
	if !update(t, a, "manual", "acquire").Allowed {
		t.Fatal("manual retry denied")
	}
}
func TestMemoryRecoveryBindingsUseWorkerLeaseAndFenceStage(t *testing.T) {
	f := memoryFixture(t)
	p := f.participant(t, "run", f.route)
	if err := f.s.Bind(t.Context(), "allocation", "run", "stage-run", f.route, false); err != nil {
		t.Fatal(err)
	}
	call := func(id, action string) (Decision, error) {
		return f.s.Update(t.Context(), "allocation", Request{Model: f.route.Model, RequestID: id, Action: action, Code: "model_unavailable"})
	}
	if _, err := call("failure", "failed"); err != nil {
		t.Fatal(err)
	}
	f.now = f.now.Add(time.Second)
	if d, err := call("probe", "acquire"); err != nil || !d.Allowed {
		t.Fatalf("worker probe %v %v", d, err)
	}
	if f.s.routes[f.route.Key()].probeUntil.Sub(f.now) != time.Minute+time.Second {
		t.Fatal("Worker inherited Planner timeout")
	}
	f.s.EndStage("run", "stage-run")
	if _, err := call("late", "failed"); !IsUnavailable(err) {
		t.Fatalf("old allocation: %v", err)
	}
	other := f.route
	other.Model = "next"
	if ok, err := f.s.Admit(t.Context(), "run", "next-stage", []Route{other}); err != nil || !ok {
		t.Fatal(err)
	}
	f.s.Activate("run", "next-stage")
	f.s.EndStage("run", "stage-run")
	if _, err := p.Update(t.Context(), Request{RequestID: "late", Action: "failed", Code: "model_unavailable"}); !IsUnavailable(err) {
		t.Fatal("old Planner recreated state")
	}
	if f.s.runs["run"].stageID != "next-stage" {
		t.Fatal("old cleanup deleted next Stage")
	}
}
func TestMemoryRecoveryAdmissionReservationReadmissionAndRelease(t *testing.T) {
	f := memoryFixture(t)
	p := f.participant(t, "first", f.route)
	update(t, p, "outage", "failed")
	f.s.EndStage("first", "stage-first")
	f.now = f.now.Add(time.Second)
	admit := func(run string, route Route) bool {
		t.Helper()
		ok, err := f.s.Admit(t.Context(), run, "stage-"+run, []Route{route})
		if err != nil {
			t.Fatal(err)
		}
		return ok
	}
	if !admit("queued", f.route) || !admit("queued", f.route) || admit("other", f.route) {
		t.Fatal("reservation is not exclusive/reentrant")
	}
	f.s.ReleaseAdmission("queued", "stage-queued")
	if !admit("other", f.route) {
		t.Fatal("released reservation retained")
	}
	different := f.route
	different.Model = "different"
	if !admit("other", different) || !admit("queued", f.route) {
		t.Fatal("dropped candidate retained reservation")
	}
}
func TestMemoryRecoveryRestartForgetsStateAndRejectsOldAllocation(t *testing.T) {
	f := memoryFixture(t)
	p := f.participant(t, "old", f.route)
	update(t, p, "outage", "failed")
	fresh, err := New(DefaultPolicy(), time.Minute, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fresh.Update(t.Context(), "old", Request{Model: f.route.Model, RequestID: "late", Action: "failed", Code: "model_unavailable"}); !IsUnavailable(err) {
		t.Fatal("old allocation adopted")
	}
	old := &allocationParticipant{fresh, "unbound", f.route.Model}
	if _, err := old.Update(t.Context(), Request{RequestID: "late", Action: "acquire"}); !IsUnavailable(err) {
		t.Fatal("old Planner adopted")
	}
	if ok, err := fresh.Admit(t.Context(), "new", "stage-new", []Route{f.route}); err != nil || !ok {
		t.Fatal("restart retained outage")
	}
}
func TestMemoryRecoveryWaitNotificationsAndExactStatus(t *testing.T) {
	f := memoryFixture(t)
	p := f.participant(t, "run", f.route)
	code, changed := f.s.Waiting("run", "stage-run")
	if code != "" {
		t.Fatal(code)
	}
	update(t, p, "failure", "failed")
	select {
	case <-changed:
	default:
		t.Fatal("wait not signalled")
	}
	code, _ = f.s.Waiting("run", "stage-run")
	if code != "model_unavailable" {
		t.Fatal(code)
	}
	status, err := f.s.ReadStatus(t.Context(), "run", "stage-run")
	if err != nil || status == nil || status.RequiresRetry {
		t.Fatalf("status %+v %v", status, err)
	}
	if wrong, err := f.s.ReadStatus(t.Context(), "run", "other-stage"); err != nil || wrong != nil {
		t.Fatal("mixed Stage status")
	}
	f.s.EndStage("run", "stage-run")
	code, ch := f.s.Waiting("run", "stage-run")
	if code != "" || ch != nil {
		t.Fatal("ended wait remains")
	}
}
func TestMemoryRecoveryBoundsDoNotEvictActiveState(t *testing.T) {
	f := memoryFixture(t)
	p := f.participant(t, "active", f.route)
	for i := 0; i < maximumRoutes-1; i++ {
		route := f.route
		route.Model = fmt.Sprint("model-", i)
		// Model a retained outage without allocating 10000 live executions.
		f.s.routes[route.Key()] = &routeState{blocked: true, members: make(map[string]*runState)}
	}
	other := f.route
	other.Model = "overflow"
	if _, err := f.s.Admit(t.Context(), "overflow", "stage-overflow", []Route{other}); !errors.Is(err, errCapacity) {
		t.Fatalf("route capacity: %v", err)
	}
	if !update(t, p, "healthy", "acquire").Allowed {
		t.Fatal("capacity evicted active route")
	}
	for i := 0; i < maximumFailures; i++ {
		f.s.runs["active"].failures[fmt.Sprint(i)] = struct{}{}
	}
	f.s.failureReceipts = maximumFailures
	if _, err := p.Update(t.Context(), Request{RequestID: "overflow", Action: "failed", Code: "model_unavailable"}); !errors.Is(err, errCapacity) {
		t.Fatalf("receipt capacity: %v", err)
	}
	if f.s.routes[f.route.Key()].blocked {
		t.Fatal("refused failure mutated gate")
	}
	f.s.EndStage("active", "stage-active")
	if f.s.failureReceipts != 0 {
		t.Fatal("receipts leaked")
	}
}
func TestMemoryRecoveryCancelledUpdateDoesNotMutate(t *testing.T) {
	f := memoryFixture(t)
	p := f.participant(t, "active", f.route)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := p.Update(ctx, Request{RequestID: "cancelled", Action: "failed", Code: "model_unavailable"}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if f.s.failureReceipts != 0 || f.s.routes[f.route.Key()].blocked {
		t.Fatal("cancelled update mutated")
	}
}

func TestMemoryRecoveryRunAndBindingCapacityIsReleasedByStageCleanup(t *testing.T) {
	f := memoryFixture(t)
	for i := range maximumRuns {
		id := fmt.Sprint("run-", i)
		if allowed, err := f.s.Admit(t.Context(), id, "stage", []Route{f.route}); err != nil || !allowed {
			t.Fatalf("admission %d: %v %v", i, allowed, err)
		}
	}
	if _, err := f.s.Admit(t.Context(), "overflow", "stage", []Route{f.route}); !errors.Is(err, errCapacity) {
		t.Fatalf("membership bound: %v", err)
	}
	f.s.EndStage("run-0", "stage")
	if ok, err := f.s.Admit(t.Context(), "replacement", "stage", []Route{f.route}); err != nil || !ok {
		t.Fatalf("membership capacity not released: %v %v", ok, err)
	}
	for i := range maximumBindingsPerRun {
		if err := f.s.Bind(t.Context(), fmt.Sprint("allocation-", i), "replacement", "stage", f.route, false); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.s.Bind(t.Context(), "overflow", "replacement", "stage", f.route, false); !errors.Is(err, errCapacity) {
		t.Fatalf("binding bound: %v", err)
	}
	f.s.EndRun("replacement")
	if len(f.s.bindings) != 0 || len(f.s.runs) != maximumRuns-1 {
		t.Fatal("cleanup leaked bindings or membership")
	}
}

func TestMemoryRecoveryExistingWaiterRefreshesChangedRouteCause(t *testing.T) {
	f := memoryFixture(t)
	a, b := f.participant(t, "a", f.route), f.participant(t, "b", f.route)
	update(t, a, "failure", "failed")
	update(t, b, "waiting", "acquire")
	_, changed := f.s.Waiting("b", "stage-b")
	if _, err := a.Update(t.Context(), Request{RequestID: "different-failure", Action: "failed", Code: "gateway_timeout"}); err != nil {
		t.Fatal(err)
	}
	update(t, b, "waiting", "acquire")
	select {
	case <-changed:
	default:
		t.Fatal("existing waiter did not signal its changed cause")
	}
	if code, _ := f.s.Waiting("b", "stage-b"); code != "gateway_timeout" {
		t.Fatal(code)
	}
}

func TestMemoryRecoveryPlannerAllocationUsesPlannerLease(t *testing.T) {
	f := memoryFixture(t)
	f.s.plannerRequestTimeout = 9 * time.Minute
	f.participant(t, "run", f.route)
	if err := f.s.Bind(t.Context(), "planner", "run", "stage-run", f.route, true); err != nil {
		t.Fatal(err)
	}
	call := func(id, action string) Decision {
		t.Helper()
		d, err := f.s.Update(t.Context(), "planner", Request{Model: f.route.Model, RequestID: id, Action: action, Code: "model_unavailable"})
		if err != nil {
			t.Fatal(err)
		}
		return d
	}
	call("failure", "failed")
	f.now = f.now.Add(time.Second)
	if !call("probe", "acquire").Allowed {
		t.Fatal("Planner probe denied")
	}
	want := f.now.Add(f.s.plannerRequestTimeout + f.s.policy.InitialDelay)
	if got := f.s.routes[f.route.Key()].probeUntil; !got.Equal(want) {
		t.Fatalf("Planner allocation lease = %s, want %s", got, want)
	}
	f.s.EndStage("run", "stage-run")
	if _, err := f.s.Update(t.Context(), "planner", Request{Model: f.route.Model, RequestID: "late", Action: "acquire"}); !IsUnavailable(err) {
		t.Fatal("retired Planner allocation recreated recovery state")
	}
}

// Exercise the production allocation protocol rather than the removed Go-model
// participant. This fixture owns no recovery state.
type allocationParticipant struct {
	s         *Service
	id, model string
}

func newAllocationParticipant(t *testing.T, s *Service, run, stage string, route Route) *allocationParticipant {
	t.Helper()
	id := "test-planner-" + stage
	if err := s.Bind(t.Context(), id, run, stage, route, true); err != nil {
		t.Fatal(err)
	}
	return &allocationParticipant{s, id, route.Model}
}
func (p *allocationParticipant) Update(ctx context.Context, r Request) (Decision, error) {
	r.Model = p.model
	return p.s.Update(ctx, p.id, r)
}
