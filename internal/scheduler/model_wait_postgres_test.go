package scheduler

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// blockRecoveryRoute uses the production participant protocol rather than SQL
// edits. A due gate has no remaining invocation waiter; otherwise the returned
// function recovers its current probe and then unregisters the participant.
func blockRecoveryRoute(t *testing.T, service *gatewayrecovery.Service, route gatewayrecovery.Route, due bool) func() {
	t.Helper()
	id := "outage-" + route.Key()
	if ok, err := service.Admit(t.Context(), id, id, []gatewayrecovery.Route{route}); err != nil || !ok {
		t.Fatalf("outage admission %v %v", ok, err)
	}
	service.Activate(id, id)
	p := service.Planner(id, id, route, llmgateway.DefaultGatewayFailureSignatures())
	if _, err := p.Update(t.Context(), gatewayrecovery.Request{RequestID: "failure", Action: "failed", Code: "model_unavailable", RetryAfterSeconds: 3600}); err != nil {
		t.Fatal(err)
	}
	recover := func() {
		t.Helper()
		if err := service.Retry(t.Context(), route.OwnerID, id); err != nil {
			t.Fatal(err)
		}
		if d, err := p.Update(t.Context(), gatewayrecovery.Request{RequestID: "probe", Action: "acquire"}); err != nil || !d.Allowed {
			t.Fatalf("probe %v %v", d, err)
		}
		if _, err := p.Update(t.Context(), gatewayrecovery.Request{RequestID: "probe", Action: "succeeded"}); err != nil {
			t.Fatal(err)
		}
		service.EndStage(id, id)
	}
	if due {
		if err := service.Retry(t.Context(), route.OwnerID, id); err != nil {
			t.Fatal(err)
		}
		service.EndStage(id, id)
	}
	return recover
}

func TestPostgresModelWaitProjectionIsIdempotentAndStageFenced(t *testing.T) {
	ctx := t.Context()
	pool := isolatedSchedulerPool(t, ctx)
	f := createRunningStageFixture(t, ctx, pool)
	claimed := fixtureClaim(t, ctx, pool, "run-1")
	for _, code := range []string{"model_unavailable", "model_unavailable", "gateway_timeout", "", ""} {
		if err := f.persistence.ReconcileModelWait(claimed, "run-1", f.executionID, code); err != nil {
			t.Fatal(err)
		}
		run, err := f.store.GetRun(ctx, "run-1")
		want := runstore.RunWaiting
		if code == "" {
			want = runstore.RunRunning
		}
		if err != nil || run.State != want {
			t.Fatalf("projection %q -> %s %v", code, run.State, err)
		}
	}
	// Once cancellation commits, even a stale wait snapshot changes nothing.
	requestTestCancellation(t, ctx, f.store)
	if err := f.persistence.ReconcileModelWait(claimed, "run-1", f.executionID, "model_unavailable"); err != nil {
		t.Fatal(err)
	}
	run, err := f.store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunCancelling {
		t.Fatalf("cancelling %s %v", run.State, err)
	}
}

type delayedWaitProjection struct {
	AtomicPersistence
	attempted chan string
	applied   chan string
	release   <-chan struct{}
	ambiguous bool
}

func (p *delayedWaitProjection) ReconcileModelWait(ctx context.Context, runID, stageID, code string) error {
	first := p.release != nil
	if first {
		p.attempted <- code
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-p.release:
		}
		p.release = nil
	}
	err := p.AtomicPersistence.ReconcileModelWait(ctx, runID, stageID, code)
	if err == nil {
		p.applied <- code
	}
	if first && p.ambiguous && err == nil {
		return errors.New("injected lost commit response")
	}
	return err
}

func TestPostgresModelOutcomeDoesNotWaitForRunProjection(t *testing.T) {
	for _, ambiguous := range []bool{false, true} {
		t.Run(fmt.Sprint("ambiguous=", ambiguous), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			defer cancel()
			pool := isolatedSchedulerPool(t, ctx)
			f := createRunningStageFixture(t, ctx, pool)
			claim := fixtureClaim(t, ctx, pool, "run-1")
			service, err := gatewayrecovery.New(gatewayrecovery.DefaultPolicy(), time.Minute, time.Minute)
			if err != nil {
				t.Fatal(err)
			}
			route := gatewayrecovery.Route{OwnerID: "user-1", GatewayDigest: "gateway", Model: "model"}
			other := route
			other.Model = "other"
			if route.Key() > other.Key() {
				route, other = other, route
			}
			if ok, err := service.Admit(ctx, "run-1", f.executionID, []gatewayrecovery.Route{route, other}); err != nil || !ok {
				t.Fatal(err)
			}
			service.Activate("run-1", f.executionID)
			release := make(chan struct{})
			projection := &delayedWaitProjection{AtomicPersistence: f.persistence, attempted: make(chan string, 1), applied: make(chan string, 4), release: release, ambiguous: ambiguous}
			s := &Scheduler{persistence: projection, options: Options{GatewayRecovery: service, Clock: realClock{}, OperationTimeout: 5 * time.Second, PollInterval: time.Second, Logger: slog.Default()}}
			run, err := f.store.GetRun(ctx, "run-1")
			if err != nil {
				t.Fatal(err)
			}
			stop := s.monitorModelWaits(claim, run, f.executionID)
			defer stop()
			recoverRoute := blockRecoveryRoute(t, service, route, false)
			recoverOther := blockRecoveryRoute(t, service, other, false)
			for _, item := range []struct {
				route gatewayrecovery.Route
				code  string
			}{{route, "model_unavailable"}, {other, "gateway_timeout"}} {
				id := "allocation-" + item.route.Model
				if err := service.Bind(ctx, id, "run-1", f.executionID, item.route, false); err != nil {
					t.Fatal(err)
				}
				if _, err := service.Update(ctx, id, gatewayrecovery.Request{Model: item.route.Model, RequestID: "failure", Action: "failed", Code: item.code}); err != nil {
					t.Fatal(err)
				}
			}
			select {
			case <-projection.attempted:
			case <-ctx.Done():
				t.Fatal("projection did not start")
			}
			// A different Run recovers the first route while the write is blocked.
			recoverRoute()
			close(release)
			for _, want := range []string{"model_unavailable", "gateway_timeout", ""} {
				if want == "" {
					recoverOther()
				}
				select {
				case got := <-projection.applied:
					if got != want {
						t.Fatalf("projection %q want %q", got, want)
					}
				case <-ctx.Done():
					t.Fatal("projection did not converge")
				}
				if want == "gateway_timeout" {
					run, err := f.store.GetRun(ctx, "run-1")
					if err != nil || run.State != runstore.RunWaiting || run.StateReason.Code != want {
						t.Fatalf("remaining wait: %s %+v %v", run.State, run.StateReason, err)
					}
					d, err := service.Update(ctx, "allocation-"+route.Model, gatewayrecovery.Request{Model: route.Model, RequestID: "next", Action: "acquire"})
					if err != nil || !d.Allowed {
						t.Fatalf("healthy reacquire: %+v %v", d, err)
					}
				}
			}
			run, err = f.store.GetRun(ctx, "run-1")
			if err != nil || run.State != runstore.RunRunning {
				t.Fatalf("stuck Run %s %v", run.State, err)
			}
			service.EndStage("run-1", f.executionID)
		})
	}
}

func TestCancellationImmediatelyUnregistersMemoryRecovery(t *testing.T) {
	service, err := gatewayrecovery.New(gatewayrecovery.DefaultPolicy(), time.Minute, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	route := gatewayrecovery.Route{OwnerID: "owner", GatewayDigest: "gateway", Model: "model"}
	if ok, err := service.Admit(t.Context(), "run", "stage", []gatewayrecovery.Route{route}); err != nil || !ok {
		t.Fatal(err)
	}
	service.Activate("run", "stage")
	p := service.Planner("run", "stage", route, llmgateway.DefaultGatewayFailureSignatures())
	if _, err := p.Update(t.Context(), gatewayrecovery.Request{RequestID: "outage", Action: "failed", Code: "model_unavailable"}); err != nil {
		t.Fatal(err)
	}
	s := &Scheduler{options: Options{GatewayRecovery: service}, wake: make(chan struct{}, 1)}
	s.Cancel("run")
	if _, err := p.Update(t.Context(), gatewayrecovery.Request{RequestID: "late", Action: "failed", Code: "model_unavailable"}); !gatewayrecovery.IsUnavailable(err) {
		t.Fatalf("cancelled participant still active: %v", err)
	}
	if err := service.Retry(t.Context(), "owner", "run"); !gatewayrecovery.IsUnavailable(err) {
		t.Fatalf("cancelled Run reopened window: %v", err)
	}
	if status, err := service.ReadStatus(t.Context(), "run", "stage"); err != nil || status != nil {
		t.Fatalf("cancelled status: %+v %v", status, err)
	}
}
