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
			if ok, err := service.Admit(ctx, "run-1", f.executionID, []gatewayrecovery.Route{route}); err != nil || !ok {
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
			p := service.Planner("run-1", f.executionID, route, llmgateway.DefaultGatewayFailureSignatures())
			if _, err := p.Update(ctx, gatewayrecovery.Request{RequestID: "failure", Action: "failed", Code: "model_unavailable"}); err != nil {
				t.Fatal(err)
			}
			select {
			case <-projection.attempted:
			case <-ctx.Done():
				t.Fatal("projection did not start")
			}
			// Model completion proceeds while the database projection is blocked.
			if err := service.Retry(ctx, "user-1", "run-1"); err != nil {
				t.Fatal(err)
			}
			if d, err := p.Update(ctx, gatewayrecovery.Request{RequestID: "probe", Action: "acquire"}); err != nil || !d.Allowed {
				t.Fatalf("probe %v %v", d, err)
			}
			if _, err := p.Update(ctx, gatewayrecovery.Request{RequestID: "probe", Action: "succeeded"}); err != nil {
				t.Fatal(err)
			}
			close(release)
			// The stale waiting write may commit (even with a lost response), but
			// the next snapshot must clear it without repeating the model call.
			for _, want := range []string{"model_unavailable", ""} {
				select {
				case got := <-projection.applied:
					if got != want {
						t.Fatalf("projection %q want %q", got, want)
					}
				case <-ctx.Done():
					t.Fatal("projection did not converge")
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
