package public

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// The fakes of this package's handler tests must refuse what the PostgreSQL
// stores refuse, or a handler regression that asks for an invalid operation
// passes its tests (V368-010). Each table below drives the fake and the real
// store with the same operations and requires the same answer.

var parityRunStates = []runstore.WorkflowRunState{
	runstore.RunInitializing, runstore.RunPending, runstore.RunRunning, runstore.RunWaiting,
	runstore.RunCancelling, runstore.RunSucceeded, runstore.RunFailed, runstore.RunCancelled,
}

// TestFakeRunStoreTransitionsMatchPostgres tries every transition between two
// WorkflowRun states on a Run that is in the expected state: the fake and the
// store accept the same edges and refuse every other one as invalid.
func TestFakeRunStoreTransitionsMatchPostgres(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	transition := func(store interface {
		TransitionRun(context.Context, string, runstore.WorkflowRunState, runstore.WorkflowRunState, runstore.Reason) (runstore.WorkflowRun, error)
	}, runID string, from, to runstore.WorkflowRunState) error {
		_, err := store.TransitionRun(ctx, runID, from, to, runstore.Reason{Code: "parity"})
		return err
	}
	fakeErrors := map[[2]runstore.WorkflowRunState]error{}
	refused := 0
	for _, from := range parityRunStates {
		for _, to := range parityRunStates {
			fake := newFakeRunStore()
			fake.runs["run-parity"] = runstore.WorkflowRun{RunID: "run-parity", State: from}
			err := transition(fake, "run-parity", from, to)
			if err != nil && !errors.Is(err, runstore.ErrInvalid) {
				t.Errorf("fake %s -> %s on a Run in %s: %v, want acceptance or ErrInvalid", from, to, from, err)
			}
			if err != nil {
				refused++
			}
			fakeErrors[[2]runstore.WorkflowRunState{from, to}] = err
		}
	}
	if err := fakeErrors[[2]runstore.WorkflowRunState{runstore.RunSucceeded, runstore.RunRunning}]; !errors.Is(err, runstore.ErrInvalid) {
		t.Fatalf("fake succeeded -> running = %v, want ErrInvalid", err)
	}
	if refused == 0 || refused == len(fakeErrors) {
		t.Fatalf("fake refused %d of %d transitions; the table tests nothing", refused, len(fakeErrors))
	}

	pool := isolatedPublicPool(t, ctx)
	store := runstore.NewPostgresStore(pool)
	for _, from := range parityRunStates {
		for _, to := range parityRunStates {
			runID := fmt.Sprintf("run-parity-%s-%s", from, to)
			createParityRunInState(t, ctx, store, runID, from)
			err := transition(store, runID, from, to)
			want := fakeErrors[[2]runstore.WorkflowRunState{from, to}]
			if (err == nil) != (want == nil) || err != nil && !errors.Is(err, runstore.ErrInvalid) {
				t.Errorf("PostgreSQL %s -> %s = %v, fake = %v", from, to, err, want)
			}
		}
	}
}

// createParityRunInState moves a new Run into state through the store's own
// operations, so the Run is exactly what a transition from state finds.
func createParityRunInState(
	t *testing.T, ctx context.Context, store *runstore.PostgresStore, runID string, state runstore.WorkflowRunState,
) {
	t.Helper()
	if _, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: "user-parity", WorkflowName: "artifact-copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
		RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	steps := map[runstore.WorkflowRunState][]runstore.WorkflowRunState{
		runstore.RunInitializing: nil,
		runstore.RunPending:      {runstore.RunPending},
		runstore.RunRunning:      {runstore.RunPending, runstore.RunRunning},
		runstore.RunWaiting:      {runstore.RunPending, runstore.RunRunning, runstore.RunWaiting},
		runstore.RunSucceeded:    {runstore.RunPending, runstore.RunRunning, runstore.RunSucceeded},
		runstore.RunFailed:       {runstore.RunFailed},
	}
	current := runstore.RunInitializing
	if state == runstore.RunCancelling || state == runstore.RunCancelled {
		if _, err := store.RequestRunCancellation(ctx, runID, runstore.WorkflowRunCancellation{
			Code: runstore.CancellationUserRequested, RequestedAt: time.Now(),
		}); err != nil {
			t.Fatal(err)
		}
		current = runstore.RunCancelling
		if state == runstore.RunCancelled {
			steps[state] = []runstore.WorkflowRunState{runstore.RunCancelled}
		}
	}
	for _, next := range steps[state] {
		if _, err := store.TransitionRun(ctx, runID, current, next, runstore.Reason{Code: "parity"}); err != nil {
			t.Fatalf("move Run %s from %s to %s: %v", runID, current, next, err)
		}
		current = next
	}
	run, err := store.GetRun(ctx, runID)
	if err != nil || run.State != state {
		t.Fatalf("Run %s = (%s, %v), want %s", runID, run.State, err, state)
	}
}

// TestFakeArtifactRepositoryWritesMatchPostgres refuses the same binding
// writes in the fake and the PostgreSQL repository: a frozen Run output takes
// no write whatever revision the write expects, and an open binding takes
// only the write that expects its current revision.
func TestFakeArtifactRepositoryWritesMatchPostgres(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	want := []error{
		artifacts.ErrArtifactFrozen, artifacts.ErrArtifactFrozen, artifacts.ErrArtifactFrozen,
		artifacts.ErrArtifactConflict, artifacts.ErrArtifactConflict,
		artifacts.ErrArtifactConflict, artifacts.ErrArtifactConflict, artifacts.ErrArtifactConflict,
	}
	check := func(name string, refusals []error) {
		t.Helper()
		if len(refusals) != len(want) {
			t.Fatalf("%s refused %d writes, want %d", name, len(refusals), len(want))
		}
		for index, err := range refusals {
			if !errors.Is(err, want[index]) ||
				errors.Is(want[index], artifacts.ErrArtifactConflict) && errors.Is(err, artifacts.ErrArtifactFrozen) {
				t.Errorf("%s write %d = %v, want %v", name, index, err, want[index])
			}
		}
	}
	check("fake", bindingWriteRefusals(t, ctx, newFakeArtifactRepository()))

	pool := isolatedPublicPool(t, ctx)
	store := runstore.NewPostgresStore(pool)
	for _, runID := range []string{"run-frozen-outputs", "run-open-outputs"} {
		createParityRunInState(t, ctx, store, runID, runstore.RunRunning)
	}
	check("PostgreSQL", bindingWriteRefusals(t, ctx, artifacts.NewPostgresRepository(pool)))
}

// bindingWriteRefusals binds an output of run-frozen-outputs and freezes it,
// binds an open output of run-open-outputs and writes a user artifact, then
// returns how each write in the table was refused, in order. The Runs exist
// wherever the repository needs them.
func bindingWriteRefusals(t *testing.T, ctx context.Context, repository artifacts.Repository) []error {
	t.Helper()
	service := artifacts.NewService(repository)
	payload := artifacts.Payload{MediaType: "text/plain", Data: []byte("parity\n")}
	source := func(runID string) artifacts.ArtifactRef {
		t.Helper()
		store, err := service.Run(runID)
		if err != nil {
			t.Fatal(err)
		}
		written, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "scratch", Name: "source"}, payload, nil)
		if err != nil {
			t.Fatal(err)
		}
		return written.Ref
	}
	frozenSource, openSource := source("run-frozen-outputs"), source("run-open-outputs")
	frozen, err := service.BindOutputExact(ctx, "run-frozen-outputs", "result", frozenSource, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := service.FreezeRunOutputs(ctx, "run-frozen-outputs"); err != nil {
		t.Fatal(err)
	}
	if _, err := service.BindOutputExact(ctx, "run-open-outputs", "result", openSource, nil); err != nil {
		t.Fatal(err)
	}
	user, err := service.User("user-parity")
	if err != nil {
		t.Fatal(err)
	}
	draft := contracts.ArtifactRef{Namespace: "notes", Name: "draft"}
	if _, err := user.Write(ctx, draft, payload, nil); err != nil {
		t.Fatal(err)
	}
	stale := "rev-stale"
	bind := func(runID string, source artifacts.ArtifactRef, expected *string) func() error {
		return func() error {
			_, err := service.BindOutputExact(ctx, runID, "result", source, expected)
			return err
		}
	}
	write := func(ref contracts.ArtifactRef, expected *string) func() error {
		return func() error {
			_, err := user.Write(ctx, ref, payload, expected)
			return err
		}
	}
	refusals := make([]error, 0, 8)
	for _, refused := range []func() error{
		bind("run-frozen-outputs", frozenSource, frozen.TargetRef.Revision),
		bind("run-frozen-outputs", frozenSource, nil),
		bind("run-frozen-outputs", frozenSource, &stale),
		bind("run-open-outputs", openSource, nil),
		bind("run-open-outputs", openSource, &stale),
		write(draft, nil),
		write(draft, &stale),
		write(contracts.ArtifactRef{Namespace: "notes", Name: "absent"}, &stale),
	} {
		refusals = append(refusals, refused())
	}
	return refusals
}
