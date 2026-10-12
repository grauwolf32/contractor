package public

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
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
			createParityRunInState(t, ctx, pool, runID, from)
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
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string, state runstore.WorkflowRunState,
) {
	t.Helper()
	store := runstore.NewPostgresStore(pool)
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
		var err error
		if next == runstore.RunWaiting {
			// Only the waiting operation moves a Run into waiting, under the
			// Run row lock that model-availability recovery holds.
			err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				return runstore.NewPostgresTxStore(tx).EnterWaiting(ctx, runID, "parity")
			})
		} else {
			_, err = store.TransitionRun(ctx, runID, current, next, runstore.Reason{Code: "parity"})
		}
		if err != nil {
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
	for _, runID := range []string{"run-frozen-outputs", "run-open-outputs"} {
		createParityRunInState(t, ctx, pool, runID, runstore.RunRunning)
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

// TestFakeRuntimeAgentPrincipalKeysMatchPostgres drives the fake and the real
// principal manager on PostgreSQL through the same keyed label replacements
// and deletions. Both remember the key of a committed mutation and replay its
// result for the same request, refuse the key for another request or
// operation as a permanent idempotency conflict, and forget the key of a
// refused mutation.
func TestFakeRuntimeAgentPrincipalKeysMatchPostgres(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	principalID := strings.Repeat("c", 64)
	fake := newFakeRuntimeAgentPrincipalManagement()
	fake.principals[principalID] = controlplane.RuntimeAgentPrincipalProjection{
		Principal: agentprincipal.Principal{RuntimeAgentID: principalID, Labels: []string{}, LabelRevision: 1},
	}
	checkPrincipalKeyMemory(t, ctx, "fake", fake, principalID)

	pool := isolatedPublicPool(t, ctx)
	// A Runtime Agent label must select Worker settings.
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool, RuntimeCredentials: parityRuntimeCredentials{},
		PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
	})
	if err != nil {
		t.Fatal(err)
	}
	published, err := publisher.Publish(ctx, []byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"parity-telemetry","version":"1"},
  "spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces"}}}
}`), "parity-telemetry", "operator-parity")
	if err != nil {
		t.Fatal(err)
	}
	at := time.Now().UTC()
	if _, err := runtimeconfig.NewRepository(pool).CreateBinding(
		ctx, "parity-label", published.Version.Ref, "operator-parity", at,
	); err != nil {
		t.Fatal(err)
	}
	if inserted, err := agentprincipal.NewRepository(pool).Insert(ctx, agentprincipal.Principal{
		RuntimeAgentID: principalID, Labels: []string{}, LabelRevision: 1,
		CreatedBy: "operator-parity", CreatedAt: at, UpdatedBy: "operator-parity", UpdatedAt: at,
	}); err != nil || !inserted {
		t.Fatalf("insert principal = (%t, %v)", inserted, err)
	}
	registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{})
	if err != nil {
		t.Fatal(err)
	}
	service, err := agentprincipal.NewService(agentprincipal.ServiceOptions{
		Pool: pool, DeletionGuard: registry,
	})
	if err != nil {
		t.Fatal(err)
	}
	operations, err := controlplane.NewPrincipalOperations(service, registry)
	if err != nil {
		t.Fatal(err)
	}
	checkPrincipalKeyMemory(t, ctx, "PostgreSQL", operations, principalID)
}

// parityRuntimeCredentials serves the publication of a RuntimeConfig that
// references no Runtime credential.
type parityRuntimeCredentials struct{}

func (parityRuntimeCredentials) ValidateRuntimeCredential(context.Context, string, ...string) error {
	return errors.New("the principal parity RuntimeConfig references no Runtime credential")
}

func (parityRuntimeCredentials) WithCredentialReferences(_ context.Context, fn func() error) error {
	return fn()
}

// checkPrincipalKeyMemory runs the keyed mutation table against a principal
// that starts unlabelled at revision 1 and may carry the label parity-label.
func checkPrincipalKeyMemory(
	t *testing.T, ctx context.Context, name string, manager RuntimeAgentPrincipalManagement, principalID string,
) {
	t.Helper()
	type outcome struct {
		err      error
		replayed bool
		revision uint64
		labels   []string
	}
	at := time.Date(2026, 10, 11, 12, 0, 0, 0, time.UTC)
	labelled, unlabelled := []string{"parity-label"}, []string{}
	replace := func(expected uint64, labels []string, key string) outcome {
		principal, replayed, err := manager.ReplaceLabels(ctx, principalID, expected, labels, key, "operator-parity", at)
		return outcome{err, replayed, principal.Principal.LabelRevision, principal.Principal.Labels}
	}
	remove := func(expected uint64, key string) outcome {
		replayed, err := manager.Delete(ctx, principalID, expected, key, "operator-parity", at)
		return outcome{err: err, replayed: replayed}
	}
	for _, step := range []struct {
		name string
		run  func() outcome
		want outcome
	}{
		{"label the principal", func() outcome { return replace(1, labelled, "label") },
			outcome{revision: 2, labels: labelled}},
		{"repeat the labelling", func() outcome { return replace(1, labelled, "label") },
			outcome{replayed: true, revision: 2, labels: labelled}},
		{"reuse the key for other labels", func() outcome { return replace(1, unlabelled, "label") },
			outcome{err: contracts.ErrIdempotencyConflict}},
		{"reuse the key for a deletion", func() outcome { return remove(2, "label") },
			outcome{err: contracts.ErrIdempotencyConflict}},
		{"delete the labelled principal", func() outcome { return remove(2, "delete") },
			outcome{err: agentprincipal.ErrInUse}},
		{"clear the labels", func() outcome { return replace(2, unlabelled, "clear") },
			outcome{revision: 3, labels: unlabelled}},
		{"repeat the labelling after a newer change", func() outcome { return replace(1, labelled, "label") },
			outcome{replayed: true, revision: 2, labels: labelled}},
		{"delete at a stale revision under the refused key", func() outcome { return remove(2, "delete") },
			outcome{err: runtimeconfig.ErrPrecondition}},
		{"delete under the refused key", func() outcome { return remove(3, "delete") },
			outcome{}},
		{"repeat the deletion", func() outcome { return remove(3, "delete") },
			outcome{replayed: true}},
		{"repeat the labelling of the deleted principal", func() outcome { return replace(1, labelled, "label") },
			outcome{err: runtimeconfig.ErrNotFound}},
		{"delete the deleted principal", func() outcome { return remove(3, "delete-again") },
			outcome{err: runtimeconfig.ErrNotFound}},
	} {
		got := step.run()
		if step.want.err != nil {
			if !errors.Is(got.err, step.want.err) || got.replayed {
				t.Errorf("%s: %s = (%v, replayed %t), want %v", name, step.name, got.err, got.replayed, step.want.err)
			}
			continue
		}
		if got.err != nil || got.replayed != step.want.replayed || got.revision != step.want.revision ||
			!slices.Equal(got.labels, step.want.labels) {
			t.Errorf("%s: %s = (%v, replayed %t, revision %d, labels %v), want (replayed %t, revision %d, labels %v)",
				name, step.name, got.err, got.replayed, got.revision, got.labels,
				step.want.replayed, step.want.revision, step.want.labels)
		}
	}
}
