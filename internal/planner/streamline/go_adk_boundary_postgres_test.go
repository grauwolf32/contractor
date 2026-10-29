package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/adk/v2/model"
	adksession "google.golang.org/adk/v2/session"
)

// The database commit is real; only the acknowledgement loss is injected by
// the forwarding Store. This must fence the turn, never repeat its append.
func TestPostgresModeledPlannerAmbiguousAppendAck(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		t.Run(profile, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool := isolatedStreamlinePool(t, ctx)
			store := newMigrationBoundaryStore(t, ctx, pool)
			store.afterOrdinary = func(_ context.Context, _ runstore.AppendPlannerEventParams, err error) error {
				if err != nil {
					return err
				}
				return io.ErrUnexpectedEOF
			}
			sessions := migrationBoundarySession(t, store)
			llm := &scriptedModel{steps: []modelStep{textStep("Bounded ordinary response")}}
			workers := appendProbeWorkers()
			factory := migrationBoundaryFactory(t, profile, sessions, workers, llm)
			invocation := appendProbeInvocation(profile)
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			result, runErr := instance.Run(ctx)
			store.returned.Store(true)
			assertAppendProbeFailure(t, result, runErr, "planner_session_unavailable", true)
			var writeError *plannersession.ADKEventWriteError
			if !errors.Is(runErr, io.ErrUnexpectedEOF) || !errors.As(runErr, &writeError) {
				t.Fatal("ambiguous append lost its original cause or required-write classification")
			}
			if store.ordinary.Load() != 1 || llm.callCount() != 1 || len(workers.bindingCalls()) != 0 {
				t.Fatal("ambiguous acknowledgement repeated an append or admitted semantic work")
			}
			assertMigrationBoundaryJournal(t, ctx, pool, profile, 1)
			assertMigrationBoundaryRecovery(t, ctx, factory, invocation, store, "planner_session_unavailable", true)
			if llm.callCount() != 1 || len(workers.bindingCalls()) != 0 {
				t.Fatal("completed recovery repeated ambiguous work")
			}
		})
	}
}

// The model barrier lets initial request writes finish before a real Run-row
// lock blocks the ordinary ADK event's SQL statement. Cancellation is observed
// after the production delegate returns, before the lock allows completion.
func TestPostgresModeledPlannerCancelDuringEventAppend(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		t.Run(profile, func(t *testing.T) {
			ctx, stop := context.WithTimeout(t.Context(), 45*time.Second)
			defer stop()
			pool := isolatedStreamlinePool(t, ctx)
			store := newMigrationBoundaryStore(t, ctx, pool)
			appendReturned := make(chan error, 1)
			store.afterOrdinary = func(_ context.Context, _ runstore.AppendPlannerEventParams, err error) error {
				appendReturned <- err
				return err
			}
			sessions := migrationBoundarySession(t, store)
			modelEntered, allowModel := make(chan struct{}), make(chan struct{})
			var release sync.Once
			releaseModel := func() { release.Do(func() { close(allowModel) }) }
			llm := &scriptedModel{steps: []modelStep{func(request *model.LLMRequest) (*model.LLMResponse, error) {
				close(modelEntered)
				<-allowModel
				return textStep("Persist this ordinary response")(request)
			}}}
			workers := appendProbeWorkers()
			factory := migrationBoundaryFactory(t, profile, sessions, workers, llm)
			invocation := appendProbeInvocation(profile)
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			runContext, cancel := context.WithCancel(ctx)
			finished, done := startMigrationBoundaryRun(instance, runContext, store)
			t.Cleanup(func() { cancel(); releaseModel(); migrationBoundaryJoin(t, ctx, finished) })
			migrationWait(t, ctx, modelEntered)
			holder, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
				defer cancel()
				_ = holder.Rollback(cleanup)
			})
			var holderPID int
			if err := holder.QueryRow(ctx, `SELECT pg_backend_pid() FROM workflow_runs WHERE run_id = 'run-1' FOR UPDATE`).Scan(&holderPID); err != nil {
				t.Fatal(err)
			}
			releaseModel()
			waitMigrationBoundarySQL(t, ctx, pool, holderPID, finished)
			cancel()
			select {
			case appendErr := <-appendReturned:
				if !errors.Is(appendErr, context.Canceled) {
					t.Fatal("in-flight PostgreSQL append did not preserve cancellation")
				}
			case <-ctx.Done():
				t.Fatal("PostgreSQL append did not join cancellation")
			}
			// Release only after the cancelled SQL returned. The independent
			// completion can now settle under its own healthy bounded context.
			if err := holder.Rollback(ctx); err != nil {
				t.Fatal(err)
			}
			migrationBoundaryJoin(t, ctx, finished)
			returned := <-done
			assertAppendProbeFailure(t, returned.result, returned.err, "planner_cancelled", true)
			if !errors.Is(returned.err, context.Canceled) || store.ordinary.Load() != 1 {
				t.Fatal("Planner lost SQL cancellation or repeated the cancelled append")
			}
			assertMigrationBoundaryJournal(t, ctx, pool, profile, 0)
			assertMigrationBoundaryRecovery(t, ctx, factory, invocation, store, "planner_cancelled", true)
			if llm.callCount() != 1 || len(workers.bindingCalls()) != 0 {
				t.Fatal("cancelled persistence admitted a later model or Worker call")
			}
		})
	}
}

func TestModeledPlannerCancelDuringWorkerJoin(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		t.Run(profile, func(t *testing.T) {
			ctx, stop := context.WithTimeout(t.Context(), 45*time.Second)
			defer stop()
			pool := isolatedStreamlinePool(t, ctx)
			store := newMigrationBoundaryStore(t, ctx, pool)
			sessions := migrationBoundarySession(t, store)
			workers := &migrationJoiningWorker{started: make(chan struct{}), cancelling: make(chan struct{}), release: make(chan struct{}), joined: make(chan struct{})}
			var release sync.Once
			releaseWorker := func() { release.Do(func() { close(workers.release) }) }
			execute := map[string]any{"subtask_id": "0"}
			if profile == "router" {
				execute["worker_name"] = "builder"
			}
			llm := &scriptedModel{steps: []modelStep{addSubtaskStep("Bounded work", "Produce bounded evidence"), functionStep(executeCurrentSubtaskToolName, execute)}}
			factory := migrationBoundaryFactory(t, profile, sessions, workers, llm)
			invocation := appendProbeInvocation(profile)
			// The execution deadline must precede the independent test hang
			// guard; otherwise context inheritance correctly shortens it.
			invocation.Deadline = time.Now().Add(30 * time.Second)
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			runContext, cancel := context.WithCancel(ctx)
			finished, done := startMigrationBoundaryRun(instance, runContext, store)
			t.Cleanup(func() { cancel(); releaseWorker(); migrationBoundaryJoin(t, ctx, finished) })
			migrationWait(t, ctx, workers.started)
			if !workers.deadline.Equal(invocation.Deadline) || !workers.requestDeadline.Equal(invocation.Deadline) {
				t.Fatal("Worker call replaced the original invocation deadline")
			}
			cancel()
			migrationWait(t, ctx, workers.cancelling)
			select {
			case <-finished:
				t.Fatal("Planner returned while dispatched Worker cleanup was still blocked")
			default:
			}
			releaseWorker()
			migrationBoundaryJoin(t, ctx, finished)
			migrationWait(t, ctx, workers.joined)
			returned := <-done
			// Both supported v1.6 and v2.6 refuse the required failDispatch
			// RecordPlan on the cancelled caller context. That first fatal
			// write error retains its documented plan-persistence class.
			assertAppendProbeFailure(t, returned.result, returned.err, "planner_plan_persistence_failed", true)
			if !errors.Is(returned.err, context.Canceled) {
				t.Fatal("required dispatch write failure lost its genuine cancellation cause")
			}
			assertMigrationBoundaryRecovery(t, ctx, factory, invocation, store, "planner_plan_persistence_failed", true)
			if workers.calls.Load() != 1 || workers.active.Load() != 0 || llm.callCount() != 2 {
				t.Fatal("Worker child survived return or semantic work was repeated")
			}
		})
	}
}

func TestPostgresModeledPlannerRecoveryDoesNotResumeSDK(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		for _, state := range []string{"completed-success", "running-noresume"} {
			t.Run(profile+"/"+state, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
				defer cancel()
				pool := isolatedStreamlinePool(t, ctx)
				store := newMigrationBoundaryStore(t, ctx, pool)
				sessions := migrationBoundarySession(t, store)
				steps, _, _ := appendProbeSteps(profile, "finish", "succeeded")
				llm, workers := &scriptedModel{steps: steps}, appendProbeWorkers()
				factory := migrationBoundaryFactory(t, profile, sessions, workers, llm)
				adkCalls := &migrationCountingADK{ADKSessionFactory: factory.adkSessions}
				factory.adkSessions = adkCalls
				invocation := appendProbeInvocation(profile)
				if state == "running-noresume" {
					start, err := sessions.Begin(ctx, invocation.StageExecutionID)
					if err != nil || !start.Invoke {
						t.Fatal("could not create the unfinished real PostgreSQL session")
					}
				} else {
					first, err := factory.Create(invocation)
					if err != nil {
						t.Fatal(err)
					}
					result, err := first.Run(ctx)
					if err != nil || result.Outcome != stageresult.StageSucceeded || llm.callCount() != 3 || len(workers.bindingCalls()) != 1 {
						t.Fatal("actual Runner did not establish a successful completion receipt")
					}
				}
				beforeModels, beforeWorkers, beforeADK := llm.callCount(), len(workers.bindingCalls()), adkCalls.calls.Load()
				store.returned.Store(true)
				recovered, err := factory.Create(invocation)
				if err != nil {
					t.Fatal(err)
				}
				result, err := recovered.Run(ctx)
				if state == "running-noresume" {
					assertAppendProbeFailure(t, result, err, "planner_invocation_in_progress", true)
					if !errors.Is(err, planner.ErrInvocationInProgress) {
						t.Fatal("unfinished session lost its no-resume cause")
					}
				} else if err != nil || result.Outcome != stageresult.StageSucceeded || result.Summary != "Completed bounded work" {
					t.Fatal("completed result did not recover exactly")
				}
				if llm.callCount() != beforeModels || len(workers.bindingCalls()) != beforeWorkers || adkCalls.calls.Load() != beforeADK || store.late.Load() != 0 {
					t.Fatal("recovery invoked SDK semantics or changed a durable receipt")
				}
			})
		}
	}
}

type migrationCountingADK struct {
	ADKSessionFactory
	calls atomic.Int64
}

func (s *migrationCountingADK) NewADKSession(ctx context.Context, identity planner.SessionIdentity, options plannersession.ADKOptions) (adksession.Service, error) {
	s.calls.Add(1)
	return s.ADKSessionFactory.NewADKSession(ctx, identity, options)
}

type migrationBoundaryStore struct {
	*runstore.PostgresStore
	ordinary      atomic.Int64
	returned      atomic.Bool
	late          atomic.Int64
	afterOrdinary func(context.Context, runstore.AppendPlannerEventParams, error) error
}

func newMigrationBoundaryStore(t *testing.T, ctx context.Context, pool *pgxpool.Pool) *migrationBoundaryStore {
	t.Helper()
	store := runstore.NewPostgresStore(pool)
	createPreparingStreamlineStage(t, ctx, store)
	return &migrationBoundaryStore{PostgresStore: store}
}

func (s *migrationBoundaryStore) AppendPlannerEvent(ctx context.Context, params runstore.AppendPlannerEventParams) error {
	if s.returned.Load() {
		s.late.Add(1)
	}
	var data struct {
		Activity struct {
			Author string `json:"author"`
		} `json:"activity"`
	}
	if err := json.Unmarshal(params.RunEvent.Data, &data); err != nil {
		return err
	}
	ordinary := params.RunEvent.Kind == runstore.RunEventPlannerActivity && (data.Activity.Author == "streamline_planner" || data.Activity.Author == "router_planner")
	if ordinary {
		s.ordinary.Add(1)
	}
	err := s.PostgresStore.AppendPlannerEvent(ctx, params)
	if ordinary && s.afterOrdinary != nil {
		return s.afterOrdinary(ctx, params, err)
	}
	return err
}

func migrationBoundarySession(t *testing.T, store plannersession.Store) *plannersession.Service {
	t.Helper()
	sessions, err := plannersession.New(store, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return sessions
}

func migrationBoundaryFactory(t *testing.T, profile string, sessions *plannersession.Service, workers planner.WorkerInvoker, llm model.LLM) *Factory {
	t.Helper()
	create := NewConfiguredFactory
	if profile == "router" {
		create = NewConfiguredRouterDelegate
	}
	factory, err := create(sessions, sessions, workers, &fakeInspector{}, unavailableWorkerStateReader{}, testModelFactory(llm), Limits{MaxWallTime: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	return factory
}

type migrationBoundaryResult struct {
	result stageresult.StageContentResult
	err    error
}

func startMigrationBoundaryRun(instance planner.Planner, ctx context.Context, store *migrationBoundaryStore) (<-chan struct{}, <-chan migrationBoundaryResult) {
	finished, done := make(chan struct{}), make(chan migrationBoundaryResult, 1)
	go func() {
		defer close(finished)
		result, err := instance.Run(ctx)
		store.returned.Store(true)
		done <- migrationBoundaryResult{result: result, err: err}
	}()
	return finished, done
}

func migrationBoundaryJoin(t *testing.T, ctx context.Context, done <-chan struct{}) {
	t.Helper()
	// Fixture cleanup runs after the test body's cancel defer. Its join gets
	// a separate hang guard, just as production completion owns its budget.
	joinContext, stop := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer stop()
	select {
	case <-done:
	case <-joinContext.Done():
		t.Error("Planner owner did not join before fixture teardown")
	}
}

func waitMigrationBoundarySQL(t *testing.T, ctx context.Context, pool *pgxpool.Pool, holderPID int, finished <-chan struct{}) {
	t.Helper()
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		var blocked int
		if err := pool.QueryRow(ctx, `SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND $1 = ANY(pg_blocking_pids(pid))`, holderPID).Scan(&blocked); err != nil {
			t.Fatal(err)
		}
		if blocked == 1 {
			return
		}
		select {
		case <-finished:
			t.Fatal("Planner returned before its event SQL reached the lock barrier")
		case <-ctx.Done():
			t.Fatal("event SQL never reached the real PostgreSQL lock barrier")
		case <-ticker.C:
		}
	}
}

func assertMigrationBoundaryJournal(t *testing.T, ctx context.Context, pool *pgxpool.Pool, profile string, wantOrdinary int) {
	t.Helper()
	var ordinary, failed, completed int
	if err := pool.QueryRow(ctx, `SELECT count(*) FILTER (WHERE kind = 'planner.activity' AND data #>> '{activity,author}' = $1), count(*) FILTER (WHERE kind = 'planner.failed'), count(*) FILTER (WHERE kind = 'planner.completed') FROM workflow_run_events WHERE run_id = 'run-1'`, profile+"_planner").Scan(&ordinary, &failed, &completed); err != nil {
		t.Fatal(err)
	}
	if ordinary != wantOrdinary || failed != 1 || completed != 0 {
		t.Fatalf("journal ordinary/failure/candidate=%d/%d/%d, want%d/1/0", ordinary, failed, completed, wantOrdinary)
	}
	var contiguous bool
	if err := pool.QueryRow(ctx, `SELECT max(sequence_number) = count(*) FROM workflow_run_events WHERE run_id = 'run-1'`).Scan(&contiguous); err != nil || !contiguous {
		t.Fatal("Run event sequence was consumed without a committed event")
	}
}

func assertMigrationBoundaryRecovery(t *testing.T, ctx context.Context, factory *Factory, invocation planner.Invocation, store *migrationBoundaryStore, code string, retryable bool) {
	t.Helper()
	stored := loadAppendProbeCompletion(t, ctx, store.PostgresStore)
	if stored.Failure == nil || stored.Failure.Code != code || stored.Failure.Retryable != retryable || stored.Result != nil {
		t.Fatal("independent completion did not durably settle the stopped invocation")
	}
	recovered, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	result, err := recovered.Run(ctx)
	assertAppendProbeFailure(t, result, err, code, retryable)
	if store.late.Load() != 0 {
		t.Fatal("a write escaped Planner return or completed recovery appended again")
	}
}

type migrationJoiningWorker struct {
	started, cancelling, release, joined chan struct{}
	calls, active                        atomic.Int64
	deadline, requestDeadline            time.Time
}

func (w *migrationJoiningWorker) Invoke(ctx context.Context, _ string, _ contracts.WorkerHandle, request contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
	if w.calls.Add(1) != 1 {
		return contracts.WorkerCompletion{}, errors.New("unexpected Worker retry")
	}
	w.deadline, _ = ctx.Deadline()
	w.requestDeadline = request.Deadline
	w.active.Add(1)
	go func() {
		close(w.started)
		<-ctx.Done()
		close(w.cancelling)
		<-w.release
		w.active.Add(-1)
		close(w.joined)
	}()
	<-w.joined
	return contracts.WorkerCompletion{}, planner.NewError("planner_cancelled", "Planner invocation was cancelled", true, ctx.Err())
}
