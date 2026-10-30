package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/adk/v2/model"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// The real PostgreSQL guard rejects a required event, while the transparent
// SessionService forwarder only observes that rejection. In particular, the
// guard leaves both finish_requested and completion/failure writes available.
func TestPostgresModeledPlannerAppendFailurePrecedence(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		for _, phase := range []string{"ordinary", "finish"} {
			outcomes := []string{"none"}
			if phase == "finish" {
				outcomes = []string{"succeeded", "failed-retryable", "failed-final"}
			}
			for _, outcome := range outcomes {
				for _, fault := range []string{"invalid", "unavailable"} {
					t.Run(profile+"/"+phase+"/"+outcome+"/"+fault, func(t *testing.T) {
						ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
						defer cancel()
						pool := isolatedStreamlinePool(t, ctx)
						store := runstore.NewPostgresStore(pool)
						createPreparingStreamlineStage(t, ctx, store)
						installRequiredEventGuard(t, ctx, pool, phase, fault)
						service := newAppendProbeSessions(t, store, phase)
						steps, wantModels, wantWorkers := appendProbeSteps(profile, phase, outcome)
						llm := &scriptedModel{steps: steps}
						workers := appendProbeWorkers()
						invocation := appendProbeInvocation(profile)
						factory := appendProbeFactory(t, profile, service, workers, llm)
						instance, err := factory.Create(invocation)
						if err != nil {
							t.Fatal(err)
						}
						result, runErr := instance.Run(ctx)
						observed := service.observation()
						if observed.matches != 1 || observed.err == nil {
							t.Fatalf("required event did not reach a real PostgreSQL rejection: matches=%d failed=%t", observed.matches, observed.err != nil)
						}
						var postgresError *pgconn.PgError
						if !errors.As(observed.err, &postgresError) {
							t.Fatal("observed append error does not retain the PostgreSQL cause")
						}
						wantClass := persistencepostgres.ClassUnavailable
						wantCode, wantRetryable := "planner_session_unavailable", outcome != "failed-final"
						if fault == "invalid" {
							wantClass = persistencepostgres.ClassInvalid
							wantCode, wantRetryable = "planner_session_rejected", false
							if !errors.Is(observed.err, runstore.ErrInvalid) || postgresError.Code != persistencepostgres.SQLStateCheckViolation || postgresError.ConstraintName != "v419_required_event_rejected" {
								t.Fatal("real CHECK rejection lost its owning ErrInvalid or constraint")
							}
						} else if postgresError.Code != persistencepostgres.SQLStateConnectionFailure {
							t.Fatal("observed PostgreSQL outage was not the injected condition")
						}
						if persistencepostgres.Classify(observed.err) != wantClass {
							t.Fatalf("observed PostgreSQL class=%d, want %d", persistencepostgres.Classify(observed.err), wantClass)
						}
						if llm.callCount() != wantModels || len(workers.bindingCalls()) != wantWorkers {
							t.Errorf("model/Worker calls=%d/%d, want %d/%d", llm.callCount(), len(workers.bindingCalls()), wantModels, wantWorkers)
						}
						assertAppendProbeFailure(t, result, runErr, wantCode, wantRetryable)
						if fault == "invalid" && !errors.Is(runErr, runstore.ErrInvalid) {
							t.Error("returned rejection does not preserve errors.Is(runstore.ErrInvalid)")
						}
						assertAppendProbeReport(t, instance, wantCode, wantRetryable, phase == "finish")
						stored := loadAppendProbeCompletion(t, ctx, store)
						if stored.Failure == nil || stored.Failure.Code != wantCode || stored.Failure.Retryable != wantRetryable || stored.Result != nil {
							t.Errorf("stored completion code=%s result=%t, want %s failure retryable=%t", appendProbeCompletionCode(stored), stored.Result != nil, wantCode, wantRetryable)
						}
						assertAppendProbeJournal(t, ctx, pool, phase)
						// Replay uses the real completed-session receipt, never SDK resume.
						recovered, err := factory.Create(invocation)
						if err != nil {
							t.Fatal(err)
						}
						replayed, replayErr := recovered.Run(ctx)
						assertAppendProbeFailure(t, replayed, replayErr, wantCode, wantRetryable)
						if llm.callCount() != wantModels || len(workers.bindingCalls()) != wantWorkers {
							t.Fatal("completed recovery repeated a model or Worker effect")
						}
						t.Logf("observed=postgres class=%d appendMatches=%d returnedCode=%s returnedOutcome=%s storedCode=%s models=%d workers=%d", wantClass, observed.matches, appendProbeErrorCode(runErr), result.Outcome, appendProbeCompletionCode(stored), llm.callCount(), len(workers.bindingCalls()))
					})
				}
			}
		}
	}
}

func installRequiredEventGuard(t *testing.T, ctx context.Context, pool *pgxpool.Pool, phase, fault string) {
	t.Helper()
	prefix := ""
	if fault == "unavailable" {
		prefix = "NEW."
	}
	predicate := fmt.Sprintf("%skind = 'planner.activity' AND %sdata #> '{activity,functionResults}' ? 'finish'", prefix, prefix)
	if phase == "ordinary" {
		predicate = fmt.Sprintf("%skind = 'planner.activity' AND %sdata #>> '{activity,author}' IN ('streamline_planner', 'router_planner')", prefix, prefix)
	}
	if fault == "invalid" {
		if _, err := pool.Exec(ctx, "ALTER TABLE workflow_run_events ADD CONSTRAINT v419_required_event_rejected CHECK (NOT ("+predicate+"))"); err != nil {
			t.Fatal(err)
		}
		return
	}
	// This is a PostgreSQL-returned, injected connection-failure condition. It
	// does not disconnect or stop a shared PostgreSQL service.
	statement := fmt.Sprintf(`CREATE FUNCTION v419_required_event_outage() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF %s THEN
        RAISE EXCEPTION 'controlled required event outage' USING ERRCODE = '%s';
    END IF;
    RETURN NEW;
END $$`, predicate, persistencepostgres.SQLStateConnectionFailure)
	if _, err := pool.Exec(ctx, statement); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `CREATE TRIGGER v419_required_event_outage BEFORE INSERT ON workflow_run_events FOR EACH ROW EXECUTE FUNCTION v419_required_event_outage()`); err != nil {
		t.Fatal(err)
	}
}

type appendProbeObservation struct {
	matches int
	err     error
}

type appendProbeSessions struct {
	*plannersession.Service
	mu             sync.Mutex
	phase          string
	observed       appendProbeObservation
	beforeComplete func(context.Context)
	afterAppend    func(error)
}

func newAppendProbeSessions(t *testing.T, store *runstore.PostgresStore, phase string) *appendProbeSessions {
	t.Helper()
	service, err := plannersession.New(store, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return &appendProbeSessions{Service: service, phase: phase}
}

func (s *appendProbeSessions) NewADKSession(ctx context.Context, identity planner.SessionIdentity, options plannersession.ADKOptions) (adksession.Service, error) {
	service, err := s.Service.NewADKSession(ctx, identity, options)
	if err != nil {
		return nil, err
	}
	return &appendProbeADK{Service: service, owner: s}, nil
}

func (s *appendProbeSessions) Complete(ctx context.Context, identity planner.SessionIdentity, completion planner.Completion) error {
	if s.beforeComplete != nil {
		s.beforeComplete(ctx)
	}
	return s.Service.Complete(ctx, identity, completion)
}

func (s *appendProbeSessions) observation() appendProbeObservation {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.observed
}

type appendProbeADK struct {
	adksession.Service
	owner *appendProbeSessions
}

func (s *appendProbeADK) AppendEvent(ctx context.Context, current adksession.Session, event *adksession.Event) error {
	matched := false
	if event != nil && !event.Partial && event.Content != nil {
		for _, part := range event.Content.Parts {
			if part == nil {
				continue
			}
			if s.owner.phase == "finish" && part.FunctionResponse != nil && part.FunctionResponse.Name == finishToolName {
				matched = true
			}
			if s.owner.phase == "ordinary" && event.Content.Role == string(genai.RoleModel) && part.Text != "" {
				matched = true
			}
		}
	}
	err := s.Service.AppendEvent(ctx, current, event)
	if matched {
		s.owner.mu.Lock()
		s.owner.observed.matches++
		s.owner.observed.err = err
		s.owner.mu.Unlock()
		if s.owner.afterAppend != nil {
			s.owner.afterAppend(err)
		}
	}
	return err
}

func appendProbeInvocation(profile string) planner.Invocation {
	invocation := testInvocation("builder")
	invocation.Stage.Planner = resolved.PlannerRef{PlannerID: profile, Version: "1"}
	invocation.Stage.Result.Artifacts = map[string]resolved.ArtifactSlot{}
	invocation.ModelAccess.ModelPolicy.MaxModelCalls = 4
	return invocation
}

func appendProbeFactory(t *testing.T, profile string, sessions *appendProbeSessions, workers *fakeWorkerInvoker, llm model.LLM) *Factory {
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

func appendProbeWorkers() *fakeWorkerInvoker {
	return &fakeWorkerInvoker{results: map[string]stageresult.StageContentResult{
		"builder": stageResult("Completed bounded work", map[string]contracts.ArtifactRef{}),
	}}
}

func appendProbeSteps(profile, phase, outcome string) ([]modelStep, int, int) {
	if phase == "ordinary" {
		return []modelStep{textStep("Continue bounded planning")}, 1, 0
	}
	finish := map[string]any{"outcome": "failed", "summary": "Required evidence unavailable", "artifacts": map[string]any{}, "error": map[string]any{
		"code": "insufficient_evidence", "message": "Required evidence unavailable", "retryable": outcome == "failed-retryable",
	}}
	if outcome != "succeeded" {
		return []modelStep{functionStep(finishToolName, finish)}, 1, 0
	}
	finish["outcome"], finish["summary"] = "succeeded", "Completed bounded work"
	delete(finish, "error")
	execute := map[string]any{"subtask_id": "0"}
	if profile == "router" {
		execute["worker_name"] = "builder"
	}
	return []modelStep{addSubtaskStep("Analyze bounded input", "Complete bounded work"), functionStep(executeCurrentSubtaskToolName, execute), functionStep(finishToolName, finish)}, 3, 1
}

func assertAppendProbeFailure(t *testing.T, result stageresult.StageContentResult, err error, code string, retryable bool) {
	t.Helper()
	var failure *planner.Error
	if !errors.As(err, &failure) || failure.Code != code || failure.Retryable != retryable || result.Outcome != "" {
		t.Errorf("required append outcome: code=%s resultOutcome=%s, want code=%s retryable=%t and no candidate", appendProbeErrorCode(err), result.Outcome, code, retryable)
	}
}

func assertAppendProbeReport(t *testing.T, instance planner.Planner, code string, retryable, finished bool) {
	t.Helper()
	report, ok := instance.(planner.ReportProvider).ExecutionReport()
	if !ok || len(report.Errors) == 0 {
		t.Errorf("required write failure is absent from report: reportPresent=%t errors=%d", ok, len(report.Errors))
	} else if last := report.Errors[len(report.Errors)-1]; last.Code != code || last.Retryable == nil || *last.Retryable != retryable {
		t.Errorf("last report code=%s, want %s retryable=%t", last.Code, code, retryable)
	}
	if finishCalled(report) != finished {
		t.Errorf("accepted finish metric=%t, want %t", finishCalled(report), finished)
	}
}

func appendProbeErrorCode(err error) string {
	if err == nil {
		return "none"
	}
	return planner.FailureFrom(err).Code
}

func appendProbeCompletionCode(completion planner.Completion) string {
	if completion.Failure != nil {
		return completion.Failure.Code
	}
	if completion.Result != nil {
		return "candidate:" + string(completion.Result.Outcome)
	}
	return "absent"
}

func loadAppendProbeCompletion(t *testing.T, ctx context.Context, store *runstore.PostgresStore) planner.Completion {
	t.Helper()
	execution, err := store.GetStageExecution(ctx, "stage-1")
	if err != nil || execution.PlannerSessionID == nil {
		t.Fatal("test Stage has no Planner session")
	}
	session, err := store.GetPlannerSession(ctx, *execution.PlannerSessionID)
	if err != nil {
		t.Fatal(err)
	}
	var value struct {
		Status     string              `json:"status"`
		Completion *planner.Completion `json:"completion"`
	}
	if err := json.Unmarshal(session.State, &value); err != nil || value.Status != "completed" || value.Completion == nil {
		t.Fatalf("session did not settle: decoded=%t status=%s completion=%t", err == nil, value.Status, value.Completion != nil)
	}
	return *value.Completion
}

func assertAppendProbeJournal(t *testing.T, ctx context.Context, pool *pgxpool.Pool, phase string) {
	t.Helper()
	var finish, response, candidates, failures int
	if err := pool.QueryRow(ctx, `SELECT
 count(*) FILTER (WHERE kind = 'planner.finish_requested'),
 count(*) FILTER (WHERE kind = 'planner.activity' AND data #> '{activity,functionResults}' ? 'finish'),
 count(*) FILTER (WHERE kind = 'planner.completed'), count(*) FILTER (WHERE kind = 'planner.failed')
 FROM workflow_run_events`).Scan(&finish, &response, &candidates, &failures); err != nil {
		t.Fatal(err)
	}
	wantFinish := 0
	if phase == "finish" {
		wantFinish = 1
	}
	if finish != wantFinish || response != 0 || candidates != 0 || failures != 1 {
		t.Errorf("durable journal finish/request-response/candidate/failure=%d/%d/%d/%d, want %d/0/0/1", finish, response, candidates, failures, wantFinish)
	}
}

// Cancellation before finish and cancellation during independent completion
// are deliberately separate controls; a persistence fix must not merge them.
func TestModeledPlannerCancellationKeepsCompletionOwnership(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		for _, phase := range []string{"before-finish", "completion-write"} {
			t.Run(profile+"/"+phase, func(t *testing.T) {
				ctx, stop := context.WithTimeout(t.Context(), 45*time.Second)
				defer stop()
				pool := isolatedStreamlinePool(t, ctx)
				store := runstore.NewPostgresStore(pool)
				createPreparingStreamlineStage(t, ctx, store)
				sessions := newAppendProbeSessions(t, store, "finish")
				started, release := make(chan struct{}), make(chan struct{})
				var releaseOnce sync.Once
				allowComplete := func() { releaseOnce.Do(func() { close(release) }) }
				invocation := appendProbeInvocation(profile)
				workers := appendProbeWorkers()
				var llm model.LLM
				var modelCalls func() int
				var completeContextErr error
				if phase == "before-finish" {
					cancelModel := &appendProbeCancellationModel{started: started}
					llm = cancelModel
					modelCalls = func() int { return int(cancelModel.calls.Load()) }
				} else {
					steps, _, _ := appendProbeSteps(profile, "finish", "failed-final")
					scripted := &scriptedModel{steps: steps}
					llm, modelCalls = scripted, scripted.callCount
					sessions.beforeComplete = func(recordContext context.Context) {
						close(started)
						<-release
						completeContextErr = recordContext.Err()
					}
				}
				factory := appendProbeFactory(t, profile, sessions, workers, llm)
				instance, err := factory.Create(invocation)
				if err != nil {
					t.Fatal(err)
				}
				runContext, cancel := context.WithCancel(ctx)
				type runResult struct {
					candidate stageresult.StageContentResult
					err       error
				}
				done := make(chan runResult, 1)
				go func() {
					defer close(done)
					candidate, err := instance.Run(runContext)
					done <- runResult{candidate, err}
				}()
				defer func() {
					cancel()
					allowComplete()
					select {
					case <-done:
					case <-time.After(5 * time.Second):
						t.Error("test owner could not join Planner iteration")
					}
				}()
				select {
				case <-started:
				case <-ctx.Done():
					t.Fatal("deterministic cancellation boundary was not reached")
				}
				if phase == "completion-write" {
					observed := sessions.observation()
					if observed.matches != 1 || observed.err != nil {
						t.Fatal("completion barrier preceded the successful real final FunctionResponse append")
					}
				}
				cancel()
				allowComplete()
				var returned runResult
				select {
				case returned = <-done:
				case <-ctx.Done():
					t.Fatal("cancelled Planner failed to settle its independent completion")
				}
				stored := loadAppendProbeCompletion(t, ctx, store)
				if phase == "before-finish" {
					assertAppendProbeFailure(t, returned.candidate, returned.err, "planner_cancelled", true)
					if stored.Failure == nil || stored.Failure.Code != "planner_cancelled" {
						t.Error("caller cancellation was not durably recorded as cancellation")
					}
				} else if returned.err != nil || returned.candidate.Outcome != stageresult.StageFailed || stored.Result == nil || stored.Failure != nil || completeContextErr != nil {
					t.Errorf("independent completion lost its authority: code=%s outcome=%s stored=%s writeContextCancelled=%t", appendProbeErrorCode(returned.err), returned.candidate.Outcome, appendProbeCompletionCode(stored), completeContextErr != nil)
				}
				if modelCalls() != 1 || len(workers.bindingCalls()) != 0 {
					t.Fatalf("cancellation control model/Worker calls=%d/%d, want1/0", modelCalls(), len(workers.bindingCalls()))
				}
			})
		}
	}
}

// The stop is admitted only after the production delegate returned its real
// PostgreSQL error. It cannot erase a known final candidate's retryability or
// turn that store failure into an unobserved/mocked persistence outcome.
func TestPostgresModeledPlannerFinalOutcomeSurvivesAppendCancellation(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		for _, fault := range []string{"invalid", "unavailable"} {
			t.Run(profile+"/"+fault, func(t *testing.T) {
				ctx, stop := context.WithTimeout(t.Context(), 45*time.Second)
				defer stop()
				pool := isolatedStreamlinePool(t, ctx)
				store := runstore.NewPostgresStore(pool)
				createPreparingStreamlineStage(t, ctx, store)
				installRequiredEventGuard(t, ctx, pool, "finish", fault)
				sessions := newAppendProbeSessions(t, store, "finish")
				failedAppend, release := make(chan struct{}), make(chan struct{})
				var releaseOnce sync.Once
				allowAppendReturn := func() { releaseOnce.Do(func() { close(release) }) }
				sessions.afterAppend = func(error) { close(failedAppend); <-release }
				steps, _, _ := appendProbeSteps(profile, "finish", "failed-final")
				llm, workers := &scriptedModel{steps: steps}, appendProbeWorkers()
				factory := appendProbeFactory(t, profile, sessions, workers, llm)
				invocation := appendProbeInvocation(profile)
				instance, err := factory.Create(invocation)
				if err != nil {
					t.Fatal(err)
				}
				runContext, cancel := context.WithCancel(ctx)
				type runResult struct {
					candidate stageresult.StageContentResult
					err       error
				}
				done := make(chan runResult, 1)
				go func() {
					defer close(done)
					candidate, err := instance.Run(runContext)
					done <- runResult{candidate, err}
				}()
				defer func() {
					cancel()
					allowAppendReturn()
					select {
					case <-done:
					case <-time.After(5 * time.Second):
						t.Error("test owner could not join the stopped Planner")
					}
				}()
				select {
				case <-failedAppend:
				case <-ctx.Done():
					t.Fatal("real PostgreSQL append-failure barrier was not reached")
				}
				observed := sessions.observation()
				var storeCause *pgconn.PgError
				if observed.matches != 1 || !errors.As(observed.err, &storeCause) {
					t.Fatal("stop did not follow exactly one real PostgreSQL failure")
				}
				wantClass := persistencepostgres.ClassUnavailable
				if fault == "invalid" {
					wantClass = persistencepostgres.ClassInvalid
				}
				if persistencepostgres.Classify(observed.err) != wantClass {
					t.Fatal("stop barrier followed a different storage outcome")
				}
				cancel()
				allowAppendReturn()
				var returned runResult
				select {
				case returned = <-done:
				case <-ctx.Done():
					t.Fatal("stopped Planner failed to settle its session")
				}
				assertAppendProbeFailure(t, returned.candidate, returned.err, "planner_cancelled", false)
				var returnedStoreCause *pgconn.PgError
				if !errors.Is(returned.err, context.Canceled) || !errors.As(returned.err, &returnedStoreCause) || returnedStoreCause != storeCause {
					t.Error("cancelled result lost the original PostgreSQL failure or caller cancellation cause")
				}
				if fault == "invalid" && !errors.Is(returned.err, runstore.ErrInvalid) {
					t.Error("cancelled result lost the definite store rejection")
				}
				assertAppendProbeReport(t, instance, "planner_cancelled", false, true)
				stored := loadAppendProbeCompletion(t, ctx, store)
				if stored.Result != nil || stored.Failure == nil || stored.Failure.Code != "planner_cancelled" || stored.Failure.Retryable {
					t.Errorf("stored stop erased the final outcome: code=%s", appendProbeCompletionCode(stored))
				}
				assertAppendProbeJournal(t, ctx, pool, "finish")
				recovered, err := factory.Create(invocation)
				if err != nil {
					t.Fatal(err)
				}
				candidate, replayErr := recovered.Run(ctx)
				assertAppendProbeFailure(t, candidate, replayErr, "planner_cancelled", false)
				if llm.callCount() != 1 || len(workers.bindingCalls()) != 0 {
					t.Fatal("stop/recovery repeated a model or Worker effect")
				}
				t.Logf("observed=postgres class=%d appendMatches=1 callerCancelled=true returnedCode=%s returnedRetryable=%t storedRetryable=%t postgresCausePreserved=%t models=1 workers=0", wantClass, appendProbeErrorCode(returned.err), planner.FailureFrom(returned.err).Retryable, stored.Failure != nil && stored.Failure.Retryable, returnedStoreCause == storeCause)
			})
		}
	}
}

type appendProbeCancellationModel struct {
	started chan struct{}
	calls   atomic.Int64
}

func (*appendProbeCancellationModel) Name() string { return "fake-streamline-model" }

func (m *appendProbeCancellationModel) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.calls.Add(1) != 1 {
			yield(nil, errors.New("unexpected cancelled model retry"))
			return
		}
		close(m.started)
		<-ctx.Done()
		yield(nil, ctx.Err())
	}
}
