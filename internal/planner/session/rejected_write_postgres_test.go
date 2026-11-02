package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// A session write that the planner_session_state_size CHECK refuses reaches
// the Planner as runstore.ErrInvalid, so starting the session, recording the
// request and recording the completion each end with the non-retryable
// planner_session_rejected, never the retryable planner_session_unavailable,
// and the refused write changes nothing (PLN-R106). The bound is lowered
// below the state the write produces; the measure and the constraint are the
// production ones.
func TestPostgresRejectedSessionWritesAreNotRetryable(t *testing.T) {
	t.Run("start", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
		defer cancel()
		pool := isolatedPlannerPool(t, ctx)
		store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
		createPlannerStage(t, ctx, store)
		service := newRejectionService(t, store)
		if err := lowerPlannerStateBound(ctx, pool, 2); err != nil {
			t.Fatal(err)
		}
		spans := &recordingSpans{}
		_, err := planner.StartSession(ctx, service, spans, "stage-planner")
		assertSessionRejected(t, err)
		if len(spans.codes) != 1 || spans.codes[0] != "planner_session_rejected" {
			t.Fatalf("session span codes = %v", spans.codes)
		}
		execution, err := store.GetStageExecution(ctx, "stage-planner")
		if err != nil || execution.State != runstore.StagePreparing || execution.PlannerSessionID != nil {
			t.Fatalf("StageExecution after a rejected start = (%+v, %v), want preparing without a session", execution, err)
		}
	})

	t.Run("request", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
		defer cancel()
		pool := isolatedPlannerPool(t, ctx)
		store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
		createPlannerStage(t, ctx, store)
		service := newRejectionService(t, store)
		start, err := planner.StartSession(ctx, service, &recordingSpans{}, "stage-planner")
		if err != nil {
			t.Fatal(err)
		}
		before := readSessionRow(t, ctx, store, start.Identity.SessionID)
		size, err := storedPlannerStateBytes(ctx, pool)
		if err != nil {
			t.Fatal(err)
		}
		if err := lowerPlannerStateBound(ctx, pool, size); err != nil {
			t.Fatal(err)
		}
		spans := &recordingSpans{}
		err = planner.RecordSessionRequest(ctx, service, spans, start.Identity, planner.RequestFacts{
			Bindings: []string{"builder"}, ParameterNames: []string{"mode"},
			ObjectiveDigest:    "sha256:" + strings.Repeat("c", 64),
			InstructionsDigest: "sha256:" + strings.Repeat("d", 64),
			Artifacts:          map[string]contracts.ArtifactRef{},
		})
		assertSessionRejected(t, err)
		if len(spans.codes) != 1 || spans.codes[0] != "planner_session_rejected" {
			t.Fatalf("session span codes = %v", spans.codes)
		}
		if after := readSessionRow(t, ctx, store, start.Identity.SessionID); after != before {
			t.Fatalf("session after a rejected request = %+v, want %+v", after, before)
		}
	})

	t.Run("completion", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
		defer cancel()
		pool := isolatedPlannerPool(t, ctx)
		store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
		createPlannerStage(t, ctx, store)
		service := newRejectionService(t, store)
		// The Worker runs after the request is recorded; it lowers the bound
		// to that state, which the completion outgrows. Its failure is
		// retryable, so only the rejection makes the outcome non-retryable.
		worker := &boundLoweringWorker{pool: pool, failure: contracts.WorkerFailure{
			Code: "worker_budget_exhausted", Message: "Worker exhausted its budget", Retryable: true,
		}}
		factory, err := planner.NewPassthroughFactory(service, worker, noArtifacts{})
		if err != nil {
			t.Fatal(err)
		}
		instance, err := factory.Create(passthroughInvocation())
		if err != nil {
			t.Fatal(err)
		}
		result, err := instance.Run(ctx)
		assertSessionRejected(t, err)
		if result.Outcome != "" || worker.calls != 1 || worker.err != nil {
			t.Fatalf("Run result = %+v, Worker calls = %d, bound error = %v", result, worker.calls, worker.err)
		}
		execution, err := store.GetStageExecution(ctx, "stage-planner")
		if err != nil || execution.PlannerSessionID == nil {
			t.Fatalf("StageExecution = (%+v, %v)", execution, err)
		}
		session, err := store.GetPlannerSession(ctx, *execution.PlannerSessionID)
		if err != nil {
			t.Fatal(err)
		}
		state, err := decodeState(session.State)
		if err != nil || state.Status != statusRunning || state.Completion != nil || !state.RequestRecorded {
			t.Fatalf("Planner session state = %s (%v), want running without a completion", session.State, err)
		}
	})
}

// The session fake measures a state as PostgreSQL measures the stored
// jsonb text, so it refuses the same states as planner_session_state_size.
func TestPostgresSessionFakeMeasuresStoredJSONBText(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedPlannerPool(t, ctx)
	encoded, err := json.Marshal(persistentState{
		Status: statusRunning, NextSequence: 12, RequestRecorded: true, ADKEventCount: 3,
		RequestDigest: "sha256:" + strings.Repeat("a", 64), FactDigests: map[string]string{
			"selected:dispatch-0001": "sha256:0123", "finish:2:failed": "sha256:4567",
		},
		Completion: &planner.Completion{Failure: &planner.Failure{
			Code: "worker_failed", Message: `a "quoted", colon: and comma, message`, Retryable: true,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{
		`{}`, `[]`, `{"a":[0,0,0]}`, `{"a":{"b":[1,{"c":"d"}]},"e":"f, g: h"}`, `{"text":"a \"b\", c: d\\"}`,
		string(encoded),
	} {
		var stored int
		if err := pool.QueryRow(ctx, `SELECT octet_length($1::jsonb::text)`, value).Scan(&stored); err != nil {
			t.Fatal(err)
		}
		if measured := storedJSONBTextBytes(json.RawMessage(value)); measured != stored {
			t.Fatalf("fake measures %s as %d bytes, PostgreSQL stores %d", value, measured, stored)
		}
	}
}

func newRejectionService(t *testing.T, store *runstore.PostgresStore) *Service {
	t.Helper()
	service, err := New(store, Options{})
	if err != nil {
		t.Fatal(err)
	}
	return service
}

// lowerPlannerStateBound replaces planner_session_state_size with the same
// measure and bound limit. NOT VALID keeps the existing session row, so only
// later writes are checked against the lowered bound.
func lowerPlannerStateBound(ctx context.Context, pool *pgxpool.Pool, limit int) error {
	_, err := pool.Exec(ctx, fmt.Sprintf(`
ALTER TABLE planner_sessions DROP CONSTRAINT planner_session_state_size,
ADD CONSTRAINT planner_session_state_size CHECK (octet_length(state::text) <= %d) NOT VALID`, limit))
	return err
}

func storedPlannerStateBytes(ctx context.Context, pool *pgxpool.Pool) (int, error) {
	var size int
	err := pool.QueryRow(ctx, `SELECT octet_length(state::text) FROM planner_sessions`).Scan(&size)
	return size, err
}

type sessionRow struct {
	state    string
	sequence int64
}

func readSessionRow(t *testing.T, ctx context.Context, store *runstore.PostgresStore, sessionID string) sessionRow {
	t.Helper()
	session, err := store.GetPlannerSession(ctx, sessionID)
	if err != nil {
		t.Fatal(err)
	}
	return sessionRow{state: string(session.State), sequence: session.NextEventSequence}
}

func assertSessionRejected(t *testing.T, err error) {
	t.Helper()
	var plannerError *planner.Error
	if !errors.As(err, &plannerError) || plannerError.Code != "planner_session_rejected" ||
		plannerError.Retryable || !errors.Is(err, runstore.ErrInvalid) {
		t.Fatalf("error = %v (cause %v), want non-retryable planner_session_rejected wrapping runstore.ErrInvalid",
			err, errors.Unwrap(err))
	}
}

// boundLoweringWorker lowers the session state bound to the stored state
// before it returns its failure.
type boundLoweringWorker struct {
	pool    *pgxpool.Pool
	failure contracts.WorkerFailure
	calls   int
	err     error
}

func (w *boundLoweringWorker) Invoke(
	ctx context.Context, _ string, _ contracts.WorkerHandle, _ contracts.StageContentRequest,
) (contracts.WorkerCompletion, error) {
	w.calls++
	size, err := storedPlannerStateBytes(ctx, w.pool)
	if err == nil {
		err = lowerPlannerStateBound(ctx, w.pool, size)
	}
	w.err = err
	failure := w.failure
	return contracts.WorkerCompletion{
		APIVersion: contracts.APIVersion, Failure: &failure,
		InvocationID: "worker-invocation-failed", StateRevision: 1,
	}, nil
}
