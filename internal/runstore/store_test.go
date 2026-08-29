package runstore

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestWorkflowRunTransitionValidation(t *testing.T) {
	t.Parallel()

	valid := [][2]WorkflowRunState{
		{RunInitializing, RunRunning}, {RunInitializing, RunFailed},
		{RunRunning, RunSucceeded}, {RunRunning, RunFailed},
		{RunCancelling, RunCancelled},
	}
	for _, transition := range valid {
		if err := validateRunTransition(transition[0], transition[1]); err != nil {
			t.Errorf("valid transition %s -> %s: %v", transition[0], transition[1], err)
		}
	}
	for _, transition := range [][2]WorkflowRunState{
		{RunInitializing, RunCancelling}, {RunRunning, RunCancelling},
		{RunRunning, RunCancelled}, {RunSucceeded, RunRunning},
		{RunCancelling, RunSucceeded}, {"unknown", RunRunning},
	} {
		if err := validateRunTransition(transition[0], transition[1]); !errors.Is(err, ErrInvalid) {
			t.Errorf("invalid transition %s -> %s error = %v", transition[0], transition[1], err)
		}
	}
}

func TestStageContextRequiresExactPresentArtifacts(t *testing.T) {
	t.Parallel()

	revision := "rev-1"
	valid := StageContextSnapshot{Artifacts: map[string]PinnedContextArtifact{
		"present": {Required: true, Artifact: &contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &revision}},
		"absent":  {Required: false},
	}}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid StageContext: %v", err)
	}
	versionless := valid
	versionless.Artifacts = map[string]PinnedContextArtifact{
		"bad": {Artifact: &contracts.ArtifactRef{Namespace: "inputs", Name: "source"}},
	}
	if err := versionless.Validate(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("versionless artifact error = %v", err)
	}
}

func TestStageTerminationValidation(t *testing.T) {
	t.Parallel()

	valid := StageTermination{
		Outcome: TerminationInterrupted, Code: "worker_lost", Message: "worker disappeared",
		Retryable: true, Phase: TerminationRunning, OccurredAt: time.Now(),
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid StageTermination: %v", err)
	}
	valid.Phase = "unknown"
	if err := valid.Validate(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid StageTermination error = %v", err)
	}
}

func TestWorkflowRunCancellationValidation(t *testing.T) {
	t.Parallel()

	requestedBy := "user-1"
	reason := "no longer needed"
	valid := WorkflowRunCancellation{
		Code: CancellationUserRequested, RequestedAt: time.Now(),
		RequestedBy: &requestedBy, Reason: &reason,
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid WorkflowRunCancellation: %v", err)
	}

	invalid := []WorkflowRunCancellation{
		{},
		{Code: "deadline", RequestedAt: time.Now()},
		{Code: CancellationUserRequested},
		{Code: CancellationUserRequested, RequestedAt: time.Now(), Reason: func() *string { value := " "; return &value }()},
	}
	for _, cancellation := range invalid {
		if err := cancellation.Validate(); !errors.Is(err, ErrInvalid) {
			t.Errorf("invalid cancellation %+v error = %v", cancellation, err)
		}
	}
}

func TestStageExecutionInputValidation(t *testing.T) {
	t.Parallel()

	valid := CreateStageExecutionParams{
		StageExecutionID: "stage-1", RunID: "run-1", StageName: "build", Attempt: 1,
		StageSpecSchemaVersion: "v1", StageSpecSnapshot: []byte(`{"objective":"build"}`),
		StageContextSchemaVersion: "v1", StageContext: StageContextSnapshot{},
	}
	if err := validateCreateStageExecution(valid); err != nil {
		t.Fatalf("valid StageExecution input: %v", err)
	}
	previous := "stage-1"
	valid.Attempt = 2
	valid.PreviousExecutionID = nil
	if err := validateCreateStageExecution(valid); !errors.Is(err, ErrInvalid) {
		t.Fatalf("missing retry lineage error = %v", err)
	}
	valid.PreviousExecutionID = &previous
	if err := validateCreateStageExecution(valid); err != nil {
		t.Fatalf("valid retry lineage: %v", err)
	}
}

func TestTypedConflictAndNoWorkResults(t *testing.T) {
	t.Parallel()

	store := NewPostgresStore(stubDB{row: stubRow{err: pgx.ErrNoRows}})
	_, err := store.TransitionRun(
		context.Background(), "run-1", RunInitializing, RunRunning, Reason{Code: "ready"},
	)
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("TransitionRun error = %v, want ErrConflict", err)
	}
	var conflict *StateConflictError
	if !errors.As(err, &conflict) || conflict.Expected != string(RunInitializing) {
		t.Fatalf("TransitionRun conflict = %#v", conflict)
	}
	_, err = store.ClaimRunnableRun(context.Background(), "claim-1", time.Second)
	if !errors.Is(err, ErrNoWork) {
		t.Fatalf("ClaimRunnableRun error = %v, want ErrNoWork", err)
	}
}

func TestStorePreservesContextCancellation(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	store := NewPostgresStore(stubDB{row: stubRow{err: ctx.Err()}})
	_, err := store.GetRun(ctx, "run-1")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("GetRun error = %v, want context.Canceled", err)
	}
}

type stubDB struct {
	row pgx.Row
}

func (s stubDB) Exec(context.Context, string, ...any) (pgconn.CommandTag, error) {
	return pgconn.CommandTag{}, errors.New("unexpected Exec")
}

func (s stubDB) Query(context.Context, string, ...any) (pgx.Rows, error) {
	return nil, errors.New("unexpected Query")
}

func (s stubDB) QueryRow(context.Context, string, ...any) pgx.Row { return s.row }

type stubRow struct{ err error }

func (r stubRow) Scan(...any) error { return r.err }
