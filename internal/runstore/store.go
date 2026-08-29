package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// Repository is the durable boundary used by the public API and Scheduler.
// PostgresStore implements it for both a pool and an explicit pgx transaction.
type Repository interface {
	CreateRun(context.Context, CreateRunParams) (WorkflowRun, error)
	GetRun(context.Context, string) (WorkflowRun, error)
	TransitionRun(context.Context, string, WorkflowRunState, WorkflowRunState, Reason) (WorkflowRun, error)
	RequestRunCancellation(context.Context, string, WorkflowRunCancellation) (WorkflowRun, error)
	ClaimRunnableRun(context.Context, string, time.Duration) (WorkflowRun, error)
	RenewRunClaim(context.Context, string, string, time.Duration) error
	ReleaseRunClaim(context.Context, string, string) error
	CreateStageExecution(context.Context, CreateStageExecutionParams) (StageExecution, error)
	GetStageExecution(context.Context, string) (StageExecution, error)
	ListStageExecutions(context.Context, string) ([]StageExecution, error)
	RecordStageTransitionDecision(context.Context, RecordStageTransitionDecisionParams) (StageTransitionDecision, error)
	GetStageTransitionDecision(context.Context, string) (StageTransitionDecision, error)
	ListStageTransitionDecisions(context.Context, string) ([]StageTransitionDecision, error)
	ListTerminalStageExecutionsWithAllocations(context.Context) ([]StageExecution, error)
	StartPlanner(context.Context, StartPlannerParams) error
	EnterFinalizing(context.Context, EnterFinalizingParams) error
	CompleteStageResult(context.Context, string, string, contracts.StageContentResult) error
	EnterAborting(context.Context, EnterAbortingParams) error
	CompleteStageTermination(context.Context, string) error
	AppendPlannerEvent(context.Context, AppendPlannerEventParams) error
	GetPlannerSession(context.Context, string) (PlannerSession, error)
	ListPlannerEvents(context.Context, string, int64) ([]PlannerEvent, error)
	RecordStageAllocation(context.Context, StageAllocation) error
	ListStageAllocations(context.Context, string) ([]StageAllocation, error)
	RecordStageExecutionReport(context.Context, RecordStageExecutionReportParams) error
	ListStageExecutionReports(context.Context, string) ([]StageExecutionReport, error)
	RecordPlannerExecutionReport(context.Context, RecordPlannerExecutionReportParams) error
	RebuildStageMetrics(context.Context, string, string) error
	CleanupExpiredTelemetry(context.Context, time.Time, int) (int64, error)
}

// PostgresStore never starts a transaction. Pass a pgx.Tx to NewPostgresStore
// when multiple repository operations must share one atomic boundary.
type PostgresStore struct {
	db persistencepostgres.DBTX
}

var _ Repository = (*PostgresStore)(nil)

func NewPostgresStore(db persistencepostgres.DBTX) *PostgresStore {
	return &PostgresStore{db: db}
}

func (s *PostgresStore) CreateRun(ctx context.Context, params CreateRunParams) (WorkflowRun, error) {
	if err := validateCreateRun(params); err != nil {
		return WorkflowRun{}, err
	}
	parameters := params.Parameters
	if parameters == nil {
		parameters = map[string]string{}
	}
	encodedParameters, err := json.Marshal(parameters)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("create WorkflowRun: encode parameters: %w", err)
	}

	row := s.db.QueryRow(ctx, `
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    state, state_reason_code, state_reason_message
) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, 'initializing', 'created', '')
RETURNING `+workflowRunColumns,
		params.RunID, params.OwnerID, params.WorkflowName, params.WorkflowVersion,
		params.WorkflowSchemaVersion, []byte(params.WorkflowSnapshot), encodedParameters,
	)
	result, err := scanWorkflowRun(row)
	if err != nil {
		if persistencepostgres.SQLState(err) == "23505" {
			return WorkflowRun{}, fmt.Errorf("create WorkflowRun %q: %w", params.RunID, ErrConflict)
		}
		return WorkflowRun{}, fmt.Errorf("create WorkflowRun %q: %w", params.RunID, err)
	}
	return result, nil
}

func (s *PostgresStore) GetRun(ctx context.Context, runID string) (WorkflowRun, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return WorkflowRun{}, err
	}
	result, err := scanWorkflowRun(s.db.QueryRow(ctx,
		`SELECT `+workflowRunColumns+` FROM workflow_runs WHERE run_id = $1`, runID,
	))
	if errors.Is(err, pgx.ErrNoRows) {
		return WorkflowRun{}, fmt.Errorf("get WorkflowRun %q: %w", runID, ErrNotFound)
	}
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("get WorkflowRun %q: %w", runID, err)
	}
	return result, nil
}

func (s *PostgresStore) TransitionRun(
	ctx context.Context,
	runID string,
	expected WorkflowRunState,
	next WorkflowRunState,
	reason Reason,
) (WorkflowRun, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return WorkflowRun{}, err
	}
	if err := validateRunTransition(expected, next); err != nil {
		return WorkflowRun{}, err
	}
	if err := validateReason(reason); err != nil {
		return WorkflowRun{}, err
	}
	result, err := scanWorkflowRun(s.db.QueryRow(ctx, `
UPDATE workflow_runs
SET state = $3,
    state_reason_code = $4,
    state_reason_message = $5,
    updated_at = clock_timestamp(),
    started_at = CASE WHEN $3 = 'running' AND started_at IS NULL THEN clock_timestamp() ELSE started_at END,
    finished_at = CASE WHEN $3 IN ('succeeded', 'failed', 'cancelled') THEN clock_timestamp() ELSE NULL END,
    scheduler_claim_id = CASE WHEN $3 IN ('succeeded', 'failed', 'cancelled') THEN NULL ELSE scheduler_claim_id END,
    scheduler_claimed_at = CASE WHEN $3 IN ('succeeded', 'failed', 'cancelled') THEN NULL ELSE scheduler_claimed_at END,
    scheduler_claim_expires_at = CASE WHEN $3 IN ('succeeded', 'failed', 'cancelled') THEN NULL ELSE scheduler_claim_expires_at END
WHERE run_id = $1 AND state = $2
RETURNING `+workflowRunColumns,
		runID, expected, next, reason.Code, reason.Message,
	))
	if errors.Is(err, pgx.ErrNoRows) {
		return WorkflowRun{}, &StateConflictError{Resource: "WorkflowRun", ID: runID, Expected: string(expected)}
	}
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("transition WorkflowRun %q from %s to %s: %w", runID, expected, next, err)
	}
	return result, nil
}

func (s *PostgresStore) RequestRunCancellation(
	ctx context.Context,
	runID string,
	cancellation WorkflowRunCancellation,
) (WorkflowRun, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return WorkflowRun{}, err
	}
	if err := cancellation.Validate(); err != nil {
		return WorkflowRun{}, err
	}
	cancellation.RequestedAt = cancellation.RequestedAt.UTC().Round(0)
	encoded, err := json.Marshal(cancellation)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("request WorkflowRun cancellation: encode payload: %w", err)
	}
	result, err := scanWorkflowRun(s.db.QueryRow(ctx, `
UPDATE workflow_runs
SET state = 'cancelling',
    state_reason_code = 'user_cancelled',
    state_reason_message = COALESCE($4, ''),
    cancellation_schema_version = $2,
    run_cancellation = $3::jsonb,
    updated_at = clock_timestamp(),
    finished_at = NULL
WHERE run_id = $1 AND state IN ('initializing', 'running')
RETURNING `+workflowRunColumns,
		runID, contracts.APIVersion, encoded, cancellation.Reason,
	))
	if err == nil {
		return result, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return WorkflowRun{}, fmt.Errorf("request WorkflowRun %q cancellation: %w", runID, err)
	}
	// A repeated cancellation or terminal-state race is an idempotent read of
	// the committed winner. This second statement gets a fresh READ COMMITTED
	// snapshot after any conflicting UPDATE finished.
	return s.GetRun(ctx, runID)
}

func (s *PostgresStore) ClaimRunnableRun(
	ctx context.Context,
	claimID string,
	duration time.Duration,
) (WorkflowRun, error) {
	if err := validateOpaque("claimID", claimID); err != nil {
		return WorkflowRun{}, err
	}
	if duration <= 0 {
		return WorkflowRun{}, invalidf("claim duration must be positive")
	}
	microseconds := duration.Microseconds()
	if microseconds <= 0 {
		return WorkflowRun{}, invalidf("claim duration must be at least one microsecond")
	}
	result, err := scanWorkflowRun(s.db.QueryRow(ctx, `
WITH candidate AS (
    SELECT run_id
    FROM workflow_runs
    WHERE state IN ('running', 'cancelling')
      AND (scheduler_claim_id IS NULL OR scheduler_claim_expires_at <= clock_timestamp())
    ORDER BY created_at, run_id
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE workflow_runs AS run
SET scheduler_claim_id = $1,
    scheduler_claimed_at = clock_timestamp(),
    scheduler_claim_expires_at = clock_timestamp() + ($2::bigint * interval '1 microsecond'),
    updated_at = clock_timestamp()
FROM candidate
WHERE run.run_id = candidate.run_id
RETURNING `+prefixedWorkflowRunColumns("run"), claimID, microseconds,
	))
	if errors.Is(err, pgx.ErrNoRows) {
		return WorkflowRun{}, ErrNoWork
	}
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("claim runnable WorkflowRun: %w", err)
	}
	return result, nil
}

func (s *PostgresStore) ReleaseRunClaim(ctx context.Context, runID, claimID string) error {
	if err := validateOpaque("runID", runID); err != nil {
		return err
	}
	if err := validateOpaque("claimID", claimID); err != nil {
		return err
	}
	tag, err := s.db.Exec(ctx, `
UPDATE workflow_runs
SET scheduler_claim_id = NULL,
    scheduler_claimed_at = NULL,
    scheduler_claim_expires_at = NULL,
    updated_at = clock_timestamp()
WHERE run_id = $1 AND scheduler_claim_id = $2`, runID, claimID)
	if err != nil {
		return fmt.Errorf("release WorkflowRun %q claim: %w", runID, err)
	}
	if tag.RowsAffected() != 1 {
		return &StateConflictError{Resource: "WorkflowRun claim", ID: runID, Expected: claimID}
	}
	return nil
}

func (s *PostgresStore) RenewRunClaim(
	ctx context.Context,
	runID string,
	claimID string,
	duration time.Duration,
) error {
	if err := validateOpaque("runID", runID); err != nil {
		return err
	}
	if err := validateOpaque("claimID", claimID); err != nil {
		return err
	}
	if duration <= 0 || duration.Microseconds() <= 0 {
		return invalidf("claim duration must be at least one microsecond")
	}
	tag, err := s.db.Exec(ctx, `
UPDATE workflow_runs
SET scheduler_claim_expires_at = clock_timestamp() + ($3::bigint * interval '1 microsecond'),
    updated_at = clock_timestamp()
WHERE run_id = $1 AND state IN ('running', 'cancelling') AND scheduler_claim_id = $2`,
		runID, claimID, duration.Microseconds())
	if err != nil {
		return fmt.Errorf("renew WorkflowRun %q claim: %w", runID, err)
	}
	if tag.RowsAffected() != 1 {
		return &StateConflictError{Resource: "WorkflowRun claim", ID: runID, Expected: claimID}
	}
	return nil
}

func validateCreateRun(params CreateRunParams) error {
	for field, value := range map[string]string{
		"runID": params.RunID, "ownerID": params.OwnerID,
		"workflowName": params.WorkflowName, "workflowVersion": params.WorkflowVersion,
		"workflowSchemaVersion": params.WorkflowSchemaVersion,
	} {
		if err := validateOpaque(field, value); err != nil {
			return err
		}
	}
	if err := validateJSONObject("workflowSnapshot", params.WorkflowSnapshot); err != nil {
		return err
	}
	for name := range params.Parameters {
		if strings.TrimSpace(name) == "" {
			return invalidf("parameter name is required")
		}
	}
	return nil
}

func validateRunTransition(expected, next WorkflowRunState) error {
	allowed := map[WorkflowRunState]map[WorkflowRunState]bool{
		RunInitializing: {RunRunning: true, RunFailed: true},
		RunRunning:      {RunSucceeded: true, RunFailed: true},
		RunCancelling:   {RunCancelled: true},
	}
	if !allowed[expected][next] {
		return invalidf("illegal WorkflowRun transition %s -> %s", expected, next)
	}
	return nil
}
