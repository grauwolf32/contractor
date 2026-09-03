package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

var credentialRunIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$`)

// Repository is the durable boundary used by the public API and Scheduler.
// PostgresStore implements it for both a pool and an explicit pgx transaction.
type Repository interface {
	PinRuntimeLabels(context.Context, []string, config.CredentialLookup) (runtimeconfig.RunSnapshot, error)
	CreateRun(context.Context, CreateRunParams) (WorkflowRun, error)
	CreateRunIdempotent(context.Context, CreateRunIdempotentParams) (WorkflowRun, bool, error)
	SetRunSkillSelections(context.Context, string, []contracts.RunSkillSnapshot) error
	CompleteRunSkillInitialization(context.Context, string, []contracts.RunSkillSnapshot) error
	LookupRunIdempotency(context.Context, string, string, string) (WorkflowRun, bool, error)
	GetRun(context.Context, string) (WorkflowRun, error)
	ListRuns(context.Context, ListRunsParams) ([]WorkflowRunSummary, error)
	ListNonTerminalRunIDsByCredential(context.Context, string, int) ([]string, error)
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
	GetRunEventCursor(context.Context, string) (WorkflowRunEventCursor, error)
	ListRunEvents(context.Context, string, int64, int) ([]WorkflowRunEvent, error)
	RecordStageAllocation(context.Context, StageAllocation) error
	ListStageAllocations(context.Context, string) ([]StageAllocation, error)
	MarkStageAllocationReleaseAttempt(context.Context, string) error
	MarkStageAllocationReleased(context.Context, string) error
	RecordStageExecutionReport(context.Context, RecordStageExecutionReportParams) error
	ListStageExecutionReports(context.Context, string) ([]StageExecutionReport, error)
	RecordPlannerExecutionReport(context.Context, RecordPlannerExecutionReportParams) error
	RebuildStageMetrics(context.Context, string, string) error
	CleanupExpiredTelemetry(context.Context, time.Time, int) (int64, error)
}

// ListNonTerminalRunIDsByCredential returns a bounded deterministic set of
// immutable Run snapshots that pin the exact non-secret credential ID.
func (s *PostgresStore) ListNonTerminalRunIDsByCredential(
	ctx context.Context,
	credentialID string,
	limit int,
) ([]string, error) {
	if err := (contracts.LLMCredentialRef{CredentialID: credentialID}).Validate(); err != nil {
		return nil, invalidf("credential ID is invalid")
	}
	if limit < 1 || limit > 128 {
		return nil, invalidf("credential Run-reference limit must be between 1 and 128")
	}
	rows, err := s.db.Query(ctx, `
WITH credential_runs AS (
    SELECT run_id, created_at
    FROM workflow_runs
    WHERE state IN ('initializing', 'running', 'cancelling')
      AND (
          jsonb_path_exists(
              workflow_snapshot,
              '$.**.credentialId ? (@ == $credential)',
              jsonb_build_object('credential', to_jsonb($1::text)),
              true
          )
          OR (runtime_config_snapshot->'llmCredentialIds') ? $1
      )
    UNION
    SELECT e.run_id, r.created_at
    FROM stage_allocations a
    JOIN stage_executions e ON e.stage_execution_id = a.stage_execution_id
    JOIN workflow_runs r ON r.run_id = e.run_id
    WHERE a.release_completed_at IS NULL
      AND a.runtime_configuration #>> '{provenance,llmCredential,credentialId}' = $1
)
SELECT run_id
FROM credential_runs
ORDER BY created_at, run_id
LIMIT $2`, credentialID, limit)
	if err != nil {
		return nil, errors.New("list non-terminal Runs by credential")
	}
	defer rows.Close()
	result := make([]string, 0)
	for rows.Next() {
		var runID string
		if err := rows.Scan(&runID); err != nil {
			return nil, errors.New("read non-terminal Run credential reference")
		}
		if !credentialRunIDPattern.MatchString(runID) {
			return nil, errors.New("stored credential Run reference is not publicly safe")
		}
		result = append(result, runID)
	}
	if err := rows.Err(); err != nil {
		return nil, errors.New("iterate non-terminal Run credential references")
	}
	return result, nil
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

func (s *PostgresStore) PinRuntimeLabels(
	ctx context.Context, labels []string, llmCredentials config.CredentialLookup,
) (runtimeconfig.RunSnapshot, error) {
	return runtimeconfig.PinRunSnapshot(
		ctx, s.db, labels, runtimeCredentialValidator{s.db}, llmCredentials,
	)
}

type runtimeCredentialValidator struct{ db persistencepostgres.DBTX }

func (v runtimeCredentialValidator) ValidateRuntimeCredential(
	ctx context.Context, credentialID string, allowedKinds ...string,
) error {
	var kind string
	err := v.db.QueryRow(ctx, `
SELECT c.credential_kind
FROM runtime_credentials AS c
LEFT JOIN runtime_credential_tombstones AS t USING (credential_id)
WHERE c.credential_id = $1 AND t.credential_id IS NULL`, credentialID).Scan(&kind)
	if err != nil {
		return err
	}
	for _, allowed := range allowedKinds {
		if kind == allowed {
			return nil
		}
	}
	return fmt.Errorf("Runtime credential kind is incompatible")
}

func (s *PostgresStore) CreateRun(ctx context.Context, params CreateRunParams) (WorkflowRun, error) {
	if err := validateCreateRun(params); err != nil {
		return WorkflowRun{}, err
	}
	metadataLabels, _ := NormalizeRunMetadataLabels(params.MetadataLabels)
	parameters := params.Parameters
	if parameters == nil {
		parameters = map[string]string{}
	}
	encodedParameters, err := json.Marshal(parameters)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("create WorkflowRun: encode parameters: %w", err)
	}
	encodedRuntimeConfig, err := json.Marshal(params.RuntimeConfig)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("create WorkflowRun: encode RuntimeConfig snapshot: %w", err)
	}
	runtimeLabels := params.RuntimeConfig.ExplicitLabels()
	encodedMetadataLabels, err := json.Marshal(metadataLabels)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("create WorkflowRun: encode metadata labels: %w", err)
	}

	row := s.db.QueryRow(ctx, `
WITH inserted_run AS (
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    runtime_labels, runtime_config_snapshot,
    state, state_reason_code, state_reason_message
) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9::jsonb, 'initializing', 'created', '')
RETURNING *
), inserted_labels AS (
    INSERT INTO workflow_run_metadata_labels (run_id, ordinal, label_key, label_value)
    SELECT inserted_run.run_id,
           row_number() OVER (ORDER BY entry.key), entry.key, entry.value
    FROM inserted_run
    CROSS JOIN LATERAL jsonb_each_text($10::jsonb) AS entry
)
SELECT `+prefixedWorkflowRunColumns("inserted_run")+`
FROM inserted_run`,
		params.RunID, params.OwnerID, params.WorkflowName, params.WorkflowVersion,
		params.WorkflowSchemaVersion, []byte(params.WorkflowSnapshot), encodedParameters,
		runtimeLabels, encodedRuntimeConfig, encodedMetadataLabels,
	)
	result, err := scanWorkflowRun(row)
	if err != nil {
		if persistencepostgres.SQLState(err) == "23505" {
			return WorkflowRun{}, fmt.Errorf("create WorkflowRun %q: %w", params.RunID, ErrConflict)
		}
		return WorkflowRun{}, fmt.Errorf("create WorkflowRun %q: %w", params.RunID, err)
	}
	result.MetadataLabels = metadataLabels
	return result, nil
}

// CreateRunIdempotent atomically claims one public create-Run key. The bool is
// true only for the transaction that inserted the Run. A concurrent or later
// retry with the same owner, key, and request digest receives the existing Run;
// reuse of the key for another request fails closed.
func (s *PostgresStore) CreateRunIdempotent(
	ctx context.Context,
	params CreateRunIdempotentParams,
) (WorkflowRun, bool, error) {
	if err := validateCreateRun(params.CreateRunParams); err != nil {
		return WorkflowRun{}, false, err
	}
	metadataLabels, _ := NormalizeRunMetadataLabels(params.MetadataLabels)
	if err := validateIdempotencyKey(params.IdempotencyKey); err != nil {
		return WorkflowRun{}, false, err
	}
	if !digestPattern.MatchString(params.RequestDigest) {
		return WorkflowRun{}, false, invalidf("request digest is invalid")
	}
	parameters := params.Parameters
	if parameters == nil {
		parameters = map[string]string{}
	}
	encodedParameters, err := json.Marshal(parameters)
	if err != nil {
		return WorkflowRun{}, false, fmt.Errorf("create idempotent WorkflowRun: encode parameters: %w", err)
	}
	encodedRuntimeConfig, err := json.Marshal(params.RuntimeConfig)
	if err != nil {
		return WorkflowRun{}, false, fmt.Errorf("create idempotent WorkflowRun: encode RuntimeConfig snapshot: %w", err)
	}
	runtimeLabels := params.RuntimeConfig.ExplicitLabels()
	encodedMetadataLabels, err := json.Marshal(metadataLabels)
	if err != nil {
		return WorkflowRun{}, false, fmt.Errorf("create idempotent WorkflowRun: encode metadata labels: %w", err)
	}
	result, err := scanWorkflowRun(s.db.QueryRow(ctx, `
WITH inserted_run AS (
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    runtime_labels, runtime_config_snapshot,
    request_idempotency_key, request_digest,
    state, state_reason_code, state_reason_message
) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9::jsonb, $10, $11, 'initializing', 'created', '')
ON CONFLICT DO NOTHING
RETURNING *
), inserted_labels AS (
    INSERT INTO workflow_run_metadata_labels (run_id, ordinal, label_key, label_value)
    SELECT inserted_run.run_id,
           row_number() OVER (ORDER BY entry.key), entry.key, entry.value
    FROM inserted_run
    CROSS JOIN LATERAL jsonb_each_text($12::jsonb) AS entry
)
SELECT `+prefixedWorkflowRunColumns("inserted_run")+`
FROM inserted_run`,
		params.RunID, params.OwnerID, params.WorkflowName, params.WorkflowVersion,
		params.WorkflowSchemaVersion, []byte(params.WorkflowSnapshot), encodedParameters,
		runtimeLabels, encodedRuntimeConfig, params.IdempotencyKey, params.RequestDigest,
		encodedMetadataLabels,
	))
	if err == nil {
		result.MetadataLabels = metadataLabels
		return result, true, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return WorkflowRun{}, false, fmt.Errorf("create idempotent WorkflowRun %q: %w", params.RunID, err)
	}
	var existingRunID, existingDigest string
	err = s.db.QueryRow(ctx, `
SELECT run_id, request_digest
FROM workflow_runs
WHERE owner_id = $1 AND request_idempotency_key = $2`,
		params.OwnerID, params.IdempotencyKey,
	).Scan(&existingRunID, &existingDigest)
	if errors.Is(err, pgx.ErrNoRows) {
		// Another unique identity, normally run_id, caused the conflict.
		return WorkflowRun{}, false, fmt.Errorf("create idempotent WorkflowRun %q: %w", params.RunID, ErrConflict)
	}
	if err != nil {
		return WorkflowRun{}, false, fmt.Errorf("read idempotent WorkflowRun: %w", err)
	}
	if existingDigest != params.RequestDigest {
		return WorkflowRun{}, false, fmt.Errorf("reuse public idempotency key: %w", ErrConflict)
	}
	existing, err := s.GetRun(ctx, existingRunID)
	if err != nil {
		return WorkflowRun{}, false, err
	}
	return existing, false, nil
}

// LookupRunIdempotency resolves an already committed public create-Run claim
// without consulting mutable Workflow dependencies such as managed
// credentials. A digest mismatch is the same fail-closed key reuse conflict as
// CreateRunIdempotent.
func (s *PostgresStore) LookupRunIdempotency(
	ctx context.Context,
	ownerID string,
	idempotencyKey string,
	requestDigest string,
) (WorkflowRun, bool, error) {
	if err := validateOpaque("ownerID", ownerID); err != nil {
		return WorkflowRun{}, false, err
	}
	if err := validateIdempotencyKey(idempotencyKey); err != nil {
		return WorkflowRun{}, false, err
	}
	if !digestPattern.MatchString(requestDigest) {
		return WorkflowRun{}, false, invalidf("request digest is invalid")
	}
	var runID, storedDigest string
	err := s.db.QueryRow(ctx, `
SELECT run_id, request_digest
FROM workflow_runs
WHERE owner_id = $1 AND request_idempotency_key = $2`, ownerID, idempotencyKey,
	).Scan(&runID, &storedDigest)
	if errors.Is(err, pgx.ErrNoRows) {
		return WorkflowRun{}, false, nil
	}
	if err != nil {
		return WorkflowRun{}, false, fmt.Errorf("lookup idempotent WorkflowRun: %w", err)
	}
	if storedDigest != requestDigest {
		return WorkflowRun{}, false, fmt.Errorf("reuse public idempotency key: %w", ErrConflict)
	}
	run, err := s.GetRun(ctx, runID)
	if err != nil {
		return WorkflowRun{}, false, err
	}
	return run, true, nil
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
	result.MetadataLabels, err = s.loadRunMetadataLabels(ctx, runID)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("get WorkflowRun %q metadata labels: %w", runID, err)
	}
	return result, nil
}

// ListRuns returns at most Limit Runs in deterministic newest-first order.
// Callers request one extra row when they need to construct a continuation
// cursor; the repository does not attach transport pagination semantics.
func (s *PostgresStore) ListRuns(ctx context.Context, params ListRunsParams) ([]WorkflowRunSummary, error) {
	if err := validateOpaque("ownerID", params.OwnerID); err != nil {
		return nil, err
	}
	selectors, err := NormalizeRunMetadataLabelSelectors(params.MetadataLabelSelectors)
	if err != nil {
		return nil, err
	}
	selectorKeys := make([]string, len(selectors))
	selectorValues := make([]string, len(selectors))
	for index, selector := range selectors {
		selectorKeys[index] = selector.Key
		selectorValues[index] = selector.Value
	}
	if params.Limit < 1 || params.Limit > 201 {
		return nil, invalidf("Run page limit must be between 1 and 201")
	}
	if (params.BeforeCreatedAt == nil) != (params.BeforeRunID == "") {
		return nil, invalidf("Run page keyset is incomplete")
	}
	if params.BeforeCreatedAt != nil {
		if params.BeforeCreatedAt.IsZero() {
			return nil, invalidf("Run page timestamp is invalid")
		}
		if err := validateOpaque("beforeRunID", params.BeforeRunID); err != nil {
			return nil, err
		}
	}
	var state *string
	if params.State != nil {
		if !validWorkflowRunState(*params.State) {
			return nil, invalidf("unknown WorkflowRun state %q", *params.State)
		}
		value := string(*params.State)
		state = &value
	}
	rows, err := s.db.Query(ctx, `
WITH page AS (
    SELECT run_id, workflow_name, workflow_version, state, created_at, updated_at, finished_at
    FROM workflow_runs
    WHERE owner_id = $1
      AND ($2::text IS NULL OR state = $2)
      AND ($3::timestamptz IS NULL OR (created_at, run_id) < ($3, $4))
      AND (
          cardinality($6::text[]) = 0
          OR (
              SELECT count(*)
              FROM workflow_run_metadata_labels AS matched
              JOIN unnest($6::text[], $7::text[]) AS required(label_key, label_value)
                ON matched.label_key = required.label_key
               AND matched.label_value = required.label_value
              WHERE matched.run_id = workflow_runs.run_id
          ) = cardinality($6::text[])
      )
    ORDER BY created_at DESC, run_id DESC
    LIMIT $5
)
SELECT page.run_id, page.workflow_name, page.workflow_version, page.state,
       page.created_at, page.updated_at, page.finished_at,
       COALESCE(
           jsonb_object_agg(labels.label_key, labels.label_value ORDER BY labels.label_key)
               FILTER (WHERE labels.label_key IS NOT NULL),
           '{}'::jsonb
       )
FROM page
LEFT JOIN workflow_run_metadata_labels AS labels USING (run_id)
GROUP BY page.run_id, page.workflow_name, page.workflow_version, page.state,
         page.created_at, page.updated_at, page.finished_at
ORDER BY page.created_at DESC, page.run_id DESC`,
		params.OwnerID, state, params.BeforeCreatedAt, params.BeforeRunID, params.Limit,
		selectorKeys, selectorValues,
	)
	if err != nil {
		return nil, fmt.Errorf("list WorkflowRuns for owner: %w", err)
	}
	defer rows.Close()
	result := make([]WorkflowRunSummary, 0, params.Limit)
	for rows.Next() {
		var run WorkflowRunSummary
		var state string
		var encodedLabels []byte
		if scanErr := rows.Scan(
			&run.RunID, &run.WorkflowName, &run.WorkflowVersion, &state,
			&run.CreatedAt, &run.UpdatedAt, &run.FinishedAt, &encodedLabels,
		); scanErr != nil {
			return nil, fmt.Errorf("scan WorkflowRun summary page: %w", scanErr)
		}
		run.State = WorkflowRunState(state)
		if decodeErr := json.Unmarshal(encodedLabels, &run.MetadataLabels); decodeErr != nil {
			return nil, fmt.Errorf("decode WorkflowRun summary metadata labels: %w", decodeErr)
		}
		if validateErr := run.MetadataLabels.Validate(); validateErr != nil {
			return nil, fmt.Errorf("validate WorkflowRun summary metadata labels: %w", validateErr)
		}
		result = append(result, run)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate WorkflowRun page: %w", err)
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
	result.MetadataLabels, err = s.loadRunMetadataLabels(ctx, runID)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("load transitioned WorkflowRun %q metadata labels: %w", runID, err)
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
		result.MetadataLabels, err = s.loadRunMetadataLabels(ctx, runID)
		if err != nil {
			return WorkflowRun{}, fmt.Errorf("load cancelled WorkflowRun %q metadata labels: %w", runID, err)
		}
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
    WHERE (state IN ('running', 'cancelling')
       OR (state = 'initializing' AND state_reason_code = 'skill_initialization_pending'))
      AND (scheduler_claim_id IS NULL OR scheduler_claim_expires_at <= clock_timestamp())
    ORDER BY CASE
                 WHEN state = 'cancelling' THEN 0
                 WHEN state = 'running' THEN 1
                 ELSE 2
             END,
             updated_at, created_at, run_id
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
	result.MetadataLabels, err = s.loadRunMetadataLabels(ctx, result.RunID)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("load claimed WorkflowRun metadata labels: %w", err)
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
WHERE run_id = $1 AND state IN ('initializing', 'running', 'cancelling') AND scheduler_claim_id = $2`,
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
	if _, err := NormalizeRunMetadataLabels(params.MetadataLabels); err != nil {
		return err
	}
	if err := params.RuntimeConfig.Validate(); err != nil {
		return invalidf("RuntimeConfig snapshot is invalid: %v", err)
	}
	return nil
}

func (s *PostgresStore) loadRunMetadataLabels(
	ctx context.Context, runID string,
) (RunMetadataLabels, error) {
	rows, err := s.db.Query(ctx, `
SELECT label_key, label_value
FROM workflow_run_metadata_labels
WHERE run_id = $1
ORDER BY ordinal`, runID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := make(RunMetadataLabels)
	for rows.Next() {
		var key, value string
		if err := rows.Scan(&key, &value); err != nil {
			return nil, err
		}
		if _, duplicate := result[key]; duplicate {
			return nil, fmt.Errorf("duplicate persisted Run metadata label %q", key)
		}
		result[key] = value
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if err := result.Validate(); err != nil {
		return nil, err
	}
	return result, nil
}

func validateIdempotencyKey(value string) error {
	if len(value) == 0 || len(value) > 128 {
		return invalidf("idempotency key must contain 1 to 128 characters")
	}
	for index, character := range value {
		valid := character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' || index > 0 && strings.ContainsRune("._:-", character)
		if !valid {
			return invalidf("idempotency key contains an invalid character")
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

func validWorkflowRunState(state WorkflowRunState) bool {
	switch state {
	case RunInitializing, RunRunning, RunCancelling, RunSucceeded, RunFailed, RunCancelled:
		return true
	default:
		return false
	}
}
