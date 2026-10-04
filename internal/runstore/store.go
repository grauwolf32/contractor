package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentialerrors"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

var credentialRunIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$`)

// Repository groups the broad RunStore operations used by e2e test helpers.
// Production consumers declare narrower interfaces at their use sites.
// PostgresStore implements it for both a pool and an explicit pgx transaction.
type Repository interface {
	PinRuntimeLabels(context.Context, []string, ...bool) (runtimeconfig.RunSnapshot, error)
	CreateRun(context.Context, CreateRunParams) (WorkflowRun, error)
	CreateRunIdempotent(context.Context, CreateRunIdempotentParams) (WorkflowRun, bool, error)
	CreateAuditRun(context.Context, CreateAuditRunParams) (WorkflowRun, error)
	SetRunSkillSelections(context.Context, string, []contracts.RunSkillSnapshot) error
	CompleteRunSkillInitialization(context.Context, string, []contracts.RunSkillSnapshot) error
	LookupRunIdempotency(context.Context, string, string, string) (WorkflowRun, bool, error)
	GetRun(context.Context, string) (WorkflowRun, error)
	ListRuns(context.Context, ListRunsParams) ([]WorkflowRunSummary, error)
	RunDeletionBlocker(context.Context, string, string) (*RunNotDeletableReason, error)
	DeleteReleasedTerminalRun(context.Context, string, string) error
	ListRunQueue(context.Context, ListRunQueueParams) ([]WorkflowRunQueueItem, error)
	GetOwnerQueueControl(context.Context, string) (OwnerQueueControl, error)
	UpdateOwnerQueueControl(context.Context, UpdateOwnerQueueControlParams) (OwnerQueueControl, error)
	RecordRunOutputPublication(context.Context, RecordRunOutputPublicationParams) (RunOutputPublication, bool, error)
	ListRunOutputPublications(context.Context, string) ([]RunOutputPublication, error)
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
	rows, err := s.db.Query(ctx, listNonTerminalRunIDsByCredentialSQL, credentialID, limit)
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

// PostgresStore runs most operations on its supplied DBTX. When backed by a
// pool, DeleteReleasedTerminalRun and ResumeFailedRun start their own retrying
// transactions; with a pgx.Tx they join the caller's transaction. Runtime
// label pinning additionally requires NewRunCreationPostgresStore.
type PostgresStore struct {
	db                persistencepostgres.DBTX
	runCreationTx     pgx.Tx
	runLLMCredentials runtimeconfig.TransactionLLMCredentialLookup
}

var _ Repository = (*PostgresStore)(nil)

func NewPostgresStore(db persistencepostgres.DBTX) *PostgresStore {
	return &PostgresStore{db: db}
}

// NewRunCreationPostgresStore is the only constructor which enables Runtime
// label pinning. The concrete transaction-bound lookup prevents a global
// pool-backed credential provider from being used inside the owning tx.
func NewRunCreationPostgresStore(
	tx pgx.Tx,
	llmCredentials runtimeconfig.TransactionLLMCredentialLookup,
) *PostgresStore {
	return &PostgresStore{db: tx, runCreationTx: tx, runLLMCredentials: llmCredentials}
}

func (s *PostgresStore) PinRuntimeLabels(
	ctx context.Context, labels []string, modelFree ...bool,
) (runtimeconfig.RunSnapshot, error) {
	if s == nil || s.runCreationTx == nil {
		return runtimeconfig.RunSnapshot{}, errors.New("Runtime label pinning requires a Run-creation transaction store")
	}
	return runtimeconfig.PinRunSnapshot(
		ctx, s.runCreationTx, labels, runtimeCredentialValidator{s.db}, s.runLLMCredentials, modelFree...,
	)
}

type runtimeCredentialValidator struct{ db persistencepostgres.DBTX }

func (v runtimeCredentialValidator) ValidateRuntimeCredential(
	ctx context.Context, credentialID string, allowedKinds ...string,
) error {
	var kind string
	err := v.db.QueryRow(ctx, `
SELECT credential_kind
FROM runtime_credentials
WHERE credential_id = $1 AND deleted_at IS NULL`, credentialID).Scan(&kind)
	if err != nil {
		return err
	}
	for _, allowed := range allowedKinds {
		if kind == allowed {
			return nil
		}
	}
	return credentialerrors.RuntimeNotFound
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
	encodedProjectHTTPTarget, err := encodeProjectHTTPTarget(params.ProjectHTTPTarget)
	if err != nil {
		return WorkflowRun{}, err
	}
	runtimeLabels := params.RuntimeConfig.ExplicitLabels()
	encodedMetadataLabels, err := json.Marshal(metadataLabels)
	if err != nil {
		return WorkflowRun{}, fmt.Errorf("create WorkflowRun: encode metadata labels: %w", err)
	}

	row := s.db.QueryRow(ctx, createRunSQL+prefixedWorkflowRunColumns("inserted_run")+`
FROM inserted_run`,
		params.RunID, params.OwnerID, params.ProjectID, params.WorkflowName, params.WorkflowVersion,
		params.WorkflowSchemaVersion, []byte(params.WorkflowSnapshot), encodedParameters,
		runtimeLabels, encodedRuntimeConfig, encodedProjectHTTPTarget, encodedMetadataLabels,
	)
	result, err := scanWorkflowRun(row)
	if err != nil {
		if persistencepostgres.SQLState(err) == "55000" {
			return WorkflowRun{}, fmt.Errorf("create WorkflowRun %q: %w", params.RunID, ErrProjectDeleting)
		}
		if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateUniqueViolation {
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
	if !contentdigest.Valid(params.RequestDigest) {
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
	encodedProjectHTTPTarget, err := encodeProjectHTTPTarget(params.ProjectHTTPTarget)
	if err != nil {
		return WorkflowRun{}, false, err
	}
	runtimeLabels := params.RuntimeConfig.ExplicitLabels()
	encodedMetadataLabels, err := json.Marshal(metadataLabels)
	if err != nil {
		return WorkflowRun{}, false, fmt.Errorf("create idempotent WorkflowRun: encode metadata labels: %w", err)
	}
	result, err := scanWorkflowRun(s.db.QueryRow(ctx, createRunIdempotentSQL+prefixedWorkflowRunColumns("inserted_run")+`
FROM inserted_run`,
		params.RunID, params.OwnerID, params.ProjectID, params.WorkflowName, params.WorkflowVersion,
		params.WorkflowSchemaVersion, []byte(params.WorkflowSnapshot), encodedParameters,
		runtimeLabels, encodedRuntimeConfig, encodedProjectHTTPTarget, params.IdempotencyKey, params.RequestDigest,
		encodedMetadataLabels,
	))
	if err == nil {
		result.MetadataLabels = metadataLabels
		return result, true, nil
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return WorkflowRun{}, false, fmt.Errorf("create idempotent WorkflowRun %q: %w", params.RunID, ErrProjectDeleting)
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
	if !contentdigest.Valid(requestDigest) {
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
	if params.ProjectID != nil {
		if err := validateOpaque("projectID", *params.ProjectID); err != nil {
			return nil, err
		}
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
	var lifecycle *string
	if params.Lifecycle != nil {
		if !params.Lifecycle.Valid() {
			return nil, invalidf("unknown WorkflowRun lifecycle %q", *params.Lifecycle)
		}
		value := string(*params.Lifecycle)
		lifecycle = &value
	}
	rows, err := s.db.Query(ctx, listRunsSQL,
		params.OwnerID, state, params.BeforeCreatedAt, params.BeforeRunID, params.Limit,
		selectorKeys, selectorValues, params.ProjectID, lifecycle,
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
			&run.RunID, &run.ProjectID, &run.WorkflowName, &run.WorkflowVersion, &state,
			&run.CreatedAt, &run.UpdatedAt, &run.FinishedAt, &run.Deletable, &encodedLabels,
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
	result, err := scanWorkflowRun(s.db.QueryRow(ctx, transitionRunSQL+workflowRunColumns,
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
	result, err := scanWorkflowRun(s.db.QueryRow(ctx, requestRunCancellationSQL+workflowRunColumns,
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
	result, err := scanWorkflowRun(s.db.QueryRow(ctx, claimRunnableRunSQL+prefixedWorkflowRunColumns("run"), claimID, microseconds, deferredClaimYield.Microseconds()))
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
	return s.releaseRunClaim(ctx, runID, claimID, false)
}

// deferredClaimYield is how long a deferred Run ranks behind pending work.
// After it, the Run competes in its own state's tier again, so a Run that stays
// blocked takes at most one claim per window and never starves behind newer
// Runs.
const deferredClaimYield = 30 * time.Second

// DeferRunClaim releases the lease while moving this Run behind pending work
// for deferredClaimYield. A later successful attempt clears the marker
// through ReleaseRunClaim.
func (s *PostgresStore) DeferRunClaim(ctx context.Context, runID, claimID string) error {
	return s.releaseRunClaim(ctx, runID, claimID, true)
}

func (s *PostgresStore) releaseRunClaim(ctx context.Context, runID, claimID string, deferred bool) error {
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
    scheduler_deferred = $3,
    updated_at = clock_timestamp()
WHERE run_id = $1 AND scheduler_claim_id = $2`, runID, claimID, deferred)
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
WHERE run_id = $1 AND state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling') AND scheduler_claim_id = $2`,
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
	if params.ProjectID != nil {
		if err := validateOpaque("projectID", *params.ProjectID); err != nil {
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
	if params.ProjectHTTPTarget != nil {
		if params.ProjectID == nil {
			return invalidf("Project HTTP target requires Project membership")
		}
		if err := params.ProjectHTTPTarget.Validate(); err != nil {
			return invalidf("Project HTTP target snapshot is invalid: %v", err)
		}
	}
	return nil
}

func encodeProjectHTTPTarget(target *contracts.HTTPOriginTargetRef) ([]byte, error) {
	if target == nil {
		return nil, nil
	}
	encoded, err := json.Marshal(target)
	if err != nil {
		return nil, fmt.Errorf("create WorkflowRun: encode Project HTTP target snapshot: %w", err)
	}
	return encoded, nil
}

func (s *PostgresStore) loadRunMetadataLabels(
	ctx context.Context, runID string,
) (RunMetadataLabels, error) {
	var encoded []byte
	if err := s.db.QueryRow(ctx, `SELECT metadata_labels FROM workflow_runs WHERE run_id = $1`, runID).Scan(&encoded); err != nil {
		return nil, err
	}
	result := make(RunMetadataLabels)
	if err := json.Unmarshal(encoded, &result); err != nil {
		return nil, fmt.Errorf("decode persisted Run metadata labels: %w", err)
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
		RunInitializing: {RunPending: true, RunRunning: true, RunFailed: true},
		RunPending:      {RunRunning: true, RunFailed: true},
		RunWaiting:      {RunRunning: true, RunSucceeded: true, RunFailed: true},
		RunRunning:      {RunWaiting: true, RunSucceeded: true, RunFailed: true},
		RunCancelling:   {RunCancelled: true},
	}
	if !allowed[expected][next] {
		return invalidf("illegal WorkflowRun transition %s -> %s", expected, next)
	}
	return nil
}

func validWorkflowRunState(state WorkflowRunState) bool {
	switch state {
	case RunInitializing, RunPending, RunRunning, RunWaiting, RunCancelling, RunSucceeded, RunFailed, RunCancelled:
		return true
	default:
		return false
	}
}
