package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

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
	ListRunQueue(context.Context, ListRunQueueParams) ([]WorkflowRunQueueItem, error)
	GetOwnerQueueControl(context.Context, string) (OwnerQueueControl, error)
	UpdateOwnerQueueControl(context.Context, UpdateOwnerQueueControlParams) (OwnerQueueControl, error)
	RecordRunOutputPublication(context.Context, RecordRunOutputPublicationParams) (RunOutputPublication, bool, error)
	ListRunOutputPublications(context.Context, string) ([]RunOutputPublication, error)
	ListNonTerminalRunIDsByCredential(context.Context, string, int) ([]string, error)
	TransitionRun(context.Context, string, WorkflowRunState, WorkflowRunState, Reason) (WorkflowRun, error)
	RequestRunCancellation(context.Context, string, WorkflowRunCancellation) (WorkflowRun, error)
	ClaimRunnableRun(context.Context, string, time.Duration) (WorkflowRun, error)
	RenewRunClaim(context.Context, string, string, time.Duration) (WorkflowRunState, error)
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
	ListStageExecutionReports(context.Context, string) ([]StageExecutionReport, error)
}

// ListNonTerminalRunIDsByCredential returns a bounded deterministic set of
// immutable Run snapshots that pin the exact non-secret credential ID.
func (s *PostgresStore) ListNonTerminalRunIDsByCredential(
	ctx context.Context,
	credentialID string,
	limit int,
) ([]string, error) {
	if err := (llmgateway.LLMCredentialRef{CredentialID: credentialID}).Validate(); err != nil {
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
		if !contracts.ValidResourceID(runID) {
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
// pool, ResumeFailedRun starts its own retrying transaction; with a pgx.Tx it
// joins the caller's transaction. Run deletion locks exist only on
// PostgresTxStore (see internal/rundeletion). Runtime label pinning
// additionally requires NewRunCreationPostgresStore.
type PostgresStore struct {
	db                    persistencepostgres.DBTX
	runCreationTx         pgx.Tx
	runLLMCredentials     runtimeconfig.TransactionLLMCredentialLookup
	runRuntimeCredentials runtimeconfig.RuntimeCredentialValidator
}

var _ Repository = (*PostgresStore)(nil)

func NewPostgresStore(db persistencepostgres.DBTX) *PostgresStore {
	return &PostgresStore{db: db}
}

// PostgresTxStore is a PostgresStore bound to the caller's transaction. Its
// own methods take row locks that must last until the caller commits, so they
// exist only on a store that cannot be backed by a pool.
type PostgresTxStore struct {
	*PostgresStore
	tx pgx.Tx
}

// NewPostgresTxStore binds a PostgresStore to the caller's transaction tx.
func NewPostgresTxStore(tx pgx.Tx) *PostgresTxStore {
	return &PostgresTxStore{PostgresStore: NewPostgresStore(tx), tx: tx}
}

// NewRunCreationPostgresStore is the only constructor which enables Runtime
// label pinning. The concrete transaction-bound lookup prevents a global
// pool-backed credential provider from being used inside the owning tx, and
// runtimeCredentials must validate through the same tx; production passes
// credentials.NewRuntimeCredentialRepository(tx).
func NewRunCreationPostgresStore(
	tx pgx.Tx,
	llmCredentials runtimeconfig.TransactionLLMCredentialLookup,
	runtimeCredentials runtimeconfig.RuntimeCredentialValidator,
) *PostgresStore {
	return &PostgresStore{
		db: tx, runCreationTx: tx,
		runLLMCredentials: llmCredentials, runRuntimeCredentials: runtimeCredentials,
	}
}

func (s *PostgresStore) PinRuntimeLabels(
	ctx context.Context, labels []string, modelFree ...bool,
) (runtimeconfig.RunSnapshot, error) {
	if s == nil || s.runCreationTx == nil || s.runRuntimeCredentials == nil {
		return runtimeconfig.RunSnapshot{}, errors.New("Runtime label pinning requires a Run-creation transaction store")
	}
	return runtimeconfig.PinRunSnapshot(
		ctx, s.runCreationTx, labels, s.runRuntimeCredentials, s.runLLMCredentials, modelFree...,
	)
}

// runRecord is everything one new initializing WorkflowRun row stores: the
// common creation parameters and, on the paths that have them, the public
// create-Run key with its request digest or the trusted Audit association.
// insertRun is its only encoder and createRunSQL its only statement.
type runRecord struct {
	CreateRunParams
	idempotencyKey     *string
	requestDigest      *string
	auditExecutionID   *string
	auditSubmissionKey *string
}

// insertRun stores record after its caller validated it. A unique conflict
// stores nothing and reports inserted false; each caller resolves that for
// its own path.
func (s *PostgresStore) insertRun(
	ctx context.Context, operation string, record runRecord,
) (WorkflowRun, bool, error) {
	metadataLabels, _ := NormalizeRunMetadataLabels(record.MetadataLabels)
	parameters := record.Parameters
	if parameters == nil {
		parameters = map[string]string{}
	}
	encodedParameters, err := json.Marshal(parameters)
	if err != nil {
		return WorkflowRun{}, false, fmt.Errorf("%s: encode parameters: %w", operation, err)
	}
	encodedRuntimeConfig, err := json.Marshal(record.RuntimeConfig)
	if err != nil {
		return WorkflowRun{}, false, fmt.Errorf("%s: encode RuntimeConfig snapshot: %w", operation, err)
	}
	encodedProjectHTTPTarget, err := encodeProjectHTTPTarget(record.ProjectHTTPTarget)
	if err != nil {
		return WorkflowRun{}, false, err
	}
	encodedMetadataLabels, err := json.Marshal(metadataLabels)
	if err != nil {
		return WorkflowRun{}, false, fmt.Errorf("%s: encode metadata labels: %w", operation, err)
	}
	publicationMode := PublicationOrdinary
	if record.auditExecutionID != nil {
		publicationMode = PublicationAuditManaged
	}
	result, err := scanWorkflowRun(s.db.QueryRow(ctx, createRunSQL+prefixedWorkflowRunColumns("inserted_run")+`
FROM inserted_run`,
		record.RunID, record.OwnerID, record.ProjectID, record.WorkflowName, record.WorkflowVersion,
		record.WorkflowSchemaVersion, []byte(record.WorkflowSnapshot), encodedParameters,
		record.RuntimeConfig.ExplicitLabels(), encodedRuntimeConfig, encodedProjectHTTPTarget, encodedMetadataLabels,
		record.idempotencyKey, record.requestDigest,
		string(publicationMode), record.auditExecutionID, record.auditSubmissionKey,
	))
	switch {
	case err == nil:
		return result, true, nil
	case errors.Is(err, pgx.ErrNoRows):
		return WorkflowRun{}, false, nil
	case persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting:
		return WorkflowRun{}, false, fmt.Errorf("%s %q: %w", operation, record.RunID, ErrProjectDeleting)
	default:
		return WorkflowRun{}, false, fmt.Errorf("%s %q: %w", operation, record.RunID, err)
	}
}

func (s *PostgresStore) CreateRun(ctx context.Context, params CreateRunParams) (WorkflowRun, error) {
	if err := validateCreateRun(params); err != nil {
		return WorkflowRun{}, err
	}
	run, inserted, err := s.insertRun(ctx, "create WorkflowRun", runRecord{CreateRunParams: params})
	if err != nil {
		return WorkflowRun{}, err
	}
	if !inserted {
		return WorkflowRun{}, fmt.Errorf("create WorkflowRun %q: %w", params.RunID, ErrConflict)
	}
	return run, nil
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
	if err := validateIdempotencyKey(params.IdempotencyKey); err != nil {
		return WorkflowRun{}, false, err
	}
	if !contentdigest.Valid(params.RequestDigest) {
		return WorkflowRun{}, false, invalidf("request digest is invalid")
	}
	result, inserted, err := s.insertRun(ctx, "create idempotent WorkflowRun", runRecord{
		CreateRunParams: params.CreateRunParams,
		idempotencyKey:  &params.IdempotencyKey, requestDigest: &params.RequestDigest,
	})
	if err != nil {
		return WorkflowRun{}, false, err
	}
	if inserted {
		return result, true, nil
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
		return WorkflowRun{}, false, fmt.Errorf("reuse public idempotency key: %w", contracts.ErrIdempotencyConflict)
	}
	existing, err := s.GetRun(ctx, existingRunID)
	if err != nil {
		return WorkflowRun{}, false, err
	}
	return existing, false, nil
}

// LookupRunIdempotency resolves an already committed public create-Run claim
// without consulting mutable Workflow dependencies such as managed
// credentials. A digest mismatch is the same permanent
// contracts.ErrIdempotencyConflict as in CreateRunIdempotent.
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
		return WorkflowRun{}, false, fmt.Errorf("reuse public idempotency key: %w", contracts.ErrIdempotencyConflict)
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
	return result, nil
}

// GetRunState reads only the lifecycle state of Run runID. Callers that
// decide on the state alone use it instead of GetRun, which reads, decodes
// and validates every snapshot of the Run.
func (s *PostgresStore) GetRunState(ctx context.Context, runID string) (WorkflowRunState, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return "", err
	}
	var state WorkflowRunState
	err := s.db.QueryRow(ctx, `SELECT state FROM workflow_runs WHERE run_id = $1`, runID).Scan(&state)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", fmt.Errorf("get WorkflowRun %q state: %w", runID, ErrNotFound)
	}
	if err != nil {
		return "", fmt.Errorf("get WorkflowRun %q state: %w", runID, err)
	}
	if !state.Valid() {
		return "", fmt.Errorf("get WorkflowRun %q state: stored state %q is not a WorkflowRun state", runID, state)
	}
	return state, nil
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
	if !pagelimit.ValidFetch(params.Limit) {
		return nil, invalidf("Run page limit must be between 1 and %d", pagelimit.MaxFetchRows)
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
		if !params.State.Valid() {
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
	if err := ValidateRunTransition(expected, next); err != nil {
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

// RenewRunClaim extends claim claimID on an active Run to duration from now
// and returns the Run's state, from which the claim's lane notices a
// cancellation. It writes only the lease expiry: updated_at, which orders
// claims and is the Run's public updatedAt, stays as it is, and no index
// covers the expiry, so a renewal is a heap-only update (SCH-R166).
func (s *PostgresStore) RenewRunClaim(
	ctx context.Context,
	runID string,
	claimID string,
	duration time.Duration,
) (WorkflowRunState, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return "", err
	}
	if err := validateOpaque("claimID", claimID); err != nil {
		return "", err
	}
	if duration <= 0 || duration.Microseconds() <= 0 {
		return "", invalidf("claim duration must be at least one microsecond")
	}
	var state WorkflowRunState
	err := s.db.QueryRow(ctx, `
UPDATE workflow_runs
SET scheduler_claim_expires_at = clock_timestamp() + ($3::bigint * interval '1 microsecond')
WHERE run_id = $1 AND state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling') AND scheduler_claim_id = $2
RETURNING state`,
		runID, claimID, duration.Microseconds()).Scan(&state)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", &StateConflictError{Resource: "WorkflowRun claim", ID: runID, Expected: claimID}
	}
	if err != nil {
		return "", fmt.Errorf("renew WorkflowRun %q claim: %w", runID, err)
	}
	return state, nil
}

func validateCreateRun(params CreateRunParams) error {
	if err := validateResourceID("runID", params.RunID); err != nil {
		return err
	}
	for field, value := range map[string]string{
		"ownerID":      params.OwnerID,
		"workflowName": params.WorkflowName, "workflowVersion": params.WorkflowVersion,
		"workflowSchemaVersion": params.WorkflowSchemaVersion,
	} {
		if err := validateOpaque(field, value); err != nil {
			return err
		}
	}
	if params.ProjectID != nil {
		if err := validateResourceID("projectID", *params.ProjectID); err != nil {
			return err
		}
	}
	// Callers store config.EncodeResolvedWorkflowSnapshot's encoding, the
	// measure of the snapshot bound (LIM-CAT-4, CAT-R76).
	if len(params.WorkflowSnapshot) > resolved.MaxWorkflowSnapshotBytes {
		return invalidf("workflowSnapshot exceeds %d bytes", resolved.MaxWorkflowSnapshotBytes)
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

func encodeProjectHTTPTarget(target *runtimesettings.HTTPOriginTargetRef) ([]byte, error) {
	if target == nil {
		return nil, nil
	}
	encoded, err := json.Marshal(target)
	if err != nil {
		return nil, fmt.Errorf("create WorkflowRun: encode Project HTTP target snapshot: %w", err)
	}
	return encoded, nil
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

// runTransitionOperation names the store operation that performs an edge of
// the WorkflowRun lifecycle.
type runTransitionOperation string

const (
	// TransitionRun performs the edges that record only a reason.
	transitionByTransitionRun runTransitionOperation = "TransitionRun"
	// RequestRunCancellation records the WorkflowRunCancellation that
	// cancelling and cancelled require (RUN-R40, RUN-R41).
	transitionByCancellation runTransitionOperation = "RequestRunCancellation"
	// ResumeFailedRun records the continuation receipt and the new attempt
	// with the one transition out of failed (RUN-R46).
	transitionByContinuation runTransitionOperation = "ResumeFailedRun"
	// EnterWaiting and LeaveWaiting move an admitted Run between running and
	// waiting for model-availability recovery (RUN-R30).
	transitionByGatewayWait runTransitionOperation = "EnterWaiting and LeaveWaiting"
)

// runLifecycle is the WorkflowRun lifecycle of docs/spec/execution/runs.md,
// edge by edge, with the operation that performs each edge;
// TestRunLifecycleMatchesSpecification holds it to the specification's state
// diagram. Initialization commits pending and only admission sets running
// (RUN-R26, RUN-R28), and success is committed from running (RUN-R35).
var runLifecycle = map[[2]WorkflowRunState]runTransitionOperation{
	{RunInitializing, RunPending}:    transitionByTransitionRun,
	{RunPending, RunRunning}:         transitionByTransitionRun,
	{RunRunning, RunWaiting}:         transitionByGatewayWait,
	{RunWaiting, RunRunning}:         transitionByGatewayWait,
	{RunRunning, RunSucceeded}:       transitionByTransitionRun,
	{RunInitializing, RunFailed}:     transitionByTransitionRun,
	{RunPending, RunFailed}:          transitionByTransitionRun,
	{RunRunning, RunFailed}:          transitionByTransitionRun,
	{RunWaiting, RunFailed}:          transitionByTransitionRun,
	{RunCancelling, RunCancelled}:    transitionByTransitionRun,
	{RunInitializing, RunCancelling}: transitionByCancellation,
	{RunPending, RunCancelling}:      transitionByCancellation,
	{RunRunning, RunCancelling}:      transitionByCancellation,
	{RunWaiting, RunCancelling}:      transitionByCancellation,
	{RunFailed, RunPending}:          transitionByContinuation,
}

// ValidateRunTransition is the transition rule TransitionRun applies before
// its compare-and-set: it returns ErrInvalid for an edge TransitionRun must
// not perform, either because the lifecycle has no such edge or because
// cancellation, manual continuation or the waiting operations own it. Test
// doubles of TransitionRun apply it too, so they reject what the store
// rejects.
func ValidateRunTransition(expected, next WorkflowRunState) error {
	if runLifecycle[[2]WorkflowRunState{expected, next}] != transitionByTransitionRun {
		return invalidf("illegal WorkflowRun transition %s -> %s", expected, next)
	}
	return nil
}
