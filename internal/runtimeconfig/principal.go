package runtimeconfig

import (
	"context"
	"encoding/json"
	"errors"
	"regexp"
	"sort"
	"strconv"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

var runtimeAgentPrincipalIDPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

const (
	principalLabelsReplaceOperation = "runtime-agent.labels-replace"
	principalDeleteOperation        = "runtime-agent.delete"
)

var ErrAgentLabelNotApplicable = errors.New("RuntimeConfig label is not applicable to a Runtime Agent")

type RuntimeAgentPrincipal struct {
	RuntimeAgentID string
	Labels         []string
	LabelRevision  uint64
	CreatedBy      string
	CreatedAt      time.Time
	UpdatedBy      string
	UpdatedAt      time.Time
}

type PrincipalMutationResult struct {
	Principal *RuntimeAgentPrincipal
	Deleted   bool
	Replayed  bool
}

type principalMutationResultWire struct {
	Principal *RuntimeAgentPrincipal `json:"principal,omitempty"`
	Deleted   bool                   `json:"deleted,omitempty"`
}

type PrincipalDeletionGuard interface {
	BeginPrincipalDeletion(string) (func(), error)
}

type PrincipalRepository struct {
	db persistencepostgres.DBTX
}

func NewPrincipalRepository(db persistencepostgres.DBTX) *PrincipalRepository {
	return &PrincipalRepository{db: db}
}

func (r *PrincipalRepository) Get(
	ctx context.Context, runtimeAgentID string,
) (RuntimeAgentPrincipal, error) {
	if err := validateRuntimeAgentID(runtimeAgentID); err != nil {
		return RuntimeAgentPrincipal{}, err
	}
	return scanPrincipal(r.db.QueryRow(ctx, principalSelect+` WHERE runtime_agent_id = $1`, runtimeAgentID))
}

func (r *PrincipalRepository) List(
	ctx context.Context, afterRuntimeAgentID string, limit int,
) ([]RuntimeAgentPrincipal, error) {
	if afterRuntimeAgentID != "" {
		if err := validateRuntimeAgentID(afterRuntimeAgentID); err != nil {
			return nil, err
		}
	}
	if limit < 1 || limit > 201 {
		return nil, invalid("Runtime Agent principal page is invalid")
	}
	rows, err := r.db.Query(ctx, principalSelect+`
WHERE runtime_agent_id > $1
ORDER BY runtime_agent_id
LIMIT $2`, afterRuntimeAgentID, limit)
	if err != nil {
		return nil, errors.New("list Runtime Agent principals")
	}
	defer rows.Close()
	result := make([]RuntimeAgentPrincipal, 0, limit)
	for rows.Next() {
		principal, scanErr := scanPrincipal(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, principal)
	}
	if rows.Err() != nil {
		return nil, errors.New("iterate Runtime Agent principals")
	}
	return result, nil
}

func (r *PrincipalRepository) HasLiveAllocation(
	ctx context.Context, runtimeAgentID string,
) (bool, error) {
	if err := validateRuntimeAgentID(runtimeAgentID); err != nil {
		return false, err
	}
	var present bool
	if err := r.db.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1
    FROM stage_allocations
    WHERE runtime_agent_id = $1 AND release_completed_at IS NULL
)`, runtimeAgentID).Scan(&present); err != nil {
		return false, errors.New("inspect Runtime Agent allocation references")
	}
	return present, nil
}

func (r *PrincipalRepository) Lock(
	ctx context.Context, runtimeAgentID string,
) (RuntimeAgentPrincipal, error) {
	if err := validateRuntimeAgentID(runtimeAgentID); err != nil {
		return RuntimeAgentPrincipal{}, err
	}
	return scanPrincipal(r.db.QueryRow(ctx, principalSelect+` WHERE runtime_agent_id = $1 FOR UPDATE`, runtimeAgentID))
}

func (r *PrincipalRepository) ListByLabel(
	ctx context.Context, label string,
) ([]RuntimeAgentPrincipal, error) {
	if err := validateLabel(label); err != nil {
		return nil, invalid("Runtime Agent principal label query is invalid")
	}
	rows, err := r.db.Query(ctx, principalSelect+`
WHERE labels @> ARRAY[$1]::text[]
ORDER BY runtime_agent_id`, label)
	if err != nil {
		return nil, errors.New("list Runtime Agent principals by label")
	}
	defer rows.Close()
	result := make([]RuntimeAgentPrincipal, 0)
	for rows.Next() {
		principal, scanErr := scanPrincipal(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, principal)
	}
	if rows.Err() != nil {
		return nil, errors.New("iterate Runtime Agent principals by label")
	}
	return result, nil
}

func (r *PrincipalRepository) LockMany(
	ctx context.Context, runtimeAgentIDs []string,
) ([]RuntimeAgentPrincipal, error) {
	ordered := append([]string(nil), runtimeAgentIDs...)
	for _, runtimeAgentID := range ordered {
		if err := validateRuntimeAgentID(runtimeAgentID); err != nil {
			return nil, err
		}
	}
	sort.Strings(ordered)
	for index := 1; index < len(ordered); index++ {
		if ordered[index] == ordered[index-1] {
			return nil, invalid("Runtime Agent principal lock set contains a duplicate")
		}
	}
	result := make([]RuntimeAgentPrincipal, 0, len(ordered))
	for _, runtimeAgentID := range ordered {
		principal, err := r.Lock(ctx, runtimeAgentID)
		if err != nil {
			return nil, err
		}
		result = append(result, principal)
	}
	return result, nil
}

func (r *PrincipalRepository) Insert(
	ctx context.Context, principal RuntimeAgentPrincipal,
) (bool, error) {
	if err := validatePrincipal(principal); err != nil {
		return false, err
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO runtime_agent_principals (
    runtime_agent_id, labels, label_revision,
    created_by, created_at, updated_by, updated_at
) VALUES ($1, $2, $3::numeric, $4, $5, $6, $7)
ON CONFLICT (runtime_agent_id) DO NOTHING`,
		principal.RuntimeAgentID, principal.Labels, strconv.FormatUint(principal.LabelRevision, 10),
		principal.CreatedBy, databaseTime(principal.CreatedAt),
		principal.UpdatedBy, databaseTime(principal.UpdatedAt),
	)
	if err != nil {
		return false, classifyWrite(err)
	}
	return command.RowsAffected() == 1, nil
}

func (r *PrincipalRepository) ReplaceLabels(
	ctx context.Context,
	runtimeAgentID string,
	expectedRevision uint64,
	labels []string,
	actor string,
	at time.Time,
) (RuntimeAgentPrincipal, error) {
	if err := validateRuntimeAgentID(runtimeAgentID); err != nil || expectedRevision == 0 ||
		validateLabels(labels) != nil || !validActor(actor) || at.IsZero() {
		return RuntimeAgentPrincipal{}, invalid("Runtime Agent principal label replacement is invalid")
	}
	if expectedRevision == ^uint64(0) {
		return RuntimeAgentPrincipal{}, ErrConflict
	}
	command, err := r.db.Exec(ctx, `
UPDATE runtime_agent_principals
SET labels = $3, label_revision = label_revision + 1,
    updated_by = $4, updated_at = $5
WHERE runtime_agent_id = $1 AND label_revision = $2::numeric
  AND labels IS DISTINCT FROM $3`,
		runtimeAgentID, strconv.FormatUint(expectedRevision, 10), labels, actor, databaseTime(at),
	)
	if err != nil {
		return RuntimeAgentPrincipal{}, classifyWrite(err)
	}
	if command.RowsAffected() == 1 {
		return r.Get(ctx, runtimeAgentID)
	}
	existing, getErr := r.Get(ctx, runtimeAgentID)
	if getErr != nil {
		return RuntimeAgentPrincipal{}, getErr
	}
	if existing.LabelRevision != expectedRevision {
		return RuntimeAgentPrincipal{}, ErrPrecondition
	}
	if equalStrings(existing.Labels, labels) {
		return existing, nil
	}
	return RuntimeAgentPrincipal{}, ErrPrecondition
}

func (r *PrincipalRepository) Delete(
	ctx context.Context, runtimeAgentID string, expectedRevision uint64,
) error {
	if err := validateRuntimeAgentID(runtimeAgentID); err != nil || expectedRevision == 0 {
		return invalid("Runtime Agent principal delete is invalid")
	}
	command, err := r.db.Exec(ctx, `
DELETE FROM runtime_agent_principals
WHERE runtime_agent_id = $1 AND label_revision = $2::numeric`,
		runtimeAgentID, strconv.FormatUint(expectedRevision, 10),
	)
	if err != nil {
		return classifyWrite(err)
	}
	if command.RowsAffected() == 1 {
		return nil
	}
	existing, getErr := r.Get(ctx, runtimeAgentID)
	if errors.Is(getErr, ErrNotFound) {
		return ErrNotFound
	}
	if getErr != nil {
		return getErr
	}
	if existing.LabelRevision != expectedRevision {
		return ErrPrecondition
	}
	return ErrConflict
}

const principalSelect = `
SELECT runtime_agent_id, labels, label_revision::text,
       created_by, created_at, updated_by, updated_at
FROM runtime_agent_principals`

func scanPrincipal(row rowScanner) (RuntimeAgentPrincipal, error) {
	var result RuntimeAgentPrincipal
	var revision string
	if err := row.Scan(
		&result.RuntimeAgentID, &result.Labels, &revision,
		&result.CreatedBy, &result.CreatedAt, &result.UpdatedBy, &result.UpdatedAt,
	); errors.Is(err, pgx.ErrNoRows) {
		return RuntimeAgentPrincipal{}, ErrNotFound
	} else if err != nil {
		return RuntimeAgentPrincipal{}, errors.New("read Runtime Agent principal")
	}
	parsed, err := strconv.ParseUint(revision, 10, 64)
	if err != nil {
		return RuntimeAgentPrincipal{}, errors.New("stored Runtime Agent principal revision is invalid")
	}
	result.LabelRevision = parsed
	if err := validatePrincipal(result); err != nil {
		return RuntimeAgentPrincipal{}, errors.New("stored Runtime Agent principal failed integrity validation")
	}
	result.Labels = append([]string{}, result.Labels...)
	return result, nil
}

type PrincipalServiceOptions struct {
	Pool          *pgxpool.Pool
	DeletionGuard PrincipalDeletionGuard
	Now           func() time.Time
}

type PrincipalService struct {
	pool          *pgxpool.Pool
	repository    *PrincipalRepository
	deletionGuard PrincipalDeletionGuard
	now           func() time.Time
}

func NewPrincipalService(options PrincipalServiceOptions) (*PrincipalService, error) {
	if options.Pool == nil {
		return nil, errors.New("Runtime Agent principal service requires PostgreSQL")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	return &PrincipalService{
		pool: options.Pool, repository: NewPrincipalRepository(options.Pool),
		deletionGuard: options.DeletionGuard, now: options.Now,
	}, nil
}

// Register seeds labels only for a previously unseen certificate-key
// principal. Existing principals are returned before mutable binding lookup.
func (s *PrincipalService) Register(
	ctx context.Context, runtimeAgentID string, initialLabels []string,
) (RuntimeAgentPrincipal, error) {
	if err := validateRuntimeAgentID(runtimeAgentID); err != nil || validateLabels(initialLabels) != nil {
		return RuntimeAgentPrincipal{}, invalid("Runtime Agent principal registration is invalid")
	}
	existing, err := s.repository.Get(ctx, runtimeAgentID)
	if err == nil {
		return existing, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return RuntimeAgentPrincipal{}, err
	}

	var result RuntimeAgentPrincipal
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := validateAgentLabelSet(ctx, tx, initialLabels); err != nil {
			return err
		}
		now := s.now().UTC()
		candidate := RuntimeAgentPrincipal{
			RuntimeAgentID: runtimeAgentID, Labels: append([]string{}, initialLabels...),
			LabelRevision: 1, CreatedBy: "runtime-registration", CreatedAt: now,
			UpdatedBy: "runtime-registration", UpdatedAt: now,
		}
		repository := NewPrincipalRepository(tx)
		inserted, err := repository.Insert(ctx, candidate)
		if err != nil {
			return err
		}
		if inserted {
			result = candidate
			return nil
		}
		result, err = repository.Get(ctx, runtimeAgentID)
		return err
	})
	return result, err
}

func (s *PrincipalService) Get(
	ctx context.Context, runtimeAgentID string,
) (RuntimeAgentPrincipal, error) {
	return s.repository.Get(ctx, runtimeAgentID)
}

func (s *PrincipalService) List(
	ctx context.Context, afterRuntimeAgentID string, limit int,
) ([]RuntimeAgentPrincipal, error) {
	return s.repository.List(ctx, afterRuntimeAgentID, limit)
}

func (s *PrincipalService) RequiredRuntimeAdapters(
	ctx context.Context, labels []string,
) ([]string, error) {
	if err := validateLabels(labels); err != nil {
		return nil, err
	}
	seen := make(map[string]struct{}, 2)
	repository := NewRepository(s.pool)
	for _, label := range labels {
		binding, err := repository.GetBinding(ctx, label)
		if err != nil {
			return nil, err
		}
		version, err := repository.GetVersionByRef(ctx, binding.Ref)
		if err != nil {
			return nil, err
		}
		if version.Spec.Worker.Telemetry.Present && !version.Spec.Worker.Telemetry.Clear {
			seen["otlp-http@1"] = struct{}{}
		}
		if version.Spec.Worker.HTTPProxy.Present && !version.Spec.Worker.HTTPProxy.Clear {
			seen["http-proxy@1"] = struct{}{}
		}
		if version.Spec.Worker.Caido.Present && !version.Spec.Worker.Caido.Clear {
			seen["caido-graphql@1"] = struct{}{}
		}
	}
	result := make([]string, 0, len(seen))
	for adapter := range seen {
		result = append(result, adapter)
	}
	sort.Strings(result)
	return result, nil
}

func (s *PrincipalService) ReplaceLabels(
	ctx context.Context,
	runtimeAgentID string,
	expectedRevision uint64,
	labels []string,
	actor string,
) (RuntimeAgentPrincipal, error) {
	if err := validateRuntimeAgentID(runtimeAgentID); err != nil || expectedRevision == 0 ||
		validateLabels(labels) != nil || !validActor(actor) {
		return RuntimeAgentPrincipal{}, invalid("Runtime Agent principal label replacement is invalid")
	}
	optimistic, err := s.repository.Get(ctx, runtimeAgentID)
	if err != nil {
		return RuntimeAgentPrincipal{}, err
	}
	lockLabels := sortedUnion(optimistic.Labels, labels)
	var result RuntimeAgentPrincipal
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		// Lock every binding that can participate in either the observed or the
		// desired set before the principal row. The union itself is not a
		// configuration layer: a label being removed may legitimately conflict
		// with one being added.
		if _, err := NewRepository(tx).LockBindings(ctx, lockLabels); err != nil {
			return err
		}
		principalRepository := NewPrincipalRepository(tx)
		locked, err := principalRepository.Lock(ctx, runtimeAgentID)
		if err != nil {
			return err
		}
		if locked.LabelRevision != expectedRevision ||
			locked.LabelRevision != optimistic.LabelRevision || !equalStrings(locked.Labels, optimistic.Labels) {
			return ErrPrecondition
		}
		// The union was locked above; validate only the desired same-layer set.
		if _, err := validateAgentLabelSetFromLocked(ctx, tx, labels); err != nil {
			return err
		}
		result, err = principalRepository.ReplaceLabels(
			ctx, runtimeAgentID, expectedRevision, labels, actor, s.now().UTC(),
		)
		return err
	})
	return result, err
}

func (s *PrincipalService) ReplaceLabelsIdempotent(
	ctx context.Context,
	runtimeAgentID string,
	expectedRevision uint64,
	labels []string,
	idempotencyKey string,
	actor string,
	at time.Time,
) (PrincipalMutationResult, error) {
	requestDigest, keyDigest, err := preparePrincipalMutation(
		principalLabelsReplaceOperation, runtimeAgentID, expectedRevision, labels, idempotencyKey,
	)
	if err != nil || !validActor(actor) || at.IsZero() {
		if err != nil {
			return PrincipalMutationResult{}, err
		}
		return PrincipalMutationResult{}, invalid("Runtime Agent label mutation is invalid")
	}
	if replay, found, replayErr := readPrincipalManagementReplay(
		ctx, s.pool, keyDigest, requestDigest, principalLabelsReplaceOperation, runtimeAgentID,
	); replayErr != nil || found {
		return replay, replayErr
	}
	optimistic, err := s.repository.Get(ctx, runtimeAgentID)
	if err != nil {
		return PrincipalMutationResult{}, err
	}
	lockLabels := sortedUnion(optimistic.Labels, labels)
	var result PrincipalMutationResult
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		replayed, found, replayErr := lookupPrincipalManagementReplay(
			ctx, tx, keyDigest, requestDigest, principalLabelsReplaceOperation, runtimeAgentID,
		)
		if replayErr != nil {
			return replayErr
		}
		if found {
			result = replayed
			return nil
		}
		if _, err := NewRepository(tx).LockBindings(ctx, lockLabels); err != nil {
			return err
		}
		repository := NewPrincipalRepository(tx)
		locked, err := repository.Lock(ctx, runtimeAgentID)
		if err != nil {
			return err
		}
		if locked.LabelRevision != expectedRevision ||
			locked.LabelRevision != optimistic.LabelRevision || !equalStrings(locked.Labels, optimistic.Labels) {
			return ErrPrecondition
		}
		if _, err := validateAgentLabelSetFromLocked(ctx, tx, labels); err != nil {
			return err
		}
		updated, err := repository.ReplaceLabels(ctx, runtimeAgentID, expectedRevision, labels, actor, at)
		if err != nil {
			return err
		}
		result = PrincipalMutationResult{Principal: &updated}
		return insertPrincipalManagementResult(
			ctx, tx, keyDigest, requestDigest, principalLabelsReplaceOperation,
			runtimeAgentID, result, actor, at,
		)
	})
	return result, err
}

func (s *PrincipalService) Delete(
	ctx context.Context, runtimeAgentID string, expectedRevision uint64,
) error {
	if s.deletionGuard == nil {
		return errors.New("Runtime Agent principal deletion guard is not configured")
	}
	release, err := s.deletionGuard.BeginPrincipalDeletion(runtimeAgentID)
	if err != nil {
		return err
	}
	defer release()
	return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		repository := NewPrincipalRepository(tx)
		principal, err := repository.Lock(ctx, runtimeAgentID)
		if err != nil {
			return err
		}
		if principal.LabelRevision != expectedRevision {
			return ErrPrecondition
		}
		if len(principal.Labels) != 0 {
			return ErrPrincipalInUse
		}
		inUse, err := repository.HasLiveAllocation(ctx, runtimeAgentID)
		if err != nil {
			return err
		}
		if inUse {
			return ErrPrincipalInUse
		}
		return repository.Delete(ctx, runtimeAgentID, expectedRevision)
	})
}

func (s *PrincipalService) DeleteIdempotent(
	ctx context.Context,
	runtimeAgentID string,
	expectedRevision uint64,
	idempotencyKey string,
	actor string,
	at time.Time,
) (PrincipalMutationResult, error) {
	requestDigest, keyDigest, err := preparePrincipalMutation(
		principalDeleteOperation, runtimeAgentID, expectedRevision, []string{}, idempotencyKey,
	)
	if err != nil || !validActor(actor) || at.IsZero() {
		if err != nil {
			return PrincipalMutationResult{}, err
		}
		return PrincipalMutationResult{}, invalid("Runtime Agent principal delete is invalid")
	}
	if replay, found, replayErr := readPrincipalManagementReplay(
		ctx, s.pool, keyDigest, requestDigest, principalDeleteOperation, runtimeAgentID,
	); replayErr != nil || found {
		return replay, replayErr
	}
	if s.deletionGuard == nil {
		return PrincipalMutationResult{}, errors.New("Runtime Agent principal deletion guard is not configured")
	}
	release, err := s.deletionGuard.BeginPrincipalDeletion(runtimeAgentID)
	if err != nil {
		return PrincipalMutationResult{}, err
	}
	defer release()
	var result PrincipalMutationResult
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		replayed, found, replayErr := lookupPrincipalManagementReplay(
			ctx, tx, keyDigest, requestDigest, principalDeleteOperation, runtimeAgentID,
		)
		if replayErr != nil {
			return replayErr
		}
		if found {
			result = replayed
			return nil
		}
		repository := NewPrincipalRepository(tx)
		principal, err := repository.Lock(ctx, runtimeAgentID)
		if err != nil {
			return err
		}
		if principal.LabelRevision != expectedRevision {
			return ErrPrecondition
		}
		if len(principal.Labels) != 0 {
			return ErrPrincipalInUse
		}
		inUse, err := repository.HasLiveAllocation(ctx, runtimeAgentID)
		if err != nil {
			return err
		}
		if inUse {
			return ErrPrincipalInUse
		}
		if err := repository.Delete(ctx, runtimeAgentID, expectedRevision); err != nil {
			return err
		}
		result = PrincipalMutationResult{Deleted: true}
		return insertPrincipalManagementResult(
			ctx, tx, keyDigest, requestDigest, principalDeleteOperation,
			runtimeAgentID, result, actor, at,
		)
	})
	return result, err
}

func preparePrincipalMutation(
	kind, runtimeAgentID string,
	expectedRevision uint64,
	labels []string,
	idempotencyKey string,
) (string, string, error) {
	if (kind != principalLabelsReplaceOperation && kind != principalDeleteOperation) ||
		validateRuntimeAgentID(runtimeAgentID) != nil || expectedRevision == 0 ||
		validateLabels(labels) != nil || kind == principalDeleteOperation && len(labels) != 0 {
		return "", "", invalid("Runtime Agent principal mutation is invalid")
	}
	keyDigest, err := DigestIdempotencyKey(idempotencyKey)
	if err != nil {
		return "", "", err
	}
	request, err := json.Marshal(struct {
		Kind             string   `json:"kind"`
		RuntimeAgentID   string   `json:"runtimeAgentId"`
		ExpectedRevision uint64   `json:"expectedRevision"`
		Labels           []string `json:"labels"`
	}{kind, runtimeAgentID, expectedRevision, labels})
	if err != nil {
		return "", "", invalid("Runtime Agent principal mutation cannot be normalized")
	}
	return digest(request), keyDigest, nil
}

func lookupPrincipalManagementReplay(
	ctx context.Context, tx pgx.Tx,
	keyDigest, requestDigest, kind, runtimeAgentID string,
) (PrincipalMutationResult, bool, error) {
	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, keyDigest); err != nil {
		return PrincipalMutationResult{}, false, errors.New("lock Runtime Agent management idempotency key")
	}
	return readPrincipalManagementReplay(ctx, tx, keyDigest, requestDigest, kind, runtimeAgentID)
}

func readPrincipalManagementReplay(
	ctx context.Context, db persistencepostgres.DBTX,
	keyDigest, requestDigest, kind, runtimeAgentID string,
) (PrincipalMutationResult, bool, error) {
	var storedRequest, storedKind, storedResource, resultText string
	err := db.QueryRow(ctx, `
SELECT request_digest, operation_kind, resource_id, result::text
FROM runtime_management_operations
WHERE idempotency_key_digest = $1`, keyDigest).Scan(
		&storedRequest, &storedKind, &storedResource, &resultText,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return PrincipalMutationResult{}, false, nil
	}
	if err != nil {
		return PrincipalMutationResult{}, false, errors.New("get Runtime Agent management replay")
	}
	if storedRequest != requestDigest || storedKind != kind || storedResource != runtimeAgentID {
		return PrincipalMutationResult{}, false, ErrConflict
	}
	var wire principalMutationResultWire
	if err := json.Unmarshal([]byte(resultText), &wire); err != nil || (wire.Principal == nil) == !wire.Deleted {
		return PrincipalMutationResult{}, false, errors.New("invalid Runtime Agent management replay")
	}
	return PrincipalMutationResult{Principal: wire.Principal, Deleted: wire.Deleted, Replayed: true}, true, nil
}

func insertPrincipalManagementResult(
	ctx context.Context, tx pgx.Tx,
	keyDigest, requestDigest, kind, runtimeAgentID string,
	result PrincipalMutationResult,
	actor string,
	at time.Time,
) error {
	if !validActor(actor) || at.IsZero() || (result.Principal == nil) == !result.Deleted {
		return invalid("Runtime Agent management audit is invalid")
	}
	encoded, err := json.Marshal(principalMutationResultWire{Principal: result.Principal, Deleted: result.Deleted})
	if err != nil {
		return invalid("Runtime Agent management result cannot be encoded")
	}
	command, err := tx.Exec(ctx, `
INSERT INTO runtime_management_operations (
    idempotency_key_digest, request_digest, operation_kind, resource_id,
    result, actor_id, performed_at
) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
ON CONFLICT DO NOTHING`, keyDigest, requestDigest, kind, runtimeAgentID, string(encoded), actor, databaseTime(at))
	if err != nil {
		return errors.New("store Runtime Agent management audit")
	}
	if command.RowsAffected() != 1 {
		return ErrConflict
	}
	return nil
}

func validateAgentLabelSet(
	ctx context.Context, db persistencepostgres.DBTX, labels []string,
) ([]LayerEntry, error) {
	if err := validateLabels(labels); err != nil {
		return nil, err
	}
	repository := NewRepository(db)
	bindings, err := repository.LockBindings(ctx, labels)
	if err != nil {
		return nil, err
	}
	return agentLayerEntries(ctx, repository, bindings)
}

func validateAgentLabelSetFromLocked(
	ctx context.Context, db persistencepostgres.DBTX, labels []string,
) ([]LayerEntry, error) {
	repository := NewRepository(db)
	bindings := make([]Binding, 0, len(labels))
	for _, label := range labels {
		binding, err := repository.GetBinding(ctx, label)
		if err != nil {
			return nil, err
		}
		bindings = append(bindings, binding)
	}
	return agentLayerEntries(ctx, repository, bindings)
}

func agentLayerEntries(
	ctx context.Context, repository *Repository, bindings []Binding,
) ([]LayerEntry, error) {
	entries := make([]LayerEntry, 0, len(bindings))
	for _, binding := range bindings {
		version, err := repository.GetVersionByRef(ctx, binding.Ref)
		if err != nil {
			return nil, err
		}
		worker := version.Spec.Worker
		if !worker.LLMGateway.Present && !worker.Telemetry.Present && !worker.HTTPProxy.Present && !worker.Caido.Present {
			return nil, ErrAgentLabelNotApplicable
		}
		entries = append(entries, LayerEntry{
			Label: binding.Label, Ref: binding.Ref, Spec: Spec{Worker: worker},
		})
	}
	if _, err := MergeSameLayer(entries); err != nil {
		return nil, err
	}
	return entries, nil
}

func validateRuntimeAgentID(value string) error {
	if !runtimeAgentPrincipalIDPattern.MatchString(value) {
		return invalid("Runtime Agent principal ID is invalid")
	}
	return nil
}

func validateLabels(labels []string) error {
	if labels == nil || len(labels) > 32 {
		return invalid("Runtime Agent labels must be a non-null bounded set")
	}
	previous := ""
	for _, label := range labels {
		if validateLabel(label) != nil || label == DefaultLabel || label <= previous {
			return invalid("Runtime Agent labels must be sorted, unique, valid, and exclude default")
		}
		previous = label
	}
	return nil
}

func validatePrincipal(value RuntimeAgentPrincipal) error {
	if err := validateRuntimeAgentID(value.RuntimeAgentID); err != nil {
		return err
	}
	if err := validateLabels(value.Labels); err != nil {
		return err
	}
	if value.LabelRevision == 0 || value.CreatedBy == "" || value.UpdatedBy == "" ||
		len(value.CreatedBy) > 256 || len(value.UpdatedBy) > 256 ||
		value.CreatedAt.IsZero() || value.UpdatedAt.IsZero() {
		return invalid("Runtime Agent principal metadata is invalid")
	}
	return nil
}

func sortedUnion(groups ...[]string) []string {
	seen := make(map[string]struct{})
	for _, group := range groups {
		for _, value := range group {
			seen[value] = struct{}{}
		}
	}
	result := make([]string, 0, len(seen))
	for value := range seen {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}
