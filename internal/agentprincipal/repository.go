package agentprincipal

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

// ErrInUse refuses to delete a principal that still holds labels, has a live
// allocation or is registered with this process.
var ErrInUse = fmt.Errorf("%w: Runtime Agent principal is in use", runtimeconfig.ErrConflict)

// Principal is one Runtime Agent identity and the labels assigned to it.
type Principal struct {
	RuntimeAgentID string
	Labels         []string
	LabelRevision  uint64
	CreatedBy      string
	CreatedAt      time.Time
	UpdatedBy      string
	UpdatedAt      time.Time
}

type Repository struct {
	db persistencepostgres.DBTX
}

func NewRepository(db persistencepostgres.DBTX) *Repository {
	return &Repository{db: db}
}

func (r *Repository) Get(ctx context.Context, runtimeAgentID string) (Principal, error) {
	if err := validateRuntimeAgentID(runtimeAgentID); err != nil {
		return Principal{}, err
	}
	return scanPrincipal(r.db.QueryRow(ctx, principalSelect+` WHERE runtime_agent_id = $1`, runtimeAgentID))
}

func (r *Repository) List(ctx context.Context, afterRuntimeAgentID string, limit int) ([]Principal, error) {
	if afterRuntimeAgentID != "" {
		if err := validateRuntimeAgentID(afterRuntimeAgentID); err != nil {
			return nil, err
		}
	}
	if !pagelimit.ValidFetch(limit) {
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
	result := make([]Principal, 0, limit)
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

func (r *Repository) HasLiveAllocation(ctx context.Context, runtimeAgentID string) (bool, error) {
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

func (r *Repository) Lock(ctx context.Context, runtimeAgentID string) (Principal, error) {
	if err := validateRuntimeAgentID(runtimeAgentID); err != nil {
		return Principal{}, err
	}
	return scanPrincipal(r.db.QueryRow(ctx, principalSelect+` WHERE runtime_agent_id = $1 FOR UPDATE`, runtimeAgentID))
}

func (r *Repository) ListByLabel(ctx context.Context, label string) ([]Principal, error) {
	if err := runtimeconfig.ValidateLabel(label); err != nil {
		return nil, invalid("Runtime Agent principal label query is invalid")
	}
	rows, err := r.db.Query(ctx, principalSelect+`
WHERE labels @> ARRAY[$1]::text[]
ORDER BY runtime_agent_id`, label)
	if err != nil {
		return nil, errors.New("list Runtime Agent principals by label")
	}
	defer rows.Close()
	result := make([]Principal, 0)
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

func (r *Repository) LockMany(ctx context.Context, runtimeAgentIDs []string) ([]Principal, error) {
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
	result := make([]Principal, 0, len(ordered))
	for _, runtimeAgentID := range ordered {
		principal, err := r.Lock(ctx, runtimeAgentID)
		if err != nil {
			return nil, err
		}
		result = append(result, principal)
	}
	return result, nil
}

func (r *Repository) Insert(ctx context.Context, principal Principal) (bool, error) {
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
		principal.CreatedBy, persistencepostgres.Timestamp(principal.CreatedAt),
		principal.UpdatedBy, persistencepostgres.Timestamp(principal.UpdatedAt),
	)
	if err != nil {
		return false, classifyWrite(err)
	}
	return command.RowsAffected() == 1, nil
}

func (r *Repository) ReplaceLabels(
	ctx context.Context,
	runtimeAgentID string,
	expectedRevision uint64,
	labels []string,
	actor string,
	at time.Time,
) (Principal, error) {
	if err := validateRuntimeAgentID(runtimeAgentID); err != nil || expectedRevision == 0 ||
		runtimeconfig.ValidateAgentLabels(labels) != nil || !runtimeconfig.ValidActor(actor) || at.IsZero() {
		return Principal{}, invalid("Runtime Agent principal label replacement is invalid")
	}
	if expectedRevision == ^uint64(0) {
		return Principal{}, runtimeconfig.ErrConflict
	}
	command, err := r.db.Exec(ctx, `
UPDATE runtime_agent_principals
SET labels = $3, label_revision = label_revision + 1,
    updated_by = $4, updated_at = $5
WHERE runtime_agent_id = $1 AND label_revision = $2::numeric
  AND labels IS DISTINCT FROM $3`,
		runtimeAgentID, strconv.FormatUint(expectedRevision, 10), labels, actor, persistencepostgres.Timestamp(at),
	)
	if err != nil {
		return Principal{}, classifyWrite(err)
	}
	if command.RowsAffected() == 1 {
		return r.Get(ctx, runtimeAgentID)
	}
	existing, getErr := r.Get(ctx, runtimeAgentID)
	if getErr != nil {
		return Principal{}, getErr
	}
	if existing.LabelRevision != expectedRevision {
		return Principal{}, runtimeconfig.ErrPrecondition
	}
	if slices.Equal(existing.Labels, labels) {
		return existing, nil
	}
	return Principal{}, runtimeconfig.ErrPrecondition
}

func (r *Repository) Delete(ctx context.Context, runtimeAgentID string, expectedRevision uint64) error {
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
	if errors.Is(getErr, runtimeconfig.ErrNotFound) {
		return runtimeconfig.ErrNotFound
	}
	if getErr != nil {
		return getErr
	}
	if existing.LabelRevision != expectedRevision {
		return runtimeconfig.ErrPrecondition
	}
	return runtimeconfig.ErrConflict
}

const principalSelect = `
SELECT runtime_agent_id, labels, label_revision::text,
       created_by, created_at, updated_by, updated_at
FROM runtime_agent_principals`

type rowScanner interface{ Scan(...any) error }

func scanPrincipal(row rowScanner) (Principal, error) {
	var result Principal
	var revision string
	if err := row.Scan(
		&result.RuntimeAgentID, &result.Labels, &revision,
		&result.CreatedBy, &result.CreatedAt, &result.UpdatedBy, &result.UpdatedAt,
	); errors.Is(err, pgx.ErrNoRows) {
		return Principal{}, runtimeconfig.ErrNotFound
	} else if err != nil {
		return Principal{}, errors.New("read Runtime Agent principal")
	}
	parsed, err := strconv.ParseUint(revision, 10, 64)
	if err != nil {
		return Principal{}, errors.New("stored Runtime Agent principal revision is invalid")
	}
	result.LabelRevision = parsed
	if err := validatePrincipal(result); err != nil {
		return Principal{}, errors.New("stored Runtime Agent principal failed integrity validation")
	}
	result.Labels = append([]string{}, result.Labels...)
	return result, nil
}

func validateRuntimeAgentID(value string) error {
	if !contracts.ValidRuntimeAgentID(value) {
		return invalid("Runtime Agent principal ID is invalid")
	}
	return nil
}

func validatePrincipal(value Principal) error {
	if err := validateRuntimeAgentID(value.RuntimeAgentID); err != nil {
		return err
	}
	if err := runtimeconfig.ValidateAgentLabels(value.Labels); err != nil {
		return err
	}
	if value.LabelRevision == 0 || value.CreatedBy == "" || value.UpdatedBy == "" ||
		len(value.CreatedBy) > 256 || len(value.UpdatedBy) > 256 ||
		value.CreatedAt.IsZero() || value.UpdatedAt.IsZero() {
		return invalid("Runtime Agent principal metadata is invalid")
	}
	return nil
}

func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: %s", runtimeconfig.ErrInvalid, fmt.Sprintf(format, args...))
}

func classifyWrite(err error) error {
	if class := persistencepostgres.ConstraintError(err, runtimeconfig.ErrConflict, runtimeconfig.ErrInvalid); class != nil {
		return persistencepostgres.WrapError(class.Error(), errors.Join(class, err))
	}
	return persistencepostgres.WrapError("persist Runtime Agent principal", err)
}
