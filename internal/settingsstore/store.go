package settingsstore

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type Repository interface {
	GetSchedulerSettings(context.Context) (SchedulerSettings, error)
	UpdateSchedulerSettings(context.Context, UpdateSchedulerSettingsParams) (SchedulerSettings, error)
}

type PostgresStore struct {
	db persistencepostgres.DBTX
}

var _ Repository = (*PostgresStore)(nil)

func NewPostgresStore(db persistencepostgres.DBTX) *PostgresStore {
	return &PostgresStore{db: db}
}

func (s *PostgresStore) GetSchedulerSettings(ctx context.Context) (SchedulerSettings, error) {
	if s == nil || s.db == nil {
		return SchedulerSettings{}, ErrInvariant
	}
	settings, err := scanSchedulerSettings(s.db.QueryRow(ctx, `
SELECT max_concurrent_runs, revision::text, updated_at
FROM scheduler_settings
WHERE singleton = true`))
	if errors.Is(err, pgx.ErrNoRows) {
		return SchedulerSettings{}, fmt.Errorf("%w: singleton row is missing", ErrInvariant)
	}
	if err != nil {
		return SchedulerSettings{}, fmt.Errorf("read Scheduler settings: %w", err)
	}
	return settings, nil
}

func (s *PostgresStore) UpdateSchedulerSettings(
	ctx context.Context,
	params UpdateSchedulerSettingsParams,
) (SchedulerSettings, error) {
	if !validConcurrentRuns(params.MaxConcurrentRuns) || params.ExpectedRevision == 0 ||
		params.ExpectedRevision == math.MaxUint64 {
		return SchedulerSettings{}, ErrInvalid
	}
	if s == nil || s.db == nil {
		return SchedulerSettings{}, ErrInvariant
	}
	settings, err := scanSchedulerSettings(s.db.QueryRow(ctx, `
WITH changed AS (
    UPDATE scheduler_settings
       SET max_concurrent_runs = $1,
           revision = revision + 1,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
     WHERE singleton = true
       AND revision = $2::numeric
       AND max_concurrent_runs IS DISTINCT FROM $1
    RETURNING max_concurrent_runs, revision::text, updated_at
), unchanged AS (
    SELECT max_concurrent_runs, revision::text, updated_at
      FROM scheduler_settings
     WHERE singleton = true
       AND revision = $2::numeric
       AND max_concurrent_runs IS NOT DISTINCT FROM $1
)
SELECT max_concurrent_runs, revision, updated_at FROM changed
UNION ALL
SELECT max_concurrent_runs, revision, updated_at FROM unchanged
LIMIT 1`, params.MaxConcurrentRuns, strconv.FormatUint(params.ExpectedRevision, 10)))
	if err == nil {
		return settings, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return SchedulerSettings{}, fmt.Errorf("update Scheduler settings: %w", err)
	}
	current, getErr := s.GetSchedulerSettings(ctx)
	if getErr != nil {
		return SchedulerSettings{}, getErr
	}
	if current.Revision != params.ExpectedRevision || current.MaxConcurrentRuns != params.MaxConcurrentRuns {
		return SchedulerSettings{}, ErrPrecondition
	}
	return SchedulerSettings{}, fmt.Errorf("%w: current row was not returned by exact update", ErrInvariant)
}

type rowScanner interface {
	Scan(...any) error
}

func scanSchedulerSettings(row rowScanner) (SchedulerSettings, error) {
	var result SchedulerSettings
	var revision string
	if err := row.Scan(&result.MaxConcurrentRuns, &revision, &result.UpdatedAt); err != nil {
		return SchedulerSettings{}, err
	}
	parsedRevision, err := strconv.ParseUint(revision, 10, 64)
	if err != nil || parsedRevision == 0 || !validConcurrentRuns(result.MaxConcurrentRuns) || result.UpdatedAt.IsZero() {
		return SchedulerSettings{}, ErrInvariant
	}
	result.Revision = parsedRevision
	result.UpdatedAt = result.UpdatedAt.UTC()
	return result, nil
}

func validConcurrentRuns(value int) bool {
	return value >= MinimumConcurrentRuns && value <= MaximumConcurrentRuns
}
