package runstore

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/jackc/pgx/v5"
)

func (s *PostgresStore) GetOwnerQueueControl(
	ctx context.Context,
	ownerID string,
) (OwnerQueueControl, error) {
	if err := validateQueueControlOwner(ownerID); err != nil {
		return OwnerQueueControl{}, err
	}
	control, err := scanOwnerQueueControl(s.db.QueryRow(ctx, `
SELECT owner_id, paused, revision, updated_at
FROM owner_queue_controls
WHERE owner_id = $1`, ownerID))
	if errors.Is(err, pgx.ErrNoRows) {
		return OwnerQueueControl{OwnerID: ownerID}, nil
	}
	if err != nil {
		return OwnerQueueControl{}, fmt.Errorf("read owner Queue control: %w", err)
	}
	return control, nil
}

func (s *PostgresStore) UpdateOwnerQueueControl(
	ctx context.Context,
	params UpdateOwnerQueueControlParams,
) (OwnerQueueControl, error) {
	if err := validateQueueControlOwner(params.OwnerID); err != nil {
		return OwnerQueueControl{}, err
	}
	if params.ExpectedRevision > math.MaxInt64 {
		return OwnerQueueControl{}, invalidf("owner Queue control revision is invalid")
	}
	control, err := scanOwnerQueueControl(s.db.QueryRow(ctx, `
WITH inserted AS (
    INSERT INTO owner_queue_controls (owner_id, paused, revision)
    SELECT $1, $2, 1
    WHERE $3::bigint = 0
    ON CONFLICT DO NOTHING
    RETURNING owner_id, paused, revision, updated_at
), updated AS (
    UPDATE owner_queue_controls
    SET paused = $2,
        revision = revision + 1,
        updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
    WHERE owner_id = $1
      AND revision = $3
      AND $3::bigint > 0
      AND paused IS DISTINCT FROM $2
    RETURNING owner_id, paused, revision, updated_at
), unchanged AS (
    SELECT owner_id, paused, revision, updated_at
    FROM owner_queue_controls
    WHERE owner_id = $1
      AND revision = $3
      AND $3::bigint > 0
      AND paused IS NOT DISTINCT FROM $2
)
SELECT owner_id, paused, revision, updated_at FROM inserted
UNION ALL
SELECT owner_id, paused, revision, updated_at FROM updated
UNION ALL
SELECT owner_id, paused, revision, updated_at FROM unchanged
LIMIT 1`, params.OwnerID, params.Paused, int64(params.ExpectedRevision)))
	if errors.Is(err, pgx.ErrNoRows) {
		return OwnerQueueControl{}, ErrPrecondition
	}
	if err != nil {
		return OwnerQueueControl{}, fmt.Errorf("update owner Queue control: %w", err)
	}
	return control, nil
}

// LockRunQueueAdmission must be called on a transaction-bound PostgresStore.
// Materializing and locking the owner's control row establishes the ordering
// shared with pause/resume updates. The caller must keep the transaction open
// until the new StageExecution is durably committed.
func (s *PostgresStore) LockRunQueueAdmission(ctx context.Context, runID string) error {
	if err := validateOpaque("runID", runID); err != nil {
		return err
	}
	if _, err := s.db.Exec(ctx, `
INSERT INTO owner_queue_controls (owner_id, paused)
SELECT owner_id, false
FROM workflow_runs
WHERE run_id = $1
ON CONFLICT DO NOTHING`, runID); err != nil {
		return fmt.Errorf("materialize owner Queue control: %w", err)
	}
	var paused bool
	err := s.db.QueryRow(ctx, `
SELECT control.paused
FROM workflow_runs AS run
JOIN owner_queue_controls AS control ON control.owner_id = run.owner_id
WHERE run.run_id = $1
FOR UPDATE OF control`, runID).Scan(&paused)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	if err != nil {
		return fmt.Errorf("lock owner Queue admission: %w", err)
	}
	if paused {
		return ErrQueuePaused
	}
	return nil
}

type queueControlScanner interface{ Scan(...any) error }

func scanOwnerQueueControl(row queueControlScanner) (OwnerQueueControl, error) {
	var control OwnerQueueControl
	var revision int64
	err := row.Scan(&control.OwnerID, &control.Paused, &revision, &control.UpdatedAt)
	if err == nil {
		if revision <= 0 {
			return OwnerQueueControl{}, errors.New("stored owner Queue control revision is invalid")
		}
		control.Revision = uint64(revision)
		control.UpdatedAt = control.UpdatedAt.UTC()
	}
	return control, err
}

func validateQueueControlOwner(ownerID string) error {
	if strings.TrimSpace(ownerID) == "" || len(ownerID) > 256 || strings.IndexByte(ownerID, 0) >= 0 {
		return invalidf("owner Queue control identity is invalid")
	}
	return nil
}
