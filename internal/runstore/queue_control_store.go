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
	control, err := scanOwnerQueueControl(s.db.QueryRow(ctx, updateOwnerQueueControlSQL, params.OwnerID, params.Paused, int64(params.ExpectedRevision)))
	if errors.Is(err, pgx.ErrNoRows) {
		return OwnerQueueControl{}, ErrPrecondition
	}
	if err != nil {
		return OwnerQueueControl{}, fmt.Errorf("update owner Queue control: %w", err)
	}
	return control, nil
}

// LockRunQueueAdmission must be called on a transaction-bound PostgresStore.
// A shared owner lock orders admission with pause/resume without creating a
// public Queue control revision. The caller keeps the transaction open until
// the new StageExecution is durably committed.
func (s *PostgresStore) LockRunQueueAdmission(ctx context.Context, runID string) error {
	if err := validateOpaque("runID", runID); err != nil {
		return err
	}
	var ownerID string
	err := s.db.QueryRow(ctx, `
WITH run_owner AS MATERIALIZED (
    SELECT owner_id FROM workflow_runs WHERE run_id = $1
), owner_lock AS MATERIALIZED (
    SELECT pg_advisory_xact_lock_shared(
        hashtextextended('owner-queue-control:' || owner_id, 0)
    ) FROM run_owner
)
SELECT run_owner.owner_id FROM run_owner CROSS JOIN owner_lock`, runID).Scan(&ownerID)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	if err != nil {
		return fmt.Errorf("lock owner Queue admission: %w", err)
	}
	var paused bool
	err = s.db.QueryRow(ctx, `
SELECT COALESCE(
    (SELECT paused FROM owner_queue_controls WHERE owner_id = $1), false
)`, ownerID).Scan(&paused)
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
