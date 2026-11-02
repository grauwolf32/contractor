package postgres

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Automatic retained-history admission (ART-R122 to ART-R127). A CAS write
// whose binding fits its limits after the write stays one statement
// (writeSQL). One that does not runs here: it locks the binding, plans the
// oldest unprotected revisions to remove within the transaction's work
// bound, and only then changes anything, so a refused write changes nothing.
// The eligible partial index (hold_count = 0) is the continuation: a later
// attempt or a released hold is found again by the next scan in creation
// order, and no cursor, queue or lease exists.

// historyWorkSetting holds the history work of the current transaction as
// "examined,removed", so the bound of LIM-ART-10 spans every binding the
// transaction admits, including writes of a caller's multi-write transaction.
// It is transaction-local and rolls back with a savepoint.
const historyWorkSetting = "contractor.artifact_history_work"

// historyRetentionSetting enables the trusted retention exception of the
// revision, version and blob delete triggers for the statements that remove
// history.
const historyRetentionSetting = "contractor.artifact_retention"

// historyBinding is a locked binding under admission.
type historyBinding struct {
	current           string
	frozen            bool
	retainedRevisions int64
	retainedBytes     int64
	readCommittedTx   bool
}

// historyVictim is one planned removal.
type historyVictim struct {
	revision  string
	versionID string
	size      int64
}

type historyWork struct {
	examined, removed int
}

// errHistoryRaceLost reports a locked victim that the delete did not
// remove, which the victim lock rules out.
var errHistoryRaceLost = errors.New("artifact history victim acquired a hold before removal")

// advanceWithHistory is the CAS write of a binding whose fast path refused
// it. It returns the stored revision row, a write precondition refusal, a
// *HistoryProtectedError or a *HistoryMaintenanceError.
func (r *Repository) advanceWithHistory(
	ctx context.Context, scope artifacts.Scope, target artifacts.ArtifactRef, expectedRevision string, write writeStatement,
) (writeRow, error) {
	policy := r.history
	var stored writeRow
	err := persistencepostgres.Within(ctx, r.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		stored = writeRow{}
		binding, found, err := lockHistoryBinding(ctx, tx, scope, target)
		if err != nil {
			return err
		}
		var current *artifacts.Metadata
		if found {
			revision := binding.current
			current = &artifacts.Metadata{Ref: artifacts.ArtifactRef{Namespace: target.Namespace, Name: target.Name, Revision: &revision},
				Frozen: binding.frozen}
		}
		if refusal := artifacts.WritePrecondition(target, current, &expectedRevision); refusal != nil {
			return refusal
		}
		work, err := readHistoryWork(ctx, tx)
		if err != nil {
			return err
		}
		plan, err := planHistory(ctx, tx, scope, target, binding, policy, 1, write.size, "", &work)
		if err != nil {
			return err
		}
		if !plan.fits {
			// The examined candidates count against the transaction even
			// when the write is refused.
			if err := writeHistoryWork(ctx, tx, work); err != nil {
				return err
			}
			return plan.refusal(scope, target, binding, policy, write.size)
		}
		victims, err := lockHistoryVictims(ctx, tx, scope, target, plan.victims)
		if err != nil {
			return err
		}
		if len(victims) != len(plan.victims) {
			// A holder protected a planned victim after the plan read it.
			return &artifacts.HistoryMaintenanceError{Scope: scope, Ref: target, Size: write.size}
		}
		if len(victims) > 0 && !binding.readCommittedTx {
			return fmt.Errorf("artifact history retention requires READ COMMITTED isolation")
		}
		stored, err = write.exec(ctx, tx, int64(len(victims)), plan.bytes)
		if errors.Is(err, pgx.ErrNoRows) {
			// The binding is locked, its precondition holds and the history
			// fits, so only a blob whose digest names other content remains.
			return fmt.Errorf("%w: blob digest names other content", blob.ErrIntegrity)
		}
		if err != nil {
			return err
		}
		if err := r.removeHistoryVictims(ctx, tx, scope, target, victims); err != nil {
			return err
		}
		work.removed += len(victims)
		return writeHistoryWork(ctx, tx, work)
	})
	var pending *artifacts.HistoryMaintenanceError
	if errors.As(err, &pending) {
		if _, owned := r.db.(*pgxpool.Pool); owned {
			// The write's own transaction rolled back; commit one bounded
			// maintenance pass for this binding before answering (ART-R127).
			return writeRow{}, r.maintain(ctx, pending)
		}
	}
	return stored, err
}

// MaintainHistory commits the maintenance pass that a write in the caller's
// transaction could not: a caller that owns the transaction in which a write
// returned *HistoryMaintenanceError rolls that transaction back first and
// then passes the error here, on a repository of the pool. It returns the
// error to report, which is the same pending outcome after the pass, or the
// pass's own failure. Any other error is returned unchanged.
func (r *Repository) MaintainHistory(ctx context.Context, err error) error {
	var pending *artifacts.HistoryMaintenanceError
	if !errors.As(err, &pending) || pending.Maintained {
		return err
	}
	if _, owned := r.db.(*pgxpool.Pool); !owned {
		return fmt.Errorf("artifact history maintenance needs the PostgreSQL pool: %w", err)
	}
	return r.maintain(ctx, pending)
}

// maintain commits one maintenance pass for the binding of pending and
// returns the pending outcome with what it removed. A failed pass leaves the
// outcome pending and carries the failure as its cause: the write itself
// stored nothing either way.
func (r *Repository) maintain(ctx context.Context, pending *artifacts.HistoryMaintenanceError) error {
	removed, err := r.maintainHistory(ctx, pending.Scope, pending.Ref, pending.Size)
	if err != nil {
		return &artifacts.HistoryMaintenanceError{Scope: pending.Scope, Ref: pending.Ref, Size: pending.Size, Cause: err}
	}
	return &artifacts.HistoryMaintenanceError{
		Scope: pending.Scope, Ref: pending.Ref, Size: pending.Size, Removed: removed, Maintained: true,
	}
}

// maintainHistory commits one bounded pass that removes the oldest
// unprotected history of one binding, never its current revision, towards
// room for a revision of size bytes. It reports how many revisions it removed.
func (r *Repository) maintainHistory(
	ctx context.Context, scope artifacts.Scope, target artifacts.ArtifactRef, size int64,
) (int, error) {
	policy := r.history
	removed := 0
	err := persistencepostgres.Within(ctx, r.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		removed = 0
		binding, found, err := lockHistoryBinding(ctx, tx, scope, target)
		if err != nil || !found || binding.frozen {
			return err
		}
		if !binding.readCommittedTx {
			return fmt.Errorf("artifact history retention requires READ COMMITTED isolation")
		}
		work := historyWork{}
		plan, err := planHistory(ctx, tx, scope, target, binding, policy, 1, size, binding.current, &work)
		if err != nil {
			return err
		}
		victims, err := lockHistoryVictims(ctx, tx, scope, target, plan.victims)
		if err != nil || len(victims) == 0 {
			return err
		}
		var bytes int64
		for _, victim := range victims {
			bytes += victim.size
		}
		if err := r.removeHistoryVictims(ctx, tx, scope, target, victims); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, releaseHistoryCountersSQL,
			scope.Kind(), scope.ID(), target.Namespace, target.Name, int64(len(victims)), bytes); err != nil {
			return storeError("release artifact history accounting", err, nil)
		}
		removed = len(victims)
		return nil
	})
	return removed, err
}

func lockHistoryBinding(
	ctx context.Context, tx pgx.Tx, scope artifacts.Scope, target artifacts.ArtifactRef,
) (historyBinding, bool, error) {
	var binding historyBinding
	var isolation string
	err := tx.QueryRow(ctx, lockHistoryBindingSQL, scope.Kind(), scope.ID(), target.Namespace, target.Name).Scan(
		&binding.current, &binding.frozen, &binding.retainedRevisions, &binding.retainedBytes, &isolation)
	if errors.Is(err, pgx.ErrNoRows) {
		return historyBinding{}, false, nil
	}
	if err != nil {
		return historyBinding{}, false, fmt.Errorf("lock artifact binding for history admission: %w", err)
	}
	binding.readCommittedTx = isolation == "read committed"
	return binding, true, nil
}

// historyPlan is the outcome of planning one admission.
type historyPlan struct {
	victims []historyVictim
	bytes   int64
	// fits reports whether removing victims makes room.
	fits bool
	// exhausted reports that the scan read every unprotected revision.
	exhausted bool
}

// refusal returns the error of a plan that does not fit: protected when the
// scan proved there is nothing left to remove, pending otherwise.
func (p historyPlan) refusal(
	scope artifacts.Scope, target artifacts.ArtifactRef, binding historyBinding, policy HistoryPolicy, size int64,
) error {
	if p.exhausted {
		return &artifacts.HistoryProtectedError{
			Ref:               artifacts.ArtifactRef{Namespace: target.Namespace, Name: target.Name},
			RetainedRevisions: binding.retainedRevisions + 1 - int64(len(p.victims)),
			RetainedBytes:     binding.retainedBytes + size - p.bytes,
			MaxRevisions:      policy.MaxRevisions(), MaxBytes: policy.MaxBytes(),
		}
	}
	return &artifacts.HistoryMaintenanceError{Scope: scope, Ref: artifacts.ArtifactRef{Namespace: target.Namespace, Name: target.Name},
		Size: size}
}

// planHistory selects the oldest unprotected revisions of the binding whose
// removal leaves room for addRevisions more revisions of addBytes, skipping
// keep (the current revision during maintenance). It examines at most what
// is left of the transaction's candidate budget, plans at most what is left
// of its removal budget, and at most MaxHistoryReclaimBytesPerPass of victim
// payload, except that the first victim always fits. It reads in chunks that
// start at the revisions still needed and double, so a write that needs one
// removal examines one candidate.
func planHistory(
	ctx context.Context, tx pgx.Tx, scope artifacts.Scope, target artifacts.ArtifactRef, binding historyBinding,
	policy HistoryPolicy, addRevisions, addBytes int64, keep string, work *historyWork,
) (historyPlan, error) {
	needRevisions := binding.retainedRevisions + addRevisions - policy.MaxRevisions()
	needBytes := binding.retainedBytes + addBytes - policy.MaxBytes()
	plan := historyPlan{fits: needRevisions <= 0 && needBytes <= 0}
	afterCreated := pgtype.Timestamptz{InfinityModifier: pgtype.NegativeInfinity, Valid: true}
	afterRevision := ""
	chunk := max(needRevisions, 1)
	for !plan.fits {
		examine := min(chunk, int64(MaxHistoryCandidatesPerTransaction-work.examined))
		if examine <= 0 || work.removed+len(plan.victims) >= MaxHistoryRemovalsPerTransaction {
			return plan, nil
		}
		rows, err := tx.Query(ctx, historyCandidatesSQL, scope.Kind(), scope.ID(), target.Namespace, target.Name,
			afterCreated, afterRevision, examine)
		if err != nil {
			return historyPlan{}, fmt.Errorf("list artifact history candidates: %w", err)
		}
		var candidates []historyVictim
		for rows.Next() {
			var candidate historyVictim
			if err := rows.Scan(&candidate.revision, &afterCreated, &candidate.versionID, &candidate.size); err != nil {
				rows.Close()
				return historyPlan{}, fmt.Errorf("scan artifact history candidate: %w", err)
			}
			afterRevision = candidate.revision
			candidates = append(candidates, candidate)
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return historyPlan{}, fmt.Errorf("list artifact history candidates: %w", err)
		}
		work.examined += len(candidates)
		for _, candidate := range candidates {
			if candidate.revision == keep {
				continue
			}
			if work.removed+len(plan.victims) >= MaxHistoryRemovalsPerTransaction ||
				(len(plan.victims) > 0 && plan.bytes+candidate.size > MaxHistoryReclaimBytesPerPass) {
				return plan, nil
			}
			plan.victims = append(plan.victims, candidate)
			plan.bytes += candidate.size
			needRevisions--
			needBytes -= candidate.size
			if needRevisions <= 0 && needBytes <= 0 {
				plan.fits = true
				return plan, nil
			}
		}
		if int64(len(candidates)) < examine {
			plan.exhausted = true
			return plan, nil
		}
		chunk *= 2
	}
	return plan, nil
}

// lockHistoryVictims locks the planned victims in key order and returns
// those still unprotected, in plan order.
func lockHistoryVictims(
	ctx context.Context, tx pgx.Tx, scope artifacts.Scope, target artifacts.ArtifactRef, planned []historyVictim,
) ([]historyVictim, error) {
	if len(planned) == 0 {
		return nil, nil
	}
	revisions := make([]string, len(planned))
	for index, victim := range planned {
		revisions[index] = victim.revision
	}
	rows, err := tx.Query(ctx, lockHistoryVictimsSQL, scope.Kind(), scope.ID(), target.Namespace, target.Name, revisions)
	if err != nil {
		return nil, fmt.Errorf("lock artifact history victims: %w", err)
	}
	locked, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		return nil, fmt.Errorf("lock artifact history victims: %w", err)
	}
	unprotected := make(map[string]bool, len(locked))
	for _, revision := range locked {
		unprotected[revision] = true
	}
	victims := make([]historyVictim, 0, len(locked))
	for _, victim := range planned {
		if unprotected[victim.revision] {
			victims = append(victims, victim)
		}
	}
	return victims, nil
}

// removeHistoryVictims deletes the locked victims under the retention
// exception and collects the versions and blobs they leave unreferenced.
// Lineage is never cut: a revision at either end of an edge is held.
func (s *Storage) removeHistoryVictims(
	ctx context.Context, tx pgx.Tx, scope artifacts.Scope, target artifacts.ArtifactRef, victims []historyVictim,
) error {
	if len(victims) == 0 {
		return nil
	}
	revisions := make([]string, len(victims))
	for index, victim := range victims {
		revisions[index] = victim.revision
	}
	if _, err := tx.Exec(ctx, `SELECT set_config($1, 'evict', true)`, historyRetentionSetting); err != nil {
		return fmt.Errorf("enable artifact history retention: %w", err)
	}
	rows, err := tx.Query(ctx, deleteHistoryVictimsSQL, scope.Kind(), scope.ID(), target.Namespace, target.Name, revisions)
	if err != nil {
		return fmt.Errorf("remove artifact history: %w", err)
	}
	versionIDs, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		return fmt.Errorf("remove artifact history: %w", err)
	}
	if len(versionIDs) != len(victims) {
		return fmt.Errorf("%w: removed %d of %d locked revisions", errHistoryRaceLost, len(versionIDs), len(victims))
	}
	if err := s.collectUnreferencedVersions(ctx, tx, versionIDs); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, `SELECT set_config($1, '', true)`, historyRetentionSetting); err != nil {
		return fmt.Errorf("disable artifact history retention: %w", err)
	}
	return nil
}

func readHistoryWork(ctx context.Context, tx pgx.Tx) (historyWork, error) {
	var value string
	if err := tx.QueryRow(ctx, `SELECT COALESCE(current_setting($1, true), '')`, historyWorkSetting).
		Scan(&value); err != nil {
		return historyWork{}, fmt.Errorf("read artifact history work: %w", err)
	}
	if value == "" {
		return historyWork{}, nil
	}
	examined, removed, ok := strings.Cut(value, ",")
	work := historyWork{}
	var examinedErr, removedErr error
	work.examined, examinedErr = strconv.Atoi(examined)
	work.removed, removedErr = strconv.Atoi(removed)
	if !ok || examinedErr != nil || removedErr != nil || work.examined < 0 || work.removed < 0 {
		return historyWork{}, fmt.Errorf("artifact history work %q is malformed", value)
	}
	return work, nil
}

func writeHistoryWork(ctx context.Context, tx pgx.Tx, work historyWork) error {
	if _, err := tx.Exec(ctx, `SELECT set_config($1, $2, true)`, historyWorkSetting,
		strconv.Itoa(work.examined)+","+strconv.Itoa(work.removed)); err != nil {
		return fmt.Errorf("record artifact history work: %w", err)
	}
	return nil
}

// revisionRemoved reports whether the exact revision of a binding no longer
// exists. A reader that found its metadata and then missed its filesystem
// content uses it to tell a revision that retention removed in between, which
// is not found, from retained content that is missing (ART-R124).
func (r *Repository) revisionRemoved(ctx context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef, revision string) (bool, error) {
	var exists bool
	if err := r.db.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3 AND name = $4 AND revision = $5)`,
		scope.Kind(), scope.ID(), ref.Namespace, ref.Name, revision).Scan(&exists); err != nil {
		return false, fmt.Errorf("recheck artifact revision: %w", err)
	}
	return !exists, nil
}
