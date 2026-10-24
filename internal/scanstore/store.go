package scanstore

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
)

type Store struct{ db persistencepostgres.DBTX }

func New(db persistencepostgres.DBTX) *Store { return &Store{db: db} }

func scanSweep(row pgx.Row) (Sweep, error) {
	var s Sweep
	err := row.Scan(&s.SweepID, &s.Workflow, &s.SnapshotDigest, &s.State, &s.Counts.Total, &s.Counts.Pending, &s.Counts.Admitted, &s.Counts.Skipped, &s.Counts.Failed, &s.Counts.Cancelled, &s.Counts.Unsettled, &s.CreatedAt, &s.SettledAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return s, ErrNotFound
	}
	if err != nil {
		return s, persistencepostgres.WrapError("read scan sweep", err)
	}
	if !contentdigest.Valid(s.SnapshotDigest) || s.Counts.Total < 1 || s.Counts.Total > MaxProjects || s.Counts.Pending+s.Counts.Admitted+s.Counts.Skipped+s.Counts.Failed+s.Counts.Cancelled != s.Counts.Total {
		return Sweep{}, fmt.Errorf("invalid stored sweep")
	}
	return s, nil
}

func (s *Store) Get(ctx context.Context, owner, id string) (Sweep, error) {
	return scanSweep(s.db.QueryRow(ctx, `SELECT `+sweepColumns+` FROM scan_sweeps WHERE owner_id=$1 AND sweep_id=$2 AND NOT purging`, owner, id))
}

func (s *Store) List(ctx context.Context, owner, after string, limit int) ([]Sweep, error) {
	if !pagelimit.ValidFetch(limit) {
		return nil, ErrInvalid
	}
	rows, err := s.db.Query(ctx, `SELECT `+sweepColumns+` FROM scan_sweeps WHERE owner_id=$1 AND sweep_id>$2 AND NOT purging ORDER BY sweep_id LIMIT $3`, owner, after, limit)
	if err != nil {
		return nil, persistencepostgres.WrapError("list scan sweeps", err)
	}
	defer rows.Close()
	result := make([]Sweep, 0, limit)
	for rows.Next() {
		item, err := scanSweep(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, item)
	}
	return result, rows.Err()
}

func (s *Store) Snapshot(ctx context.Context, id string) (Snapshot, error) {
	var data []byte
	var owner string
	err := s.db.QueryRow(ctx, `SELECT snapshot,owner_id FROM scan_sweeps WHERE sweep_id=$1`, id).Scan(&data, &owner)
	if errors.Is(err, pgx.ErrNoRows) {
		return Snapshot{}, ErrNotFound
	}
	if err != nil {
		return Snapshot{}, err
	}
	var result Snapshot
	if len(data) > MaxSnapshotBytes || strictjson.Decode(data, &result) != nil || result.SchemaVersion != 1 || result.OwnerID != owner || !contentdigest.Valid(result.WorkflowDigest) || result.Runtime.Validate() != nil {
		return Snapshot{OwnerID: owner}, fmt.Errorf("%w: invalid stored sweep snapshot", ErrInvalid)
	}
	if resolved.ValidateWorkflowGraph(result.Workflow) != nil {
		return Snapshot{OwnerID: owner}, ErrInvalid
	}
	if err := contentdigest.CheckPinnedSelection(result.WorkflowDigest, result.Workflow); err != nil {
		return Snapshot{OwnerID: owner}, ErrInvalid
	}
	if result.Rulepack != nil && (result.Rulepack.Source.Validate() != nil || result.Rulepack.Retained.ValidateExact() != nil || result.Rulepack.Retained.Namespace != artifactpolicy.RunSystemNamespace) {
		return Snapshot{OwnerID: owner}, ErrInvalid
	}
	return result, nil
}

func (s *Store) Insert(ctx context.Context, id, owner, workflow, digest string, snapshot []byte, members []Member) error {
	if len(members) < 1 || len(members) > MaxProjects || len(snapshot) > MaxSnapshotBytes || !contentdigest.Valid(digest) {
		return ErrInvalid
	}
	pending := 0
	totalBytes := len(snapshot)
	batch := &pgx.Batch{}
	for i, m := range members {
		if i != m.Ordinal || len(m.Child) > MaxChildBytes || !contentdigest.Valid(m.Digest) || (m.State != "pending" && m.State != "skipped") {
			return ErrInvalid
		}
		totalBytes += len(m.Child)
		if totalBytes > MaxSnapshotBytes {
			return ErrInvalid
		}
		if m.State == "pending" {
			pending++
		}
	}
	batch.Queue(insertSweepSQL, id, owner, workflow, snapshot, digest, len(members), pending, len(members)-pending)
	for _, m := range members {
		batch.Queue(`INSERT INTO scan_sweep_members(sweep_id,ordinal,project_id,target,child,child_key,child_digest,state,reason,settled) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)`, id, m.Ordinal, m.ProjectID, m.Target, []byte(m.Child), m.Key, m.Digest, m.State, m.Reason, m.State == "skipped")
	}
	// The caller owns a transaction; SendBatch never acquires another connection.
	sender, ok := s.db.(interface {
		SendBatch(context.Context, *pgx.Batch) pgx.BatchResults
	})
	if !ok {
		return fmt.Errorf("sweep insertion requires a transaction")
	}
	results := sender.SendBatch(ctx, batch)
	if err := results.Close(); err != nil {
		return persistencepostgres.WrapError("insert scan sweep", err)
	}
	return nil
}

func scanMembers(rows pgx.Rows) ([]Member, error) {
	defer rows.Close()
	result := make([]Member, 0)
	for rows.Next() {
		var m Member
		if err := rows.Scan(&m.Ordinal, &m.ProjectID, &m.Target, &m.State, &m.Reason, &m.RunID, &m.RunState, &m.Settled, &m.Child, &m.Key, &m.Digest, &m.Attempts); err != nil {
			return nil, err
		}
		if len(m.Child) > MaxChildBytes || !contentdigest.Valid(m.Digest) {
			return nil, fmt.Errorf("invalid stored sweep child")
		}
		result = append(result, m)
	}
	return result, rows.Err()
}

func (s *Store) Members(ctx context.Context, owner, id string, after, limit int) ([]Member, error) {
	if !pagelimit.ValidFetch(limit) || after < -1 {
		return nil, ErrInvalid
	}
	if _, err := s.Get(ctx, owner, id); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, membersSQL, id, after, limit)
	if err != nil {
		return nil, err
	}
	return scanMembers(rows)
}

func (s *Store) Work(ctx context.Context, claim persistencepostgres.ControllerClaim) ([]Member, error) {
	if err := s.Lock(ctx, claim); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, workSQL, claim.SubjectID, TickMembers)
	if err != nil {
		return nil, err
	}
	return scanMembers(rows)
}

func (s *Store) Claim(ctx context.Context, holder string, lease time.Duration) (persistencepostgres.ControllerClaim, bool, error) {
	if holder == "" || !persistencepostgres.ValidControllerLease(lease) {
		return persistencepostgres.ControllerClaim{}, false, ErrInvalid
	}
	c, err := persistencepostgres.ScanControllerClaim(s.db.QueryRow(ctx, claimSQL, holder, lease.Milliseconds()))
	if errors.Is(err, pgx.ErrNoRows) {
		return c, false, nil
	}
	return c, err == nil, err
}
func (s *Store) Lock(ctx context.Context, c persistencepostgres.ControllerClaim) error {
	return claims.Lock(ctx, s.db, c.SubjectID, c.HolderID, c.Epoch)
}
func (s *Store) Release(ctx context.Context, c persistencepostgres.ControllerClaim) error {
	// Rotation is fenced by the exact holder. Release accepts its expired lease.
	_, err := s.db.Exec(ctx, `UPDATE scan_sweeps SET next_tick_at=clock_timestamp()+interval '1 second' WHERE sweep_id=$1 AND holder_id=$2 AND epoch=$3`, c.SubjectID, c.HolderID, c.Epoch)
	if err != nil {
		return err
	}
	return claims.Release(ctx, s.db, c.SubjectID, c.HolderID, c.Epoch)
}

// Record must run in the same transaction as child creation when runID is new.
// The parent claim lock serializes it with cancellation and physical purge.
func (s *Store) Record(ctx context.Context, c persistencepostgres.ControllerClaim, ordinal int, state, reason string, runID *string, settled bool, retrySeconds int) error {
	if err := s.Lock(ctx, c); err != nil {
		return err
	}
	if retrySeconds < 0 || retrySeconds > 300 {
		return ErrInvalid
	}
	var oldState string
	var oldSettled bool
	var oldRunID *string
	err := s.db.QueryRow(ctx, `SELECT state,settled,run_id FROM scan_sweep_members WHERE sweep_id=$1 AND ordinal=$2 FOR UPDATE`, c.SubjectID, ordinal).Scan(&oldState, &oldSettled, &oldRunID)
	if err != nil {
		return err
	}
	if oldSettled {
		return nil
	}
	if oldState == "admitted" && state != "admitted" {
		return nil
	} // A lost acknowledgement already committed the authoritative fact.
	delta := map[string]int{"pending": 0, "admitted": 0, "skipped": 0, "failed": 0, "cancelled": 0}
	if _, ok := delta[state]; !ok {
		return ErrInvalid
	}
	delta[oldState]--
	delta[state]++
	unsettledDelta := 0
	if settled {
		unsettledDelta = -1
	}
	_, err = s.db.Exec(ctx, `UPDATE scan_sweep_members SET state=$3,reason=$4,run_id=$5,settled=$6,attempts=attempts+CASE WHEN $3='pending' THEN 1 ELSE 0 END,next_attempt_at=clock_timestamp()+$7*interval '1 second' WHERE sweep_id=$1 AND ordinal=$2`, c.SubjectID, ordinal, state, reason, runID, settled, retrySeconds)
	if err != nil {
		return err
	}
	_, err = s.db.Exec(ctx, advanceCountersSQL, c.SubjectID, delta["pending"], delta["admitted"], delta["skipped"], delta["failed"], delta["cancelled"], unsettledDelta)
	if err != nil {
		return persistencepostgres.WrapError("advance scan sweep counters", err)
	}
	return nil
}

func (s *Store) Cancel(ctx context.Context, owner, id string) (Sweep, error) {
	_, err := s.db.Exec(ctx, `UPDATE scan_sweeps SET state='cancelling',next_tick_at=clock_timestamp() WHERE owner_id=$1 AND sweep_id=$2 AND NOT purging AND state='active'`, owner, id)
	if err != nil {
		return Sweep{}, err
	}
	return s.Get(ctx, owner, id)
}

// DeleteTerminal takes the parent lock, joining every fenced child mutation.
// No asynchronous child admission or snapshot read survives this transaction.
func (s *Store) DeleteTerminal(ctx context.Context, owner, id string) error {
	var state string
	var live bool
	err := s.db.QueryRow(ctx, `SELECT state,holder_id IS NOT NULL AND expires_at>clock_timestamp() FROM scan_sweeps WHERE owner_id=$1 AND sweep_id=$2 FOR UPDATE`, owner, id).Scan(&state, &live)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	if err != nil {
		return err
	}
	if live || (state != "completed" && state != "cancelled") {
		return ErrConflict
	}
	_, err = s.db.Exec(ctx, `DELETE FROM scan_sweeps WHERE sweep_id=$1`, id)
	return err
}

// Cancelling reads the intent while the caller holds the parent's claim lock.
func (s *Store) Cancelling(ctx context.Context, id string, result *bool) error {
	return s.db.QueryRow(ctx, `SELECT state='cancelling' FROM scan_sweeps WHERE sweep_id=$1`, id).Scan(result)
}
