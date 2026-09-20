package evalstore

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
)

type DirtyMember struct {
	MemberID string
	Revision int64
}

func (s *Store) LockCollection(ctx context.Context, scope Scope, id string, claim Claim) (Experiment, error) {
	if _, err := s.project(ctx, scope, true); err != nil {
		return Experiment{}, err
	}
	e, err := s.locked(ctx, scope, id)
	if err != nil {
		return e, err
	}
	if e.DeletionRequestedAt != nil {
		return e, evaldomain.Failure("eval_project_deleting")
	}
	return e, s.checkClaim(ctx, id, claim)
}
func (s *Store) ProjectionRevision(ctx context.Context, owner, id, member string) (int64, error) {
	var revision int64
	err := s.db.QueryRow(ctx, `
SELECT p.revision
FROM eval_member_projections p
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id = $1
    AND e.experiment_id = $2
    AND p.member_id = $3
`, owner, id, member).Scan(&revision)
	return revision, normalize(err)
}

func (s *Store) DirtyMembers(ctx context.Context, owner, id string, limit int) ([]DirtyMember, error) {
	if limit < 1 || limit > evaldomain.MaxPageSize {
		return nil, evaldomain.Failure("eval_invalid")
	}
	rows, err := s.db.Query(ctx, `
SELECT p.member_id,p.revision
FROM eval_member_projections p
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id = $1
    AND e.experiment_id = $2
    AND p.revision <> p.projected_revision
ORDER BY p.checked_at NULLS FIRST,p.member_id
LIMIT $3
`, owner, id, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []DirtyMember{}
	for rows.Next() {
		var m DirtyMember
		if err = rows.Scan(&m.MemberID, &m.Revision); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}
func (s *Store) ProjectMember(ctx context.Context, scope Scope, id, member string, claim Claim, revision int64, view evaldomain.MemberView, complete bool, observation json.RawMessage) error {
	if _, err := s.project(ctx, scope, true); err != nil {
		return err
	}
	e, err := s.locked(ctx, scope, id)
	if err != nil {
		return err
	}
	if err = s.checkClaim(ctx, id, claim); err != nil {
		return err
	}
	if e.DeletionRequestedAt != nil {
		return evaldomain.Failure("eval_project_deleting")
	}
	if view.Member.ID != member {
		return evaldomain.Failure("eval_member_conflict")
	}
	doc, err := json.Marshal(view)
	if err != nil {
		return err
	}
	if err = evaldomain.Validate("MemberView", doc); err != nil {
		return err
	}
	tag, err := s.db.Exec(ctx, `
UPDATE eval_member_projections
SET projected_revision = revision,document = $4,collection_complete = $5,observed_document = $6,checked_at = clock_timestamp()
WHERE experiment_id = $1
    AND member_id = $2
    AND revision = $3
`, id, member, revision, doc, complete, []byte(observation))
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return evaldomain.Failure("eval_view_changed")
	}
	return nil
}

// RotateProjectionFailure prevents one damaged member from starving the rest.
func (s *Store) RotateProjectionFailure(ctx context.Context, owner, id, member string) error {
	_, err := s.db.Exec(ctx, `
UPDATE eval_member_projections p
SET checked_at = clock_timestamp()
FROM eval_experiments e
WHERE e.experiment_id = p.experiment_id
    AND e.owner_id = $1
    AND e.experiment_id = $2
    AND p.member_id = $3
`, owner, id, member)
	return err
}

type View struct {
	Snapshot     string
	Generation   int64
	Freshness    string
	Summary      evaldomain.Summary
	Suites       map[string]evaldomain.Summary
	PinsVerified bool
	CreatedAt    time.Time
}

func (s *Store) LatestView(ctx context.Context, owner, id string) (*View, error) {
	var v View
	var summary, suites []byte
	err := s.db.QueryRow(ctx, `
SELECT v.snapshot_id, v.generation,
    CASE WHEN q.revision = q.published_revision THEN 'current' ELSE 'stale' END,
    v.summary, v.suites, v.pins_verified, v.created_at
FROM eval_projection_queue q
JOIN eval_experiments e USING (experiment_id)
JOIN eval_view_generations v ON v.experiment_id = q.experiment_id AND v.snapshot_id = q.snapshot_id
WHERE e.owner_id = $1 AND e.experiment_id = $2
`, owner, id).Scan(&v.Snapshot, &v.Generation, &v.Freshness, &summary, &suites, &v.PinsVerified, &v.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if err = json.Unmarshal(summary, &v.Summary); err != nil {
		return nil, err
	}
	err = json.Unmarshal(suites, &v.Suites)
	return &v, err
}

// PublishView holds only private Eval locks. All rows and summaries become
// visible in one commit; failed/cancelled construction retains the old view.
func (s *Store) PublishView(ctx context.Context, scope Scope, id string, claim Claim, comparison evaldomain.Comparison, pinsVerified bool) (*View, error) {
	if _, err := s.project(ctx, scope, true); err != nil {
		return nil, err
	}
	e, err := s.locked(ctx, scope, id)
	if err != nil {
		return nil, err
	}
	if err = s.checkClaim(ctx, id, claim); err != nil {
		return nil, err
	}
	if e.DeletionRequestedAt != nil {
		return nil, evaldomain.Failure("eval_project_deleting")
	}
	var revision, published int64
	var snapshot *string
	err = s.db.QueryRow(ctx, `
SELECT revision,published_revision,snapshot_id
FROM eval_projection_queue
WHERE experiment_id = $1
FOR UPDATE
`, id).Scan(&revision, &published, &snapshot)
	if err != nil {
		return nil, normalize(err)
	}
	if revision == published {
		return s.LatestView(ctx, e.OwnerID, id)
	}
	rows, err := s.db.Query(ctx, `
SELECT m.ordinal,m.pair_id,p.member_id,p.revision,p.projected_revision,p.document,p.collection_complete
FROM eval_members m
JOIN eval_member_projections p USING(experiment_id,member_id)
WHERE m.experiment_id = $1
ORDER BY m.ordinal
LIMIT $2
`, id, evaldomain.MaxMembers+1)
	if err != nil {
		return nil, err
	}
	members := []evaldomain.SelectedMember{}
	copies := [][]any{}
	dirty := false
	for rows.Next() {
		var ordinal int
		var pair, member string
		var current, projected int64
		var document []byte
		var complete bool
		if err = rows.Scan(&ordinal, &pair, &member, &current, &projected, &document, &complete); err != nil {
			rows.Close()
			return nil, err
		}
		if current != projected || document == nil {
			dirty = true
			continue
		}
		var v evaldomain.MemberView
		if err = json.Unmarshal(document, &v); err != nil {
			rows.Close()
			return nil, err
		}
		members = append(members, evaldomain.SelectedMember{View: v, PairID: pair, CollectionComplete: complete})
		copies = append(copies, []any{id, int64(0), ordinal, member, pair, document, complete})
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return nil, err
	}
	if dirty {
		return nil, evaldomain.Failure("eval_not_ready")
	}
	if len(members) != e.Expected {
		return nil, evaldomain.Failure("eval_member_conflict")
	}
	comparisonView, err := evaldomain.BuildComparison(members, comparison, pinsVerified)
	if err != nil {
		return nil, err
	}
	identity := bytesOf(struct {
		ExperimentID      string
		SelectionRevision int64
		SourceRevision    int64
		Members           []evaldomain.SelectedMember
		Comparison        evaldomain.Comparison
		Pins              bool
	}{id, e.ViewGeneration, revision, members, comparison, pinsVerified})
	newSnapshot := "view-" + evaldomain.Digest(identity)[7:]
	if snapshot == nil || *snapshot != newSnapshot {
		var generation int64
		if err = s.db.QueryRow(ctx, `
SELECT COALESCE(max(generation), 0) + 1
FROM eval_view_generations
WHERE experiment_id = $1
`, id).Scan(&generation); err != nil {
			return nil, err
		}
		// The source revision makes evidence invalidation/recovery a new snapshot,
		// even when the recovered selected documents equal a previous generation.
		_, err = s.db.Exec(ctx, `
INSERT INTO eval_view_generations(
    experiment_id, generation, snapshot_id, summary, suites, pins_verified, source_revision
)
VALUES ($1, $2, $3, $4, $5, $6, $7)
`, id, generation, newSnapshot, bytesOf(comparisonView.Summary), bytesOf(comparisonView.Suites), pinsVerified, revision)
		if err != nil {
			return nil, err
		}
		for _, row := range copies {
			row[1] = generation
		}
		_, err = s.tx.CopyFrom(ctx, pgx.Identifier{"eval_view_members"}, []string{"experiment_id", "generation", "ordinal", "member_id", "pair_id", "document", "collection_complete"}, pgx.CopyFromRows(copies))
		if err != nil {
			return nil, err
		}
		if err = s.projectComparison(ctx, id, generation, comparisonView, pinsVerified); err != nil {
			return nil, err
		}
		if e.StartedAt != nil {
			_, err = s.db.Exec(ctx, `
INSERT INTO eval_progress_observations(experiment_id, counts, observed_at)
VALUES($1,$2,(SELECT created_at FROM eval_view_generations WHERE experiment_id = $1 AND snapshot_id = $3))
`, id, bytesOf(map[string]any{"a": comparisonView.Summary.Counts[comparison.Baseline].Terminal, "b": comparisonView.Summary.Counts[comparison.Candidate].Terminal, "suites": comparisonView.Suites}), newSnapshot)
			if err != nil {
				return nil, err
			}
		}
	}
	_, err = s.db.Exec(ctx, `
UPDATE eval_projection_queue
SET published_revision = $2,snapshot_id = $3
WHERE experiment_id = $1
`, id, revision, newSnapshot)
	if err != nil {
		return nil, err
	}
	return s.LatestView(ctx, e.OwnerID, id)
}

// ViewMembers reads a single already-published complete generation. Public
// pages filter in SQL; whole-view consumers are bounded by the format maximum.
func (s *Store) ViewMembers(ctx context.Context, owner, id, snapshot string) ([]evaldomain.SelectedMember, error) {
	rows, err := s.db.Query(ctx, `
SELECT m.pair_id,m.document,m.collection_complete
FROM eval_view_members m
JOIN eval_view_generations v USING(experiment_id,generation)
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id = $1
    AND e.experiment_id = $2
    AND v.snapshot_id = $3
ORDER BY m.ordinal
LIMIT $4
`, owner, id, snapshot, evaldomain.MaxMembers+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []evaldomain.SelectedMember{}
	for rows.Next() {
		var m evaldomain.SelectedMember
		var doc []byte
		if err = rows.Scan(&m.PairID, &doc, &m.CollectionComplete); err != nil {
			return nil, err
		}
		if err = json.Unmarshal(doc, &m.View); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	if len(out) > evaldomain.MaxMembers {
		return nil, evaldomain.Failure("eval_limit_exceeded")
	}
	return out, rows.Err()
}
