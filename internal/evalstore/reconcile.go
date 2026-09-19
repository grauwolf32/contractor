package evalstore

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
)

type CommandRecord struct {
	ID, ExperimentID, State string
	Kind                    evaldomain.CommandKind
	Revision                int64
	Diagnostic              json.RawMessage
	CreatedAt               time.Time
	FinishedAt              *time.Time
}

func (s *Store) PendingCommands(ctx context.Context, owner, id string) ([]CommandRecord, error) {
	rows, err := s.db.Query(ctx, `SELECT c.command_id,c.experiment_id,c.kind,c.state,c.accepted_revision,c.diagnostic,c.created_at,c.finished_at FROM eval_commands c JOIN eval_experiments e USING(experiment_id) WHERE e.owner_id=$1 AND e.experiment_id=$2 AND c.state IN ('accepted','running') ORDER BY c.created_at,c.command_id LIMIT 100`, owner, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []CommandRecord{}
	for rows.Next() {
		var c CommandRecord
		if err = rows.Scan(&c.ID, &c.ExperimentID, &c.Kind, &c.State, &c.Revision, &c.Diagnostic, &c.CreatedAt, &c.FinishedAt); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}
func (s *Store) Member(ctx context.Context, owner, id, member string) (Member, error) {
	var m Member
	var raw []byte
	err := s.db.QueryRow(ctx, `SELECT m.member_id,m.pair_id,m.ordinal,m.suite_id,m.case_id,m.sample,m.variant_id,m.eligibility,m.execution_kind,m.recipe,m.submission_key,m.case_sha256,m.binding_sha256 FROM eval_members m JOIN eval_experiments e USING(experiment_id) WHERE e.owner_id=$1 AND e.experiment_id=$2 AND m.member_id=$3`, owner, id, member).Scan(&m.MemberID, &m.PairID, &m.Ordinal, &m.SuiteID, &m.CaseID, &m.Sample, &m.VariantID, &m.Eligibility, &m.ExecutionKind, &raw, &m.SubmissionKey, &m.CaseSHA256, &m.BindingSHA256)
	if err != nil {
		return m, normalize(err)
	}
	err = json.Unmarshal(raw, &m.Recipe)
	return m, err
}
func (s *Store) NextMember(ctx context.Context, owner, id string) (*Member, error) {
	var member string
	err := s.db.QueryRow(ctx, `SELECT m.member_id FROM eval_members m JOIN eval_experiments e USING(experiment_id) WHERE e.owner_id=$1 AND e.experiment_id=$2 AND m.eligibility='eligible' AND NOT EXISTS(SELECT 1 FROM eval_submissions s WHERE s.experiment_id=m.experiment_id AND s.member_id=m.member_id) ORDER BY m.ordinal LIMIT 1`, owner, id).Scan(&member)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	m, err := s.Member(ctx, owner, id, member)
	return &m, err
}
func (s *Store) Outstanding(ctx context.Context, owner, id string, limit int) ([]string, error) {
	if limit < 1 || limit > 100 {
		return nil, evaldomain.Failure("eval_invalid")
	}
	rows, err := s.db.Query(ctx, `SELECT s.member_id FROM eval_submissions s JOIN eval_experiments e USING(experiment_id) WHERE e.owner_id=$1 AND e.experiment_id=$2 AND s.state IN ('intent','accepted') ORDER BY s.updated_at,s.member_id LIMIT $3`, owner, id, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []string{}
	for rows.Next() {
		var mid string
		if err = rows.Scan(&mid); err != nil {
			return nil, err
		}
		out = append(out, mid)
	}
	return out, rows.Err()
}

// LockMemberRecovery establishes all Project locks before the experiment lock
// when a composed transaction will also write its Audit workspace artifacts.
func (s *Store) LockMemberRecovery(ctx context.Context, scope Scope, id, member string, claim Claim) error {
	if _, err := s.project(ctx, scope, true); err != nil {
		return err
	}
	var child *string
	err := s.db.QueryRow(ctx, `SELECT execution_project_id FROM eval_submissions WHERE experiment_id=$1 AND member_id=$2`, id, member).Scan(&child)
	if err != nil {
		return normalize(err)
	}
	if child != nil {
		var exists bool
		err = s.db.QueryRow(ctx, `SELECT true FROM projects WHERE owner_id=$1 AND project_id=$2 FOR SHARE`, scope.OwnerID, *child).Scan(&exists)
		if err != nil {
			return normalize(err)
		}
	}
	_, err = s.recovery(ctx, scope, id, claim)
	return err
}

// ObserveTokens retains a per-member and experiment high-water mark. Unknown
// usage is not a public zero; this is only the known lower bound used by the
// optional observed-token stop policy. Updated timestamps rotate bounded polls.
func (s *Store) ObserveTokens(ctx context.Context, scope Scope, id, member string, claim Claim, tokens int64) error {
	if tokens < 0 {
		return evaldomain.Failure("eval_invalid")
	}
	if _, err := s.recovery(ctx, scope, id, claim); err != nil {
		return err
	}
	var prior int64
	if err := s.db.QueryRow(ctx, `SELECT observed_tokens FROM eval_submissions WHERE experiment_id=$1 AND member_id=$2`, id, member).Scan(&prior); err != nil {
		return normalize(err)
	}
	if tokens < prior {
		tokens = prior
	}
	if _, err := s.db.Exec(ctx, `UPDATE eval_submissions SET observed_tokens=$3,updated_at=clock_timestamp() WHERE experiment_id=$1 AND member_id=$2`, id, member, tokens); err != nil {
		return err
	}
	if tokens > prior {
		_, err := s.db.Exec(ctx, `UPDATE eval_experiments SET observed_tokens=observed_tokens+$2,`+advance+` WHERE experiment_id=$1`, id, tokens-prior)
		return err
	}
	return nil
}
func (s *Store) Dependencies(ctx context.Context, owner, id, after string, limit int) ([]string, error) {
	if limit < 1 || limit > 100 {
		return nil, evaldomain.Failure("eval_invalid")
	}
	rows, err := s.db.Query(ctx, `SELECT d.project_id FROM eval_project_dependencies d JOIN projects p USING(project_id) WHERE d.owner_id=$1 AND d.experiment_id=$2 AND d.project_id>$3 ORDER BY d.project_id LIMIT $4`, owner, id, after, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []string{}
	for rows.Next() {
		var p string
		if err = rows.Scan(&p); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func (s *Store) GetClaimed(ctx context.Context, claim Claim) (Experiment, error) {
	var owner string
	err := s.db.QueryRow(ctx, `SELECT e.owner_id FROM eval_experiments e JOIN eval_controller_claims c USING(experiment_id) WHERE c.experiment_id=$1 AND c.holder_id=$2 AND c.epoch=$3 AND c.expires_at>clock_timestamp()`, claim.ExperimentID, claim.HolderID, claim.Epoch).Scan(&owner)
	if errors.Is(err, pgx.ErrNoRows) {
		return Experiment{}, ErrClaimLost
	}
	if err != nil {
		return Experiment{}, err
	}
	return s.Get(ctx, owner, claim.ExperimentID)
}
