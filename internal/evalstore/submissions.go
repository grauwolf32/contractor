package evalstore

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
)

type Admission struct {
	Scope                              Scope
	ExperimentID, MemberID, PlanSHA256 string
	Claim                              *Claim
	Mutation                           evaldomain.MutationIdentity
}
type Submission struct {
	ExperimentID, MemberID, State, ActorID         string
	ExecutionID, ExecutionProjectID, TerminalState *string
}

func (s *Store) Submission(ctx context.Context, owner, id, member string) (Submission, error) {
	var v Submission
	err := s.db.QueryRow(ctx, `SELECT s.experiment_id,s.member_id,s.state,s.actor_id,s.execution_id,s.execution_project_id,s.terminal_state FROM eval_submissions s JOIN eval_experiments e USING(experiment_id) WHERE e.owner_id=$1 AND e.experiment_id=$2 AND s.member_id=$3`, owner, id, member).Scan(&v.ExperimentID, &v.MemberID, &v.State, &v.ActorID, &v.ExecutionID, &v.ExecutionProjectID, &v.TerminalState)
	return v, normalize(err)
}
func (s *Store) Admit(ctx context.Context, p Admission) (Receipt, error) {
	return mutate(ctx, s, p.Scope, p.ExperimentID+":"+p.MemberID, "submission", p.Mutation, func() (AcceptedSubmissionReceipt, error) {
		e, err := s.locked(ctx, p.Scope, p.ExperimentID)
		if err != nil {
			return AcceptedSubmissionReceipt{}, err
		}
		if e.DeletionRequestedAt != nil {
			return AcceptedSubmissionReceipt{}, evaldomain.Failure("eval_project_deleting")
		}
		if p.Claim != nil {
			if e.ControlMode != "server" {
				return AcceptedSubmissionReceipt{}, evaldomain.Failure("eval_external_control")
			}
			if err = s.checkClaim(ctx, e.ID, *p.Claim); err != nil {
				return AcceptedSubmissionReceipt{}, err
			}
		} else if e.ControlMode != "external" {
			return AcceptedSubmissionReceipt{}, evaldomain.Failure("eval_external_control")
		}
		plan, err := s.FrozenPlan(ctx, e.OwnerID, e.ID)
		if err != nil {
			return AcceptedSubmissionReceipt{}, err
		}
		if plan.SHA256 != p.PlanSHA256 {
			return AcceptedSubmissionReceipt{}, evaldomain.Failure("eval_pin_mismatch")
		}
		var eligible, key string
		err = s.db.QueryRow(ctx, `SELECT eligibility,submission_key FROM eval_members WHERE experiment_id=$1 AND member_id=$2`, e.ID, p.MemberID).Scan(&eligible, &key)
		if err != nil {
			return AcceptedSubmissionReceipt{}, normalize(err)
		}
		var existing string
		err = s.db.QueryRow(ctx, `SELECT state FROM eval_submissions WHERE experiment_id=$1 AND member_id=$2`, e.ID, p.MemberID).Scan(&existing)
		if err == nil {
			return AcceptedSubmissionReceipt{SubmissionKey: key, ExperimentRevision: e.Revision, State: existing}, nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return AcceptedSubmissionReceipt{}, err
		}
		if eligible != "eligible" {
			return AcceptedSubmissionReceipt{}, evaldomain.Failure("eval_not_ready")
		}
		var expired bool
		err = s.db.QueryRow(ctx, `SELECT COALESCE(deadline_at<=clock_timestamp(),false) FROM eval_experiments WHERE experiment_id=$1`, e.ID).Scan(&expired)
		if err != nil {
			return AcceptedSubmissionReceipt{}, err
		}
		exhausted := expired || e.TokenLimit != nil && e.ObservedTokens >= *e.TokenLimit
		if err := e.Lifecycle().ValidateAdmission(exhausted, e.MaxInFlight); err != nil {
			return AcceptedSubmissionReceipt{}, err
		}
		_, err = s.db.Exec(ctx, `INSERT INTO eval_submissions(experiment_id,member_id,state,actor_id,execution_kind) SELECT $1,$2,'intent',$3,execution_kind FROM eval_members WHERE experiment_id=$1 AND member_id=$2`, e.ID, p.MemberID, p.Scope.OwnerID)
		if err != nil {
			return AcceptedSubmissionReceipt{}, normalize(err)
		}
		_, err = s.db.Exec(ctx, `UPDATE eval_experiments SET state='running',outstanding_count=outstanding_count+1,
    started_at=COALESCE(started_at,statement_timestamp()),deadline_at=COALESCE(deadline_at,statement_timestamp()+wall_ms*interval '1 millisecond'),
    last_producer_activity_at=CASE WHEN control_mode='external' THEN clock_timestamp() ELSE last_producer_activity_at END,`+advance+` WHERE experiment_id=$1`, e.ID)
		return AcceptedSubmissionReceipt{SubmissionKey: key, ExperimentRevision: e.Revision + 1, State: "intent"}, err
	})
}

// Recovery locks accept a deleting workspace: accepted work must be reconciled.
// A current claim is mandatory, including reconciliation of external submissions.
func (s *Store) recovery(ctx context.Context, scope Scope, id string, claim Claim) (Experiment, error) {
	if _, err := s.project(ctx, scope, true); err != nil {
		return Experiment{}, err
	}
	e, err := s.locked(ctx, scope, id)
	if err != nil {
		return e, err
	}
	return e, s.checkClaim(ctx, id, claim)
}

// RegisterExecutionProject must compose with ordinary Project creation in the
// same transaction. No arbitrary pre-existing workspace can be attached.
func (s *Store) RegisterExecutionProject(ctx context.Context, scope Scope, id, member, projectID, creationKey string, claim Claim) error {
	// Lock both workspace rows before the experiment, matching Project deletion.
	if err := s.requireTx(); err != nil {
		return err
	}
	if _, err := s.project(ctx, scope, true); err != nil {
		return err
	}
	var ok bool
	err := s.db.QueryRow(ctx, `SELECT true FROM projects WHERE owner_id=$1 AND project_id=$2 AND kind='project' AND lifecycle_state='active' AND request_idempotency_key=$3 AND request_digest=(SELECT request_sha256 FROM eval_suboperations WHERE experiment_id=$4 AND member_id=$5 AND kind='project-create') FOR SHARE`, scope.OwnerID, projectID, creationKey, id, member).Scan(&ok)
	if err != nil {
		return normalize(err)
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
	var key string
	err = s.db.QueryRow(ctx, `SELECT op.operation_key FROM eval_suboperations op JOIN eval_members m USING(experiment_id,member_id) WHERE op.experiment_id=$1 AND op.member_id=$2 AND op.kind='project-create' AND m.execution_kind='audit'`, id, member).Scan(&key)
	if err != nil {
		return normalize(err)
	}
	if key != creationKey {
		return evaldomain.Failure("eval_member_conflict")
	}
	_, err = s.db.Exec(ctx, `INSERT INTO eval_project_dependencies(project_id,experiment_id,member_id,owner_id) VALUES($1,$2,$3,$4) ON CONFLICT (project_id) DO NOTHING`, projectID, id, member, scope.OwnerID)
	if err != nil {
		return normalize(err)
	}
	tag, err := s.db.Exec(ctx, `UPDATE eval_submissions SET execution_project_id=$3 WHERE experiment_id=$1 AND member_id=$2 AND state='intent' AND (execution_project_id IS NULL OR execution_project_id=$3)`, id, member, projectID)
	if err == nil && tag.RowsAffected() != 1 {
		return evaldomain.Failure("eval_member_conflict")
	}
	return err
}

type Suboperation struct {
	Kind, Key, State  string
	Request, Response json.RawMessage
}

func (s *Store) Suboperation(ctx context.Context, owner, id, member, kind string) (Suboperation, error) {
	var out Suboperation
	err := s.db.QueryRow(ctx, `SELECT op.kind,op.operation_key,op.state,op.request,op.response FROM eval_suboperations op JOIN eval_experiments e USING(experiment_id) WHERE e.owner_id=$1 AND e.experiment_id=$2 AND op.member_id=$3 AND op.kind=$4`, owner, id, member, kind).Scan(&out.Kind, &out.Key, &out.State, &out.Request, &out.Response)
	return out, normalize(err)
}
func (s *Store) PutSuboperation(ctx context.Context, scope Scope, id, member string, claim Claim, op Suboperation) error {
	if _, err := evaldomain.StrictJSON(op.Request); err != nil {
		return err
	}
	if !resourceID.MatchString(op.Key) {
		return evaldomain.Failure("eval_invalid")
	}
	e, err := s.recovery(ctx, scope, id, claim)
	if err != nil {
		return err
	}
	// Replay existing effects through a fence, but create no new side effect after
	// cancellation except the explicit cancellation operation itself.
	old, err := s.Suboperation(ctx, scope.OwnerID, id, member, op.Kind)
	if err == nil {
		if old.Key != op.Key || evaldomain.Digest(old.Request) != evaldomain.Digest(op.Request) {
			return evaldomain.Failure("eval_idempotency_conflict")
		}
		return nil
	}
	var domain *evaldomain.Error
	if !errors.As(err, &domain) || domain.Code != "eval_not_found" {
		return err
	}
	if (e.DeletionRequestedAt != nil || e.State == "cancelling") && op.Kind != "cancel" {
		return evaldomain.Failure("eval_project_deleting")
	}
	var state string
	if err = s.db.QueryRow(ctx, `SELECT state FROM eval_submissions WHERE experiment_id=$1 AND member_id=$2`, id, member).Scan(&state); err != nil {
		return normalize(err)
	}
	if state != "intent" && !(state == "accepted" && op.Kind == "cancel") {
		return evaldomain.Failure("eval_not_ready")
	}
	_, err = s.db.Exec(ctx, `INSERT INTO eval_suboperations(experiment_id,member_id,kind,operation_key,request) VALUES($1,$2,$3,$4,$5)`, id, member, op.Kind, op.Key, []byte(op.Request))
	return normalize(err)
}
func (s *Store) ResolveSuboperation(ctx context.Context, scope Scope, id, member, kind string, claim Claim, response []byte, rejected bool) error {
	if _, err := evaldomain.StrictJSON(response); err != nil {
		return err
	}
	if _, err := s.recovery(ctx, scope, id, claim); err != nil {
		return err
	}
	op, err := s.Suboperation(ctx, scope.OwnerID, id, member, kind)
	if err != nil {
		return err
	}
	state := "succeeded"
	if rejected {
		state = "rejected"
	}
	if op.State != "intent" {
		if op.State == state && evaldomain.Digest(op.Response) == evaldomain.Digest(response) {
			return nil
		}
		return evaldomain.Failure("eval_member_conflict")
	}
	_, err = s.db.Exec(ctx, `UPDATE eval_suboperations SET state=$4,response=$5 WHERE experiment_id=$1 AND member_id=$2 AND kind=$3`, id, member, kind, state, response)
	return err
}

// BindExecution checks ordinary service authority, not a caller's labels. It
// can recover an already accepted creation after an experiment is fenced.
func (s *Store) BindExecution(ctx context.Context, scope Scope, id, member, executionID string, claim Claim) error {
	if _, err := s.recovery(ctx, scope, id, claim); err != nil {
		return err
	}
	var kind, key, digest string
	err := s.db.QueryRow(ctx, `SELECT m.execution_kind,op.operation_key,op.request_sha256 FROM eval_members m JOIN eval_suboperations op USING(experiment_id,member_id) WHERE m.experiment_id=$1 AND m.member_id=$2 AND op.kind=CASE WHEN m.execution_kind='run' THEN 'run-create' ELSE 'audit-create' END`, id, member).Scan(&kind, &key, &digest)
	if err != nil {
		return normalize(err)
	}
	var verified bool
	if kind == "run" {
		err = s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM workflow_runs WHERE run_id=$1 AND owner_id=$2 AND project_id=$3 AND request_idempotency_key=$4 AND request_digest=$5 AND publication_mode='ordinary')`, executionID, scope.OwnerID, scope.ProjectID, key, digest).Scan(&verified)
	} else {
		err = s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audits a JOIN audit_idempotency i USING(audit_id) JOIN eval_project_dependencies d ON d.project_id=a.project_id WHERE a.audit_id=$1 AND a.owner_id=$2 AND d.experiment_id=$3 AND d.member_id=$4 AND i.owner_id=$2 AND i.operation='audit.create' AND i.idempotency_key=$5 AND i.request_digest=$6)`, executionID, scope.OwnerID, id, member, key, digest).Scan(&verified)
	}
	if err != nil {
		return err
	}
	if !verified {
		return evaldomain.Failure("eval_member_conflict")
	}
	tag, err := s.db.Exec(ctx, `UPDATE eval_submissions SET state='accepted',execution_id=$3,updated_at=clock_timestamp() WHERE experiment_id=$1 AND member_id=$2 AND (state='intent' OR (state='accepted' AND execution_id=$3))`, id, member, executionID)
	if err == nil && tag.RowsAffected() != 1 {
		return evaldomain.Failure("eval_member_conflict")
	}
	return normalize(err)
}

// Settle decreases outstanding once, only after an authoritative terminal
// execution and all durable suboperations have resolved. A failed transport
// cannot be turned into a rejected intent by this method.
func (s *Store) Settle(ctx context.Context, scope Scope, id, member string, claim Claim) error {
	e, err := s.recovery(ctx, scope, id, claim)
	if err != nil {
		return err
	}
	sub, err := s.Submission(ctx, scope.OwnerID, id, member)
	if err != nil {
		return err
	}
	if sub.State == "terminal" || sub.State == "rejected" {
		return nil
	}
	var unresolved bool
	err = s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM eval_suboperations WHERE experiment_id=$1 AND member_id=$2 AND state='intent')`, id, member).Scan(&unresolved)
	if err != nil {
		return err
	}
	if unresolved {
		return ErrDrain
	}
	state, terminal := "rejected", ""
	if sub.State == "accepted" {
		var kind string
		err = s.db.QueryRow(ctx, `SELECT execution_kind FROM eval_members WHERE experiment_id=$1 AND member_id=$2`, id, member).Scan(&kind)
		if err != nil {
			return err
		}
		if kind == "run" {
			err = s.db.QueryRow(ctx, `SELECT state FROM workflow_runs WHERE owner_id=$1 AND run_id=$2 AND state IN ('succeeded','failed','cancelled')`, scope.OwnerID, *sub.ExecutionID).Scan(&terminal)
		} else {
			err = s.db.QueryRow(ctx, `SELECT state FROM audits WHERE owner_id=$1 AND audit_id=$2 AND state IN ('completed','failed','cancelled') AND outstanding_run_count=0`, scope.OwnerID, *sub.ExecutionID).Scan(&terminal)
		}
		if errors.Is(err, pgx.ErrNoRows) {
			tombstone, tombErr := s.Tombstone(ctx, scope.OwnerID, id, member)
			if tombErr != nil {
				return ErrDrain
			}
			if tombstone.ID != *sub.ExecutionID {
				return evaldomain.Failure("eval_member_conflict")
			}
			terminal = tombstone.TerminalState
			err = nil
		}
		if err != nil {
			return err
		}
		state = "terminal"
	} else {
		var rejected bool
		err = s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM eval_execution_tombstones WHERE experiment_id=$1 AND member_id=$2 AND never_started) OR (NOT EXISTS(SELECT 1 FROM eval_suboperations WHERE experiment_id=$1 AND member_id=$2 AND kind IN ('run-create','audit-create') AND state IN ('intent','succeeded')) AND ($3 OR EXISTS(SELECT 1 FROM eval_suboperations WHERE experiment_id=$1 AND member_id=$2 AND state='rejected')))`, id, member, e.State == "cancelling").Scan(&rejected)
		if err != nil {
			return err
		}
		if !rejected {
			return ErrDrain
		}
	}
	var terminalPtr *string
	if terminal != "" {
		terminalPtr = &terminal
	}
	_, err = s.db.Exec(ctx, `UPDATE eval_submissions SET state=$3,terminal_state=$4,updated_at=clock_timestamp() WHERE experiment_id=$1 AND member_id=$2`, id, member, state, terminalPtr)
	if err != nil {
		return err
	}
	_, err = s.db.Exec(ctx, `UPDATE eval_experiments SET outstanding_count=outstanding_count-1,`+advance+` WHERE experiment_id=$1`, id)
	return err
}
