package evalstore

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
)

type Admission struct {
	Scope                              Scope
	ExperimentID, MemberID, PlanSHA256 string
	Claim                              *Claim
	Mutation                           evaldomain.MutationIdentity
}

// SubmissionState is the stored state of a member's submission: an intent
// whose execution is not yet confirmed, an accepted execution, or a settled
// submission whose execution ended (terminal) or never started (rejected).
// The eval_submissions state CHECK lists exactly these values.
type SubmissionState string

const (
	SubmissionIntent   SubmissionState = "intent"
	SubmissionAccepted SubmissionState = "accepted"
	SubmissionTerminal SubmissionState = "terminal"
	SubmissionRejected SubmissionState = "rejected"
)

type Submission struct {
	ExperimentID, MemberID, ActorID                string
	State                                          SubmissionState
	ExecutionID, ExecutionProjectID, TerminalState *string
}

func (s *Store) Submission(ctx context.Context, owner, id, member string) (Submission, error) {
	var v Submission
	err := s.db.QueryRow(ctx, `
SELECT s.experiment_id, s.member_id, s.state, s.actor_id, s.execution_id, s.execution_project_id, s.terminal_state
FROM eval_submissions s
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id=$1
    AND e.experiment_id=$2
    AND s.member_id=$3
`, owner, id, member).Scan(&v.ExperimentID, &v.MemberID, &v.State, &v.ActorID, &v.ExecutionID, &v.ExecutionProjectID, &v.TerminalState)
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
			if e.ControlMode != evaldomain.ControlServer {
				return AcceptedSubmissionReceipt{}, evaldomain.Failure("eval_external_control")
			}
			if err = s.checkClaim(ctx, e.ID, *p.Claim); err != nil {
				return AcceptedSubmissionReceipt{}, err
			}
		} else if e.ControlMode != evaldomain.ControlExternal {
			return AcceptedSubmissionReceipt{}, evaldomain.Failure("eval_external_control")
		}
		plan, err := s.FrozenPlanMetadata(ctx, e.OwnerID, e.ID)
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
		_, err = s.db.Exec(ctx, `
INSERT INTO eval_submissions(experiment_id, member_id, state, actor_id, execution_kind) SELECT $1, $2, 'intent', $3, execution_kind
FROM eval_members
WHERE experiment_id=$1
    AND member_id=$2
`, e.ID, p.MemberID, p.Scope.OwnerID)
		if err != nil {
			return AcceptedSubmissionReceipt{}, normalize(err)
		}
		_, err = s.db.Exec(ctx, `
UPDATE eval_experiments SET state='running',outstanding_count=outstanding_count+1,
started_at=COALESCE(started_at,statement_timestamp()),deadline_at=COALESCE(deadline_at,statement_timestamp()+wall_ms*interval '1 millisecond'),
last_producer_activity_at=CASE WHEN control_mode='external' THEN clock_timestamp() ELSE last_producer_activity_at END,
`+observe+` WHERE experiment_id=$1`, e.ID)
		return AcceptedSubmissionReceipt{SubmissionKey: key, ExperimentRevision: e.Revision, State: string(SubmissionIntent)}, err
	})
}

// RegisterExecutionProject must compose with ordinary Project creation in the
// same transaction. No arbitrary pre-existing workspace can be attached.
func (r *MemberRecovery) RegisterExecutionProject(ctx context.Context, projectID, creationKey string) error {
	s, scope, id, member := r.store, r.scope, r.id, r.member
	// Lock both workspace rows before the experiment, matching Project deletion.
	if err := s.requireTx(); err != nil {
		return err
	}
	if !r.held {
		if _, err := s.project(ctx, scope, true); err != nil {
			return err
		}
	}
	var ok bool
	err := s.db.QueryRow(ctx, registerExecutionProjectSQL, scope.OwnerID, projectID, creationKey, id, member).Scan(&ok)
	if err != nil {
		return normalize(err)
	}
	fence, err := r.afterProjectLocks(ctx)
	if err != nil {
		return err
	}
	if fence.deletionRequested {
		return evaldomain.Failure("eval_project_deleting")
	}
	var key string
	err = s.db.QueryRow(ctx, `
SELECT op.operation_key
FROM eval_suboperations op
JOIN eval_members m USING(experiment_id, member_id)
WHERE op.experiment_id=$1
    AND op.member_id=$2
    AND op.kind='project-create'
    AND m.execution_kind='audit'
`, id, member).Scan(&key)
	if err != nil {
		return normalize(err)
	}
	if key != creationKey {
		return evaldomain.Failure("eval_member_conflict")
	}
	tag, err := s.db.Exec(ctx, `
UPDATE eval_submissions
SET execution_project_id=$3
WHERE experiment_id=$1
    AND member_id=$2
    AND state='intent'
    AND (execution_project_id IS NULL OR execution_project_id=$3)
`, id, member, projectID)
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
	err := s.db.QueryRow(ctx, `
SELECT op.kind, op.operation_key, op.state, op.request, op.response
FROM eval_suboperations op
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id=$1
    AND e.experiment_id=$2
    AND op.member_id=$3
    AND op.kind=$4
`, owner, id, member, kind).Scan(&out.Kind, &out.Key, &out.State, &out.Request, &out.Response)
	return out, normalize(err)
}

// PutSuboperation journals the intent of an outbound call before the call.
func (r *MemberRecovery) PutSuboperation(ctx context.Context, op Suboperation) error {
	s, scope, id, member := r.store, r.scope, r.id, r.member
	if _, err := evaldomain.StrictJSON(op.Request); err != nil {
		return err
	}
	if !contracts.ValidResourceID(op.Key) {
		return evaldomain.Failure("eval_invalid")
	}
	fence, err := r.authority(ctx)
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
	if (fence.deletionRequested || fence.state == evaldomain.StateCancelling) && op.Kind != "cancel" {
		return evaldomain.Failure("eval_project_deleting")
	}
	var state SubmissionState
	if err = s.db.QueryRow(ctx, `SELECT state FROM eval_submissions WHERE experiment_id=$1 AND member_id=$2`, id, member).Scan(&state); err != nil {
		return normalize(err)
	}
	if state != SubmissionIntent && !(state == SubmissionAccepted && op.Kind == "cancel") {
		return evaldomain.Failure("eval_not_ready")
	}
	_, err = s.db.Exec(ctx, `INSERT INTO eval_suboperations(experiment_id,member_id,kind,operation_key,request) VALUES($1,$2,$3,$4,$5)`, id, member, op.Kind, op.Key, []byte(op.Request))
	return normalize(err)
}

// ResolveSuboperation records the outcome of the member's suboperation kind
// once; a repeated identical outcome replays.
func (r *MemberRecovery) ResolveSuboperation(ctx context.Context, kind string, response []byte, rejected bool) error {
	s, scope, id, member := r.store, r.scope, r.id, r.member
	if _, err := evaldomain.StrictJSON(response); err != nil {
		return err
	}
	if _, err := r.authority(ctx); err != nil {
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
func (r *MemberRecovery) BindExecution(ctx context.Context, executionID string) error {
	s, scope, id, member := r.store, r.scope, r.id, r.member
	if _, err := r.authority(ctx); err != nil {
		return err
	}
	var kind, key, digest string
	err := s.db.QueryRow(ctx, `
SELECT m.execution_kind, op.operation_key, op.request_sha256
FROM eval_members m
JOIN eval_suboperations op USING(experiment_id, member_id)
WHERE m.experiment_id=$1
    AND m.member_id=$2
    AND op.kind=CASE WHEN m.execution_kind='run' THEN 'run-create' ELSE 'audit-create' END
`, id, member).Scan(&kind, &key, &digest)
	if err != nil {
		return normalize(err)
	}
	var verified bool
	if kind == "run" {
		err = s.db.QueryRow(ctx, `
SELECT EXISTS(SELECT 1
    FROM workflow_runs
    WHERE run_id=$1
        AND owner_id=$2
        AND project_id=$3
        AND request_idempotency_key=$4
        AND request_digest=$5
        AND publication_mode='ordinary')
`, executionID, scope.OwnerID, scope.ProjectID, key, digest).Scan(&verified)
	} else {
		verified, err = s.auditCreatedInWorkspace(ctx, scope.OwnerID, id, member, executionID, key, digest)
	}
	if err != nil {
		return err
	}
	if !verified {
		return evaldomain.Failure("eval_member_conflict")
	}
	tag, err := s.db.Exec(ctx, `
UPDATE eval_submissions
SET state='accepted', execution_id=$3, updated_at=clock_timestamp()
WHERE experiment_id=$1
    AND member_id=$2
    AND (state='intent' OR (state='accepted'
            AND execution_id=$3))
`, id, member, executionID)
	if err == nil && tag.RowsAffected() != 1 {
		return evaldomain.Failure("eval_member_conflict")
	}
	return normalize(err)
}

// auditCreatedInWorkspace reports whether owner's Audit auditID was created
// under the member's audit-create suboperation key and request digest and
// lives in the workspace Project registered on the member's submission.
// The Audit store resolves the Audit that the creation key made.
func (s *Store) auditCreatedInWorkspace(
	ctx context.Context, owner, id, member, auditID, key, digest string,
) (bool, error) {
	audit, found, err := hookstore.NewAuditCreations(s.db, s.storage).LookupCreation(ctx, owner, key, digest)
	switch {
	case errors.Is(err, auditstore.ErrInvalid), errors.Is(err, contracts.ErrIdempotencyConflict),
		errors.Is(err, auditstore.ErrNotFound):
		return false, nil
	case err != nil:
		return false, err
	case !found || audit.AuditID != auditID:
		return false, nil
	}
	var inWorkspace bool
	err = s.db.QueryRow(ctx, `
SELECT EXISTS(SELECT 1
    FROM eval_submissions
    WHERE experiment_id=$1
        AND member_id=$2
        AND execution_project_id=$3)
`, id, member, audit.ProjectID).Scan(&inWorkspace)
	return inWorkspace, err
}

// Settle decreases outstanding once, only after an authoritative terminal
// execution and all durable suboperations have resolved. A failed transport
// cannot be turned into a rejected intent by this method.
func (r *MemberRecovery) Settle(ctx context.Context) error {
	s, scope, id, member := r.store, r.scope, r.id, r.member
	fence, err := r.authority(ctx)
	if err != nil {
		return err
	}
	sub, err := s.Submission(ctx, scope.OwnerID, id, member)
	if err != nil {
		return err
	}
	if sub.State == SubmissionTerminal || sub.State == SubmissionRejected {
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
	state, terminal := SubmissionRejected, ""
	if sub.State == SubmissionAccepted {
		var kind string
		err = s.db.QueryRow(ctx, `SELECT execution_kind FROM eval_members WHERE experiment_id=$1 AND member_id=$2`, id, member).Scan(&kind)
		if err != nil {
			return err
		}
		if kind == "run" {
			err = s.db.QueryRow(ctx, `SELECT state FROM workflow_runs WHERE owner_id=$1 AND run_id=$2 AND state IN ('succeeded','failed','cancelled')`, scope.OwnerID, *sub.ExecutionID).Scan(&terminal)
		} else {
			err = s.db.QueryRow(ctx, `
SELECT state
FROM audits
WHERE owner_id=$1
    AND audit_id=$2
    AND state IN ('completed', 'failed', 'cancelled')
    AND outstanding_run_count=0
`, scope.OwnerID, *sub.ExecutionID).Scan(&terminal)
		}
		if errors.Is(err, pgx.ErrNoRows) {
			tombstone, tombErr := s.Tombstone(ctx, scope.OwnerID, id, member)
			if evaldomain.IsCode(tombErr, "eval_not_found") {
				// The execution is not terminal yet and left no tombstone.
				return ErrDrain
			}
			if tombErr != nil {
				return tombErr
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
		state = SubmissionTerminal
	} else {
		var rejected bool
		err = s.db.QueryRow(ctx, settleSubmissionSQL, id, member, fence.state == evaldomain.StateCancelling).Scan(&rejected)
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
	_, err = s.db.Exec(ctx, `UPDATE eval_experiments SET outstanding_count=outstanding_count-1,`+observe+` WHERE experiment_id=$1`, id)
	return err
}
