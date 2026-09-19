package evalstore

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type CommandParams struct {
	Scope                            Scope
	ExperimentID, CommandID          string
	DuplicateID, DuplicatePortableID string
	Command                          evaldomain.Command
	Mutation                         evaldomain.MutationIdentity
}

func (s *Store) Command(ctx context.Context, p CommandParams) (Receipt, error) {
	if !resourceID.MatchString(p.CommandID) {
		return Receipt{}, evaldomain.Failure("eval_invalid")
	}
	if err := evaldomain.Validate("Command", bytesOf(p.Command)); err != nil {
		return Receipt{}, err
	}
	return s.mutateJSON(ctx, p.Scope, p.ExperimentID, "command", p.Mutation, func() (json.RawMessage, error) {
		e, err := s.locked(ctx, p.Scope, p.ExperimentID)
		if err != nil {
			return nil, err
		}
		if err = checkMutable(e, p.Mutation); err != nil {
			return nil, err
		}
		target, err := e.Lifecycle().CommandTarget(p.Command.Kind)
		if err != nil {
			return nil, err
		}
		if p.Command.Kind != "prepare" && (p.Command.Kind != "duplicate" || p.Command.PlanSHA256 != "") {
			plan, err := s.FrozenPlan(ctx, e.OwnerID, e.ID)
			if err != nil {
				return nil, err
			}
			if plan.SHA256 != p.Command.PlanSHA256 {
				return nil, evaldomain.Failure("eval_pin_mismatch")
			}
		}
		if p.Command.Kind == "duplicate" {
			if !resourceID.MatchString(p.DuplicateID) || evaldomain.Validate("Id", bytesOf(p.DuplicatePortableID)) != nil || e.Draft.Kind() != "Draft" {
				return nil, evaldomain.Failure("eval_invalid")
			}
			// Copy authoring intent only: no member, receipt, clock or frozen plan is reused.
			_, err = s.db.Exec(ctx, `INSERT INTO eval_experiments(experiment_id,owner_id,project_id,portable_id,control_mode,name,state,draft,dataset_id,dataset_revision,max_in_flight,wall_ms,token_limit)
 SELECT $2,owner_id,project_id,$3,'server',name,'draft',draft,dataset_id,dataset_revision,max_in_flight,wall_ms,token_limit FROM eval_experiments WHERE experiment_id=$1`, e.ID, p.DuplicateID, p.DuplicatePortableID)
			if err != nil {
				return nil, normalize(err)
			}
			if _, err = s.db.Exec(ctx, `INSERT INTO eval_controller_claims(experiment_id) VALUES($1)`, p.DuplicateID); err != nil {
				return nil, err
			}
			return json.Marshal(ExperimentReceipt{ExperimentID: p.DuplicateID, Revision: 1, State: evaldomain.StateDraft})
		}

		_, err = s.db.Exec(ctx, `UPDATE eval_experiments SET state=$2,
   started_at=CASE WHEN $3 THEN COALESCE(started_at,statement_timestamp()) ELSE started_at END,
   deadline_at=CASE WHEN $3 THEN COALESCE(deadline_at,statement_timestamp()+wall_ms*interval '1 millisecond') ELSE deadline_at END,
   last_producer_activity_at=CASE WHEN control_mode='external' THEN clock_timestamp() ELSE last_producer_activity_at END,`+advance+` WHERE experiment_id=$1`, e.ID, target, p.Command.Kind == "start")
		if err != nil {
			return nil, err
		}
		_, err = s.db.Exec(ctx, `INSERT INTO eval_commands(command_id,experiment_id,actor_id,kind,state,accepted_revision) VALUES($1,$2,$3,$4,'accepted',$5)`, p.CommandID, e.ID, p.Scope.OwnerID, p.Command.Kind, e.Revision+1)
		if err != nil {
			return nil, normalize(err)
		}
		return json.Marshal(AcceptedCommandReceipt{CommandID: p.CommandID, ExperimentRevision: e.Revision + 1, State: "accepted"})
	})
}

// Transition commits observed progress, not an arbitrary caller lifecycle edit.
// Recovery needs the current epoch even when it only records terminal drain.
func (s *Store) Transition(ctx context.Context, scope Scope, id string, claim Claim, from, to evaldomain.State, observedTokens int64, diagnostic json.RawMessage) error {
	if err := s.requireTx(); err != nil {
		return err
	}
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
	if e.State != from {
		return evaldomain.Failure("eval_not_ready")
	}
	if observedTokens < e.ObservedTokens {
		return evaldomain.Failure("eval_invalid")
	}
	if err := e.Lifecycle().ValidateObservation(to); err != nil {
		if errors.Is(err, evaldomain.ErrOutstandingExecutions) {
			return ErrDrain
		}
		return err
	}

	if diagnostic != nil {
		if err := evaldomain.Validate("Diagnostic", diagnostic); err != nil {
			return err
		}
	}
	_, err = s.db.Exec(ctx, `UPDATE eval_experiments SET state=$2,observed_tokens=$3,diagnostic=$4,`+advance+` WHERE experiment_id=$1`, id, to, observedTokens, diagnostic)
	return err
}
func (s *Store) CompleteCommand(ctx context.Context, scope Scope, id, commandID string, claim Claim, succeeded bool, diagnostic json.RawMessage) error {
	if _, err := s.project(ctx, scope, true); err != nil {
		return err
	}
	if _, err := s.locked(ctx, scope, id); err != nil {
		return err
	}
	if err := s.checkClaim(ctx, id, claim); err != nil {
		return err
	}
	if diagnostic != nil {
		if err := evaldomain.Validate("Diagnostic", diagnostic); err != nil {
			return err
		}
	}
	state := "failed"
	if succeeded {
		state = "succeeded"
	}
	tag, err := s.db.Exec(ctx, `UPDATE eval_commands SET state=$3,diagnostic=$4,finished_at=clock_timestamp() WHERE experiment_id=$1 AND command_id=$2 AND state IN ('accepted','running')`, id, commandID, state, diagnostic)
	if err == nil && tag.RowsAffected() != 1 {
		return evaldomain.Failure("eval_not_found")
	}
	return err
}
