package evalstore

import (
	"context"
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type CommandParams struct {
	Scope                   Scope
	ExperimentID, CommandID string
	Command                 evaldomain.Command
	Mutation                evaldomain.MutationIdentity
}

func (s *Store) Command(ctx context.Context, p CommandParams) (Receipt, error) {
	if !resourceID.MatchString(p.CommandID) {
		return Receipt{}, evaldomain.Failure("eval_invalid")
	}
	if err := evaldomain.Validate("Command", bytesOf(p.Command)); err != nil {
		return Receipt{}, err
	}
	return s.mutate(ctx, p.Scope, p.ExperimentID, "command", p.Mutation, func() (Reference, error) {
		e, err := s.locked(ctx, p.Scope, p.ExperimentID)
		if err != nil {
			return Reference{}, err
		}
		if err = checkMutable(e, p.Mutation); err != nil {
			return Reference{}, err
		}
		if err = evaldomain.CheckControlMode(e.ControlMode, p.Command.Kind); err != nil {
			return Reference{}, err
		}
		if p.Command.Kind != "prepare" {
			plan, err := s.FrozenPlan(ctx, e.OwnerID, e.ID)
			if err != nil {
				return Reference{}, err
			}
			if plan.SHA256 != p.Command.PlanSHA256 {
				return Reference{}, evaldomain.Failure("eval_pin_mismatch")
			}
		}
		target := ""
		switch p.Command.Kind {
		case "prepare":
			if e.State == "draft" {
				target = "preparing"
			}
		case "start":
			if e.State == "ready" {
				target = "running"
			}
		case "pause":
			if e.State == "running" {
				target = "pausing"
			}
		case "resume":
			if e.State == "paused" || e.State == "interrupted" {
				target = "running"
			}
		case "cancel":
			if e.State != "finished" && e.State != "cancelled" {
				target = "cancelling"
			}
		case "finalize":
			if e.State == "ready" || e.State == "running" {
				target = "settling"
			}
		}
		if target == "" {
			return Reference{}, evaldomain.Failure("eval_not_ready")
		}
		_, err = s.db.Exec(ctx, `UPDATE eval_experiments SET state=$2,
   started_at=CASE WHEN $3 THEN COALESCE(started_at,statement_timestamp()) ELSE started_at END,
   deadline_at=CASE WHEN $3 THEN COALESCE(deadline_at,statement_timestamp()+wall_ms*interval '1 millisecond') ELSE deadline_at END,
   last_producer_activity_at=CASE WHEN control_mode='external' THEN clock_timestamp() ELSE last_producer_activity_at END,`+advance+` WHERE experiment_id=$1`, e.ID, target, p.Command.Kind == "start")
		if err != nil {
			return Reference{}, err
		}
		_, err = s.db.Exec(ctx, `INSERT INTO eval_commands(command_id,experiment_id,actor_id,kind,state,accepted_revision) VALUES($1,$2,$3,$4,'accepted',$5)`, p.CommandID, e.ID, p.Scope.OwnerID, p.Command.Kind, e.Revision+1)
		return Reference{ID: p.CommandID, Revision: e.Revision + 1, State: "accepted"}, normalize(err)
	})
}

// Transition commits observed progress, not an arbitrary caller lifecycle edit.
// Recovery needs the current epoch even when it only records terminal drain.
func (s *Store) Transition(ctx context.Context, scope Scope, id string, claim Claim, from, to string, observedTokens int64, diagnostic json.RawMessage) error {
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
	allowed := map[string]map[string]bool{
		"preparing": {"draft": true, "interrupted": true}, "running": {"running": true, "settling": true, "cancelling": true, "interrupted": true},
		"pausing": {"paused": true, "cancelling": true, "interrupted": true}, "paused": {"cancelling": true, "settling": true},
		"settling": {"finished": true, "cancelling": true, "interrupted": true}, "cancelling": {"cancelled": true, "interrupted": true}, "interrupted": {"cancelling": true},
	}
	if !allowed[from][to] || observedTokens < e.ObservedTokens {
		return evaldomain.Failure("eval_invalid")
	}
	if e.DeletionRequestedAt != nil && to != "cancelled" && to != "cancelling" && to != "interrupted" {
		return evaldomain.Failure("eval_project_deleting")
	}
	if (to == "paused" || to == "finished" || to == "cancelled") && e.Outstanding != 0 {
		return ErrDrain
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
