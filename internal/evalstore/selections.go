package evalstore

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
)

type Selection struct {
	evaldomain.SelectionEntry
	Revision int64  `json:"revision"`
	ActorID  string `json:"actorId"`
}

func (s *Store) Selected(ctx context.Context, owner, id, member string) (*Selection, error) {
	var out Selection
	err := s.db.QueryRow(ctx, `
SELECT s.member_id,s.result_sha256,s.assessment_sha256,s.revision,s.actor_id
FROM eval_selections s
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id = $1
    AND e.experiment_id = $2
    AND s.member_id = $3
`, owner, id, member).Scan(&out.MemberID, &out.ResultSHA256, &out.AssessmentSHA256, &out.Revision, &out.ActorID)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	return &out, err
}

// SelectRecords atomically selects exact revisions. View construction may occur
// later; the receipt names a durable selection revision, not a fabricated view.
func (s *Store) SelectRecords(ctx context.Context, scope Scope, id, actor string, input evaldomain.SelectionInput, mutation evaldomain.MutationIdentity) (Receipt, error) {
	return s.mutateJSON(ctx, scope, id, "selection", mutation, func() (json.RawMessage, error) {
		e, err := s.locked(ctx, scope, id)
		if err != nil {
			return nil, err
		}
		if err = checkMutable(e, mutation); err != nil {
			return nil, err
		}
		plan, err := s.FrozenPlan(ctx, e.OwnerID, id)
		if err != nil {
			return nil, err
		}
		if input.PlanSHA256 != plan.SHA256 {
			return nil, evaldomain.Failure("eval_pin_mismatch")
		}
		if err = evaldomain.Validate("SelectionInput", bytesOf(input)); err != nil {
			return nil, err
		}
		for _, entry := range input.Selections {
			if err = s.selectRecord(ctx, e, actor, entry); err != nil {
				return nil, err
			}
		}
		_, err = s.db.Exec(ctx, `
UPDATE eval_experiments
SET view_generation = view_generation + 1,
    last_producer_activity_at = CASE WHEN control_mode = 'external'
        THEN clock_timestamp() ELSE last_producer_activity_at END,
`+advance+` WHERE experiment_id=$1`, id)
		if err != nil {
			return nil, err
		}
		return json.Marshal(map[string]any{"revision": e.Revision + 1, "viewSnapshot": nil})
	})
}
func (s *Store) selectRecord(ctx context.Context, e Experiment, actor string, entry evaldomain.SelectionEntry) error {
	if actor == "" {
		return evaldomain.Failure("eval_invalid")
	}
	if _, err := s.Record(ctx, e.OwnerID, e.ID, entry.MemberID, evaldomain.RecordKindResult, entry.ResultSHA256); err != nil {
		return err
	}
	if entry.AssessmentSHA256 != nil {
		r, err := s.Record(ctx, e.OwnerID, e.ID, entry.MemberID, evaldomain.RecordKindAssessment, *entry.AssessmentSHA256)
		if err != nil {
			return err
		}
		var a evaldomain.AssessmentInput
		if err = json.Unmarshal(r.Document.Bytes(), &a); err != nil {
			return err
		}
		if a.ResultSHA256 != entry.ResultSHA256 {
			return evaldomain.Failure("eval_member_conflict")
		}
	}
	var revision int64
	err := s.db.QueryRow(ctx, `
INSERT INTO eval_selections(experiment_id, member_id, result_sha256, assessment_sha256, actor_id)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (experiment_id, member_id) DO UPDATE SET
    result_sha256 = EXCLUDED.result_sha256,
    assessment_sha256 = EXCLUDED.assessment_sha256,
    actor_id = EXCLUDED.actor_id,
    revision = eval_selections.revision + 1
RETURNING revision
`, e.ID, entry.MemberID, entry.ResultSHA256, entry.AssessmentSHA256, actor).Scan(&revision)
	if err != nil {
		return normalize(err)
	}
	_, err = s.db.Exec(ctx, `
INSERT INTO eval_selection_history(
    experiment_id, member_id, experiment_revision, selection_revision,
    result_sha256, assessment_sha256, actor_id
)
VALUES ($1, $2, $3, $4, $5, $6, $7)
`, e.ID, entry.MemberID, e.Revision+1, revision, entry.ResultSHA256, entry.AssessmentSHA256, actor)
	return err
}

// SelectFirstNative is an explicit audited system CAS. The experiment lock and
// missing-selection predicate make first collection safe under concurrent review.
func (s *Store) SelectFirstNative(ctx context.Context, scope Scope, id, member string, claim Claim, result, assessment evaldomain.Frozen) error {
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
	if e.ControlMode != "server" || e.DeletionRequestedAt != nil {
		return evaldomain.Failure("eval_external_control")
	}
	selected, err := s.Selected(ctx, e.OwnerID, id, member)
	if err != nil {
		return err
	}
	if selected != nil {
		return nil
	}
	r, err := s.insertRecord(ctx, e, member, evaldomain.NativeCollectorActor, result)
	if err != nil {
		return err
	}
	entry := evaldomain.SelectionEntry{MemberID: member, ResultSHA256: r.SHA256}
	if assessment.Kind() != "" {
		a, err := s.insertRecord(ctx, e, member, evaldomain.NativeCollectorActor, assessment)
		if err != nil {
			return err
		}
		entry.AssessmentSHA256 = &a.SHA256
	}
	if err = s.selectRecord(ctx, e, evaldomain.NativeCollectorActor, entry); err != nil {
		return err
	}
	_, err = s.db.Exec(ctx, `
UPDATE eval_experiments
SET view_generation = view_generation + 1,
`+advance+` WHERE experiment_id=$1`, id)
	return err
}
