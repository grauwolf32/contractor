package evalstore

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
)

type Record struct {
	MemberID    string            `json:"memberId"`
	Kind        string            `json:"kind"`
	SHA256      string            `json:"recordSha256"`
	ActorID     string            `json:"actorId"`
	Predecessor *string           `json:"previousRecordSha256"`
	CreatedAt   time.Time         `json:"createdAt"`
	Document    evaldomain.Frozen `json:"-"`
}
type RecordParams struct {
	Scope                                      Scope
	ExperimentID, MemberID, ActorID, Operation string
	Mutation                                   evaldomain.MutationIdentity
	// Called under the Project/experiment locks, after exact replay. Validation
	// and ordinary observations share the caller's transaction snapshot.
	Build func(Experiment) (evaldomain.Frozen, error)
}

func (s *Store) PutRecord(ctx context.Context, p RecordParams) (Receipt, error) {
	return s.mutateJSON(ctx, p.Scope, p.ExperimentID+":"+p.MemberID, p.Operation, p.Mutation, func() (json.RawMessage, error) {
		e, err := s.locked(ctx, p.Scope, p.ExperimentID)
		if err != nil {
			return nil, err
		}
		if e.DeletionRequestedAt != nil {
			return nil, evaldomain.Failure("eval_project_deleting")
		}
		if _, err = evaldomain.CheckMutation(p.Mutation, nil, uint64(e.Revision)); err != nil {
			return nil, err
		}
		if _, err = s.Member(ctx, p.Scope.OwnerID, e.ID, p.MemberID); err != nil {
			return nil, err
		}
		document, err := p.Build(e)
		if err != nil {
			return nil, err
		}
		record, err := s.insertRecord(ctx, e, p.MemberID, p.ActorID, document)
		if err != nil {
			return nil, err
		}
		if err = s.recordProducerActivity(ctx, e, document); err != nil {
			return nil, err
		}
		return json.Marshal(map[string]any{"memberId": p.MemberID, "recordSha256": record.SHA256, "previousRecordSha256": record.Predecessor})
	})
}

// Human review and native checks do not indicate external producer activity.
// Receipt replay returns before this update and preserves the original clock.
func (s *Store) recordProducerActivity(ctx context.Context, e Experiment, document evaldomain.Frozen) error {
	if e.ControlMode != "external" {
		return nil
	}
	if document.Kind() == "AssessmentInput" {
		var assessment evaldomain.AssessmentInput
		if err := json.Unmarshal(document.Bytes(), &assessment); err != nil {
			return err
		}
		if assessment.Source.Kind != "external" {
			return nil
		}
	}
	_, err := s.db.Exec(ctx, `
UPDATE eval_experiments
SET last_producer_activity_at = clock_timestamp(), `+advance+`
WHERE experiment_id = $1`, e.ID)
	return err
}

func (s *Store) insertRecord(ctx context.Context, e Experiment, member, actor string, document evaldomain.Frozen) (Record, error) {
	r := Record{MemberID: member, SHA256: document.Digest(), ActorID: actor, Document: document}
	if actor == "" {
		return r, evaldomain.Failure("eval_invalid")
	}
	if err := evaldomain.Validate(document.Kind(), document.Bytes()); err != nil {
		return r, err
	}
	refs := []evaldomain.Artifact{}
	switch document.Kind() {
	case "ResultInput":
		r.Kind = evaldomain.RecordKindResult
		var input evaldomain.ResultInput
		if err := json.Unmarshal(document.Bytes(), &input); err != nil {
			return r, err
		}
		r.Predecessor = input.PreviousResultSHA256
		plan, err := s.FrozenPlan(ctx, e.OwnerID, e.ID)
		if err != nil {
			return r, err
		}
		if input.MemberID != member || input.PlanSHA256 != plan.SHA256 {
			return r, evaldomain.Failure("eval_member_conflict")
		}
		for _, ref := range input.Outputs {
			refs = append(refs, ref)
		}
		for _, ref := range input.Evidence {
			refs = append(refs, ref.Artifact)
		}
	case "AssessmentInput":
		r.Kind = evaldomain.RecordKindAssessment
		var input evaldomain.AssessmentInput
		if err := json.Unmarshal(document.Bytes(), &input); err != nil {
			return r, err
		}
		r.Predecessor = input.PreviousAssessmentSHA256
		if _, err := s.Record(ctx, e.OwnerID, e.ID, member, evaldomain.RecordKindResult, input.ResultSHA256); err != nil {
			return r, err
		}
	default:
		return r, evaldomain.Failure("eval_invalid")
	}
	if r.Predecessor != nil {
		if *r.Predecessor == r.SHA256 {
			return r, evaldomain.Failure("eval_member_conflict")
		}
		if _, err := s.Record(ctx, e.OwnerID, e.ID, member, r.Kind, *r.Predecessor); err != nil {
			return r, err
		}
	}
	_, err := s.db.Exec(ctx, `
INSERT INTO eval_records(
    experiment_id, member_id, kind, record_sha256, document, actor_id, predecessor_sha256
)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT DO NOTHING
`, e.ID, member, r.Kind, r.SHA256, document.Bytes(), actor, r.Predecessor)
	if err != nil {
		return r, normalize(err)
	}
	for _, ref := range refs {
		_, err = s.db.Exec(ctx, `
INSERT INTO eval_evidence_refs(
    experiment_id, member_id, record_sha256, scope_kind, scope_id, namespace, name, revision
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT DO NOTHING
`, e.ID, member, r.SHA256, ref.Scope, ref.ScopeID, ref.Namespace, ref.Name, ref.Revision)
		if err != nil {
			return r, err
		}
	}
	return s.Record(ctx, e.OwnerID, e.ID, member, r.Kind, r.SHA256)
}
func (s *Store) Record(ctx context.Context, owner, id, member, kind, digest string) (Record, error) {
	var r Record
	var doc []byte
	err := s.db.QueryRow(ctx, `
SELECT r.member_id,r.kind,r.record_sha256,r.actor_id,r.predecessor_sha256,r.created_at,r.document
FROM eval_records r
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id = $1
    AND e.experiment_id = $2
    AND r.member_id = $3
    AND r.kind = $4
    AND r.record_sha256 = $5
`, owner, id, member, kind, digest).Scan(&r.MemberID, &r.Kind, &r.SHA256, &r.ActorID, &r.Predecessor, &r.CreatedAt, &doc)
	if err != nil {
		return r, normalize(err)
	}
	dto := "ResultInput"
	if kind == evaldomain.RecordKindAssessment {
		dto = "AssessmentInput"
	}
	r.CreatedAt = r.CreatedAt.UTC()
	r.Document, err = evaldomain.Freeze(dto, doc)
	return r, err
}

func (s *Store) LatestNativeRecord(ctx context.Context, owner, id, member, kind string) (*Record, error) {
	var digest string
	err := s.db.QueryRow(ctx, `
SELECT r.record_sha256
FROM eval_records r
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id = $1
    AND e.experiment_id = $2
    AND r.member_id = $3
    AND r.kind = $4
    AND r.actor_id = 'system:eval-collector'
ORDER BY r.created_at DESC,r.record_sha256 DESC
LIMIT 1
`, owner, id, member, kind).Scan(&digest)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	r, err := s.Record(ctx, owner, id, member, kind, digest)
	return &r, err
}

func (s *Store) LatestRecord(ctx context.Context, owner, id, member, kind string) (*Record, error) {
	var digest string
	err := s.db.QueryRow(ctx, `
SELECT r.record_sha256
FROM eval_records r
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id = $1
    AND e.experiment_id = $2
    AND r.member_id = $3
    AND r.kind = $4
ORDER BY r.created_at DESC,r.record_sha256 DESC
LIMIT 1
`, owner, id, member, kind).Scan(&digest)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	r, err := s.Record(ctx, owner, id, member, kind, digest)
	return &r, err
}
func (s *Store) SaveNativeRecord(ctx context.Context, scope Scope, id, member string, claim Claim, document evaldomain.Frozen) (Record, error) {
	e, err := s.LockCollection(ctx, scope, id, claim)
	if err != nil {
		return Record{}, err
	}
	if e.ControlMode != "server" {
		return Record{}, evaldomain.Failure("eval_external_control")
	}
	return s.insertRecord(ctx, e, member, evaldomain.NativeCollectorActor, document)
}
