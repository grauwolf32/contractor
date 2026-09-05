package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type materializedItemJSON struct {
	ItemID       string          `json:"item_id"`
	ItemKey      string          `json:"item_key"`
	Ordinal      int             `json:"ordinal"`
	Kind         string          `json:"kind"`
	SubjectKey   string          `json:"subject_key"`
	TaskRef      json.RawMessage `json:"task_ref"`
	TaskDigest   string          `json:"task_digest"`
	Origin       json.RawMessage `json:"origin"`
	WorkflowRole string          `json:"workflow_role"`
	InitialState string          `json:"initial_state"`
	Status       string          `json:"status"`
	Requested    []string        `json:"requested"`
	Completed    []string        `json:"completed"`
	Gaps         []string        `json:"gaps"`
	Rationale    string          `json:"rationale"`
}

func (s *PostgresStore) MaterializeRound(
	ctx context.Context,
	params MaterializeRoundParams,
) (Audit, bool, error) {
	if err := validateMaterialize(params); err != nil {
		return Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, "audit.start", params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	encodedManifestRef, _ := json.Marshal(params.Manifest.Ref)
	items := make([]materializedItemJSON, len(params.Items))
	for index, item := range params.Items {
		encodedTaskRef, _ := json.Marshal(item.Task.Ref)
		encodedOrigin, _ := json.Marshal(item.Origin)
		items[index] = materializedItemJSON{
			ItemID: item.ItemID, ItemKey: item.ItemKey, Ordinal: item.Ordinal,
			Kind: item.Kind, SubjectKey: item.SubjectKey, TaskRef: encodedTaskRef,
			TaskDigest: item.Task.Digest, Origin: encodedOrigin, WorkflowRole: item.WorkflowRole,
			InitialState: string(item.InitialState), Status: string(item.Coverage.Status),
			Requested: nonNilStrings(item.Coverage.Requested), Completed: nonNilStrings(item.Coverage.Completed),
			Gaps: nonNilStrings(item.Coverage.Gaps), Rationale: item.Coverage.Rationale,
		}
	}
	encodedItems, _ := json.Marshal(items)
	response, _ := json.Marshal(map[string]string{"auditId": params.AuditID, "roundId": params.RoundID})
	audit, err := scanAudit(s.db.QueryRow(ctx, `
WITH project_gate AS MATERIALIZED (
    SELECT contractor_require_active_audit_project(project_id, owner_id)
      FROM audits
     WHERE audit_id = $2 AND owner_id = $1
), started AS (
    UPDATE audits AS audit
       SET baseline_snapshot = $7::jsonb,
           state = 'active', current_round_id = $4,
           hold_state = 'held', deadline_at = $8,
           started_at = clock_timestamp(),
           revision = revision + 1,
           next_event_sequence = next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
      FROM project_gate
     WHERE audit.owner_id = $1 AND audit.audit_id = $2
       AND audit.revision = $3 AND audit.state = 'draft'
       AND $5 <= audit.max_rounds
       AND jsonb_array_length($9::jsonb) <= audit.max_items_per_round
       AND jsonb_array_length($9::jsonb) <= audit.max_items_total
    RETURNING audit.*
), inserted_round AS (
    INSERT INTO audit_rounds (
        round_id, audit_id, ordinal, manifest_ref, manifest_digest,
        state, expected_item_count
    )
    SELECT $4, audit_id, $5, $6::jsonb, $10,
           'accepted', jsonb_array_length($9::jsonb)
      FROM started
    RETURNING round_id, audit_id
), item_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($9::jsonb) AS item(
        item_id text, item_key text, ordinal integer, kind text,
        subject_key text, task_ref jsonb, task_digest text, origin jsonb,
        workflow_role text, initial_state text, status text,
        requested jsonb, completed jsonb, gaps jsonb, rationale text
    )
), inserted_items AS (
    INSERT INTO audit_items (
        item_id, audit_id, round_id, item_key, ordinal, kind, subject_key,
        task_ref, task_digest, origin, workflow_role, state
    )
    SELECT item.item_id, round.audit_id, round.round_id,
           item.item_key, item.ordinal, item.kind, item.subject_key,
           item.task_ref, item.task_digest, item.origin, item.workflow_role, item.initial_state
      FROM inserted_round AS round CROSS JOIN item_input AS item
    RETURNING item_id, audit_id, round_id, item_key, subject_key
), inserted_coverage AS (
    INSERT INTO audit_coverage_rows (
        audit_id, round_id, item_id, item_key, subject_key,
        status, requested, completed, gaps, rationale
    )
    SELECT stored.audit_id, stored.round_id, stored.item_id,
           stored.item_key, stored.subject_key,
           source.status, source.requested, source.completed, source.gaps, source.rationale
      FROM inserted_items AS stored
      JOIN item_input AS source USING (item_id)
), idempotency_row AS (
    INSERT INTO audit_idempotency (
        owner_id, operation, idempotency_key, request_digest,
        audit_id, resource_id, response_snapshot
    )
    SELECT $1, 'audit.start', $11, $12, audit_id, $4, $13::jsonb
      FROM started
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT audit_id, next_event_sequence - 1, 'round.accepted', $4, 1,
           jsonb_build_object('round', $5::integer, 'items', jsonb_array_length($9::jsonb))
      FROM started
)
SELECT `+prefixedAuditColumns("started")+` FROM started`,
		params.OwnerID, params.AuditID, params.ExpectedRevision,
		params.RoundID, params.RoundOrdinal, encodedManifestRef,
		[]byte(params.BaselineSnapshot), params.DeadlineAt, encodedItems,
		params.Manifest.Digest, params.IdempotencyKey, params.RequestDigest, response,
	))
	if err == nil {
		return audit, true, nil
	}
	switch persistencepostgres.SQLState(err) {
	case "55000":
		return Audit{}, false, ErrProjectDeleting
	case "23505":
		return s.replayAudit(ctx, params.OwnerID, "audit.start", params.IdempotencyKey, params.RequestDigest)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, false, fmt.Errorf("materialize Audit round: %w", err)
	}
	if replay, found, replayErr := s.lookupAuditReplay(
		ctx, params.OwnerID, "audit.start", params.IdempotencyKey, params.RequestDigest,
	); replayErr != nil || found {
		return replay, false, replayErr
	}
	if _, getErr := s.Get(ctx, params.OwnerID, params.AuditID); errors.Is(getErr, ErrNotFound) {
		return Audit{}, false, ErrNotFound
	} else if getErr != nil {
		return Audit{}, false, getErr
	}
	return Audit{}, false, ErrPrecondition
}

func (s *PostgresStore) TransitionRound(
	ctx context.Context,
	params RoundTransitionParams,
) (Round, error) {
	if err := validateRoundTransition(params); err != nil {
		return Round{}, err
	}
	round, err := scanRound(s.db.QueryRow(ctx, `
	WITH live_claim AS MATERIALIZED (
	    SELECT claim.audit_id
	      FROM audit_controller_claims AS claim
	     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
	       AND claim.expires_at > clock_timestamp()
	     FOR UPDATE OF claim
	), claim_gate AS MATERIALIZED (
	    SELECT audit.audit_id,
	           CASE WHEN $7 IN ('accepted', 'executing', 'assessing')
	                THEN contractor_require_active_audit_project(audit.project_id, audit.owner_id)
	           END
	      FROM audits AS audit
	      JOIN live_claim USING (audit_id)
	     WHERE audit.audit_id = $1
	       AND (
	           $7 NOT IN ('accepted', 'executing', 'assessing')
	           OR (
	               audit.state = 'active' AND audit.dispatch_state = 'open'
	               AND audit.current_round_id = $4
	               AND audit.deadline_at > clock_timestamp()
	           )
	       )
	     FOR UPDATE OF audit
	), changed AS (
    UPDATE audit_rounds AS round
       SET state = $7, revision = round.revision + 1,
           updated_at = GREATEST(clock_timestamp(), round.updated_at + interval '1 microsecond')
      FROM claim_gate
     WHERE round.audit_id = claim_gate.audit_id AND round.round_id = $4
       AND round.revision = $5 AND round.state = $6
    RETURNING round.*
), advanced_audit AS (
    UPDATE audits AS audit
       SET revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM changed
     WHERE audit.audit_id = changed.audit_id
    RETURNING audit.audit_id, audit.next_event_sequence
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, entity_revision, summary)
    SELECT changed.audit_id, advanced.next_event_sequence - 1,
           'round.state_changed', changed.round_id, changed.revision,
           jsonb_build_object('from', $6::text, 'to', $7::text)
      FROM changed JOIN advanced_audit AS advanced USING (audit_id)
)
SELECT changed.round_id, changed.audit_id, changed.ordinal,
       changed.manifest_ref, changed.manifest_digest, changed.state,
       changed.expected_item_count, changed.revision,
       changed.created_at, changed.updated_at
  FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.RoundID, params.ExpectedRevision,
		string(params.ExpectedState), string(params.TargetState),
	))
	if err == nil {
		return round, nil
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return Round{}, ErrProjectDeleting
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Round{}, fmt.Errorf("transition Audit round: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Round{}, liveErr
	} else if !live {
		return Round{}, ErrClaimLost
	}
	existing, getErr := s.GetRound(ctx, params.Claim.AuditID, params.RoundID)
	if getErr != nil {
		return Round{}, getErr
	}
	if existing.State == params.TargetState && existing.Revision == params.ExpectedRevision+1 {
		return existing, nil
	}
	return Round{}, ErrPrecondition
}
