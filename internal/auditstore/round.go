package auditstore

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type materializedItemJSON struct {
	ItemID          string                   `json:"item_id"`
	ItemKey         string                   `json:"item_key"`
	Ordinal         int                      `json:"ordinal"`
	Kind            string                   `json:"kind"`
	SubjectKey      string                   `json:"subject_key"`
	TaskRef         json.RawMessage          `json:"task_ref"`
	TaskDigest      string                   `json:"task_digest"`
	Origin          json.RawMessage          `json:"origin"`
	WorkflowRole    string                   `json:"workflow_role"`
	InitialState    string                   `json:"initial_state"`
	ApprovalKind    string                   `json:"approval_kind"`
	ApprovalDigest  string                   `json:"approval_digest,omitempty"`
	Status          string                   `json:"status"`
	Requested       []string                 `json:"requested"`
	Completed       []string                 `json:"completed"`
	Gaps            []string                 `json:"gaps"`
	Rationale       string                   `json:"rationale"`
	ProposalSources []proposalItemSourceJSON `json:"proposal_sources"`
}

type proposalItemSourceJSON struct {
	ItemID               string        `json:"item_id"`
	ReceiptID            string        `json:"receipt_id"`
	ProposedCheckOrdinal int           `json:"proposed_check_ordinal"`
	Proposal             ExactArtifact `json:"proposal"`
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
		approvalKind := item.ApprovalKind
		if approvalKind == "" {
			approvalKind = ItemApprovalNone
		}
		items[index] = materializedItemJSON{
			ItemID: item.ItemID, ItemKey: item.ItemKey, Ordinal: item.Ordinal,
			Kind: item.Kind, SubjectKey: item.SubjectKey, TaskRef: encodedTaskRef,
			TaskDigest: item.Task.Digest, Origin: encodedOrigin, WorkflowRole: item.WorkflowRole,
			InitialState: string(item.InitialState), ApprovalKind: string(approvalKind),
			ApprovalDigest: item.ApprovalDigest, Status: string(item.Coverage.Status),
			Requested: nonNilStrings(item.Coverage.Requested), Completed: nonNilStrings(item.Coverage.Completed),
			Gaps: nonNilStrings(item.Coverage.Gaps), Rationale: item.Coverage.Rationale,
			ProposalSources: []proposalItemSourceJSON{},
		}
	}
	encodedItems, _ := json.Marshal(items)
	links := make([]artifactLinkJSON, len(params.InitialRetained))
	for index, link := range params.InitialRetained {
		ref, _ := json.Marshal(link.Artifact.Ref)
		links[index] = artifactLinkJSON{
			LogicalKey: link.LogicalKey, ArtifactRef: ref,
			ArtifactDigest: link.Artifact.Digest, MediaType: link.Artifact.MediaType,
			SizeBytes: link.Artifact.SizeBytes, SourceProvenance: link.SourceProvenance,
			DisplayRef: link.DisplayRef,
		}
	}
	encodedLinks, _ := json.Marshal(links)
	retainedBytes, _ := validateArtifactLinks(params.InitialRetained)
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
	           retained_evidence_bytes = $15,
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
	       AND $15 <= audit.max_evidence_bytes
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
        workflow_role text, initial_state text, approval_kind text,
        approval_digest text, status text,
        requested jsonb, completed jsonb, gaps jsonb, rationale text
    )
), inserted_items AS (
    INSERT INTO audit_items (
        item_id, audit_id, round_id, item_key, ordinal, kind, subject_key,
        task_ref, task_digest, origin, workflow_role, state,
        approval_kind, approval_subject_digest
    )
    SELECT item.item_id, round.audit_id, round.round_id,
           item.item_key, item.ordinal, item.kind, item.subject_key,
           item.task_ref, item.task_digest, item.origin, item.workflow_role,
           item.initial_state, item.approval_kind,
           NULLIF(item.approval_digest, '')
      FROM inserted_round AS round CROSS JOIN item_input AS item
    RETURNING item_id, audit_id, round_id, item_key, subject_key,
              approval_kind, approval_subject_digest
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
), inserted_reviews AS (
    INSERT INTO audit_review_requests (
        request_id, audit_id, finding_id, subject_kind, subject_id, kind,
        subject_revision, subject_digest, requested_actions, state,
        expires_at, idempotency_key, request_digest
    )
    SELECT 'review-' || item.item_id, item.audit_id, NULL,
           'audit-item-action', item.item_id, item.approval_kind,
           1, item.approval_subject_digest,
           CASE WHEN item.approval_kind = 'requirement-applicability'
                THEN '["approve","reject","not_applicable"]'::jsonb
                ELSE '["approve","reject"]'::jsonb END,
           'pending', started.deadline_at, 'auto:' || item.item_id,
           item.approval_subject_digest
      FROM inserted_items AS item JOIN started USING (audit_id)
	     WHERE item.approval_kind <> 'none'
	    RETURNING request_id
), link_input AS MATERIALIZED (
	SELECT * FROM jsonb_to_recordset($14::jsonb) AS link(
	    logical_key text, artifact_ref jsonb, artifact_digest text,
	    media_type text, size_bytes bigint, source_provenance jsonb, display_ref text
	)
), inserted_links AS (
	INSERT INTO audit_artifact_links (
	    audit_id, logical_key, artifact_ref, artifact_digest,
	    media_type, size_bytes, source_provenance, display_ref
	)
	SELECT started.audit_id, link.logical_key, link.artifact_ref,
	       link.artifact_digest, link.media_type, link.size_bytes,
	       link.source_provenance, link.display_ref
	  FROM started CROSS JOIN link_input AS link
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
           jsonb_build_object(
               'round', $5::integer, 'items', jsonb_array_length($9::jsonb),
               'reviews', (SELECT count(*) FROM inserted_reviews)
           )
      FROM started
)
SELECT `+prefixedAuditColumns("started")+` FROM started`,
		params.OwnerID, params.AuditID, params.ExpectedRevision,
		params.RoundID, params.RoundOrdinal, encodedManifestRef,
		[]byte(params.BaselineSnapshot), params.DeadlineAt, encodedItems,
		params.Manifest.Digest, params.IdempotencyKey, params.RequestDigest, response,
		encodedLinks, retainedBytes,
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

func (s *PostgresStore) AcceptNextRound(
	ctx context.Context,
	params AcceptRoundParams,
) (Round, bool, error) {
	if err := validateAcceptRound(params); err != nil {
		return Round{}, false, err
	}
	if replay, found, err := s.lookupAcceptedRoundReplay(ctx, params); err != nil || found {
		return replay, false, err
	}
	acceptanceDigest, err := roundAcceptanceDigest(params)
	if err != nil {
		return Round{}, false, err
	}
	encodedManifestRef, _ := json.Marshal(params.Manifest.Ref)
	items := make([]materializedItemJSON, len(params.Items))
	sources := make([]proposalItemSourceJSON, 0, len(params.Items))
	for index, item := range params.Items {
		encodedTaskRef, _ := json.Marshal(item.Task.Ref)
		encodedOrigin, _ := json.Marshal(item.Origin)
		approvalKind := item.ApprovalKind
		if approvalKind == "" {
			approvalKind = ItemApprovalNone
		}
		itemSources := make([]proposalItemSourceJSON, len(item.ProposalSources))
		for sourceIndex, source := range item.ProposalSources {
			itemSources[sourceIndex] = proposalItemSourceJSON{
				ItemID: item.ItemID, ReceiptID: source.ReceiptID,
				ProposedCheckOrdinal: source.ProposedCheckOrdinal, Proposal: source.Proposal,
			}
			sources = append(sources, itemSources[sourceIndex])
		}
		items[index] = materializedItemJSON{
			ItemID: item.ItemID, ItemKey: item.ItemKey, Ordinal: item.Ordinal,
			Kind: item.Kind, SubjectKey: item.SubjectKey, TaskRef: encodedTaskRef,
			TaskDigest: item.Task.Digest, Origin: encodedOrigin, WorkflowRole: item.WorkflowRole,
			InitialState: string(item.InitialState), ApprovalKind: string(approvalKind),
			ApprovalDigest: item.ApprovalDigest, Status: string(item.Coverage.Status),
			Requested: nonNilStrings(item.Coverage.Requested), Completed: nonNilStrings(item.Coverage.Completed),
			Gaps: nonNilStrings(item.Coverage.Gaps), Rationale: item.Coverage.Rationale,
			ProposalSources: itemSources,
		}
	}
	encodedItems, _ := json.Marshal(items)
	encodedSources, _ := json.Marshal(sources)
	round, err := scanRound(s.db.QueryRow(ctx, `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), previous_round AS MATERIALIZED (
    SELECT round.round_id, round.audit_id, round.ordinal
      FROM audit_rounds AS round
      JOIN live_claim USING (audit_id)
     WHERE round.round_id = $5 AND round.state = 'closed'
     FOR UPDATE OF round
), item_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($10::jsonb) AS item(
        item_id text, item_key text, ordinal integer, kind text,
        subject_key text, task_ref jsonb, task_digest text, origin jsonb,
        workflow_role text, initial_state text, approval_kind text,
        approval_digest text, status text,
        requested jsonb, completed jsonb, gaps jsonb, rationale text,
        proposal_sources jsonb
    )
), source_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($11::jsonb) AS source(
        item_id text, receipt_id text, proposed_check_ordinal integer, proposal jsonb
    )
), source_gate AS MATERIALIZED (
    SELECT count(*)::integer AS source_count
      FROM source_input AS source
      JOIN item_input AS item USING (item_id)
      JOIN finding_proposal_audit_holds AS hold
        ON hold.audit_id = $1 AND hold.receipt_id = source.receipt_id
       AND hold.proposal_ref = source.proposal
      JOIN audits AS source_audit
        ON source_audit.audit_id = hold.audit_id
       AND source_audit.project_id = hold.project_id
     WHERE NOT EXISTS (
         SELECT 1 FROM audit_proposal_items AS used
          WHERE used.audit_id = $1 AND used.receipt_id = source.receipt_id
            AND used.proposed_check_ordinal = source.proposed_check_ordinal
     )
), audit_gate AS MATERIALIZED (
    SELECT audit.audit_id, audit.max_items_per_round, audit.max_items_total,
           audit.max_rounds, audit.next_event_sequence,
           contractor_require_active_audit_project(audit.project_id, audit.owner_id)
      FROM audits AS audit
      JOIN live_claim USING (audit_id)
      JOIN previous_round AS previous USING (audit_id)
      CROSS JOIN source_gate
     WHERE audit.audit_id = $1 AND audit.revision = $4
       AND audit.state = 'active' AND audit.dispatch_state = 'open'
       AND audit.current_round_id = previous.round_id
       AND audit.deadline_at > clock_timestamp()
       AND $7 = previous.ordinal + 1 AND $7 <= audit.max_rounds
       AND jsonb_array_length($10::jsonb) > 0
       AND jsonb_array_length($10::jsonb) <= audit.max_items_per_round
       AND (SELECT count(*) FROM audit_items WHERE audit_id = $1)
             + jsonb_array_length($10::jsonb) <= audit.max_items_total
       AND source_gate.source_count = jsonb_array_length($11::jsonb)
       AND jsonb_array_length($11::jsonb) = jsonb_array_length($10::jsonb)
     FOR UPDATE OF audit
), advanced AS (
    UPDATE audits AS audit
       SET current_round_id = $6,
           revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM audit_gate
     WHERE audit.audit_id = audit_gate.audit_id
    RETURNING audit.*
), inserted_round AS (
    INSERT INTO audit_rounds (
        round_id, audit_id, ordinal, manifest_ref, manifest_digest,
        state, expected_item_count, acceptance_digest
    )
    SELECT $6, audit_id, $7, $8::jsonb, $9,
           'accepted', jsonb_array_length($10::jsonb), $12
      FROM advanced
    RETURNING *
), inserted_items AS (
    INSERT INTO audit_items (
        item_id, audit_id, round_id, item_key, ordinal, kind, subject_key,
        task_ref, task_digest, origin, workflow_role, state,
        approval_kind, approval_subject_digest
    )
    SELECT item.item_id, round.audit_id, round.round_id,
           item.item_key, item.ordinal, item.kind, item.subject_key,
           item.task_ref, item.task_digest, item.origin, item.workflow_role,
           item.initial_state, item.approval_kind,
           NULLIF(item.approval_digest, '')
      FROM inserted_round AS round CROSS JOIN item_input AS item
    RETURNING item_id, audit_id, round_id, item_key, subject_key,
              approval_kind, approval_subject_digest
), inserted_coverage AS (
    INSERT INTO audit_coverage_rows (
        audit_id, round_id, item_id, item_key, subject_key,
        status, requested, completed, gaps, rationale
    )
    SELECT stored.audit_id, stored.round_id, stored.item_id,
           stored.item_key, stored.subject_key, source.status,
           source.requested, source.completed, source.gaps, source.rationale
      FROM inserted_items AS stored JOIN item_input AS source USING (item_id)
), inserted_reviews AS (
    INSERT INTO audit_review_requests (
        request_id, audit_id, finding_id, subject_kind, subject_id, kind,
        subject_revision, subject_digest, requested_actions, state,
        expires_at, idempotency_key, request_digest
    )
    SELECT 'review-' || item.item_id, item.audit_id, NULL,
           'audit-item-action', item.item_id, item.approval_kind,
           1, item.approval_subject_digest,
           CASE WHEN item.approval_kind = 'requirement-applicability'
                THEN '["approve","reject","not_applicable"]'::jsonb
                ELSE '["approve","reject"]'::jsonb END,
           'pending', advanced.deadline_at, 'auto:' || item.item_id,
           item.approval_subject_digest
      FROM inserted_items AS item JOIN advanced USING (audit_id)
     WHERE item.approval_kind <> 'none'
    RETURNING request_id
), inserted_sources AS (
    INSERT INTO audit_proposal_items (
        audit_id, receipt_id, proposed_check_ordinal, round_id, item_id,
        proposal_ref, proposal_digest
    )
    SELECT stored.audit_id, source.receipt_id, source.proposed_check_ordinal,
           stored.round_id, stored.item_id, source.proposal->'ref',
           source.proposal->>'digest'
      FROM inserted_items AS stored JOIN source_input AS source USING (item_id)
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT round.audit_id, advanced.next_event_sequence - 1,
           'round.accepted', round.round_id, round.revision,
           jsonb_build_object(
               'round', round.ordinal, 'items', round.expected_item_count,
               'reviews', (SELECT count(*) FROM inserted_reviews)
           )
      FROM inserted_round AS round JOIN advanced USING (audit_id)
)
SELECT round_id, audit_id, ordinal, manifest_ref, manifest_digest, state,
       expected_item_count, revision, created_at, updated_at
  FROM inserted_round`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExpectedAuditRevision, params.PreviousRoundID,
		params.RoundID, params.RoundOrdinal, encodedManifestRef,
		params.Manifest.Digest, encodedItems, encodedSources, acceptanceDigest,
	))
	if err == nil {
		return round, true, nil
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return Round{}, false, ErrProjectDeleting
	}
	if persistencepostgres.SQLState(err) == "23505" || errors.Is(err, pgx.ErrNoRows) {
		if replay, found, replayErr := s.lookupAcceptedRoundReplay(ctx, params); replayErr != nil || found {
			return replay, false, replayErr
		}
		if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
			return Round{}, false, liveErr
		} else if !live {
			return Round{}, false, ErrClaimLost
		}
		if persistencepostgres.SQLState(err) == "23505" {
			return Round{}, false, ErrConflict
		}
		return Round{}, false, ErrPrecondition
	}
	return Round{}, false, fmt.Errorf("accept next Audit round: %w", err)
}

func (s *PostgresStore) lookupAcceptedRoundReplay(
	ctx context.Context, params AcceptRoundParams,
) (Round, bool, error) {
	acceptanceDigest, err := roundAcceptanceDigest(params)
	if err != nil {
		return Round{}, false, err
	}
	round, err := s.GetRound(ctx, params.Claim.AuditID, params.RoundID)
	if errors.Is(err, ErrNotFound) {
		return Round{}, false, nil
	}
	if err != nil {
		return Round{}, false, err
	}
	if round.Ordinal != params.RoundOrdinal || round.ExpectedItemCount != len(params.Items) ||
		round.Manifest.Digest != params.Manifest.Digest || !sameRoundArtifactRef(round.Manifest.Ref, params.Manifest.Ref) {
		return Round{}, true, ErrConflict
	}
	var storedDigest *string
	if err := s.db.QueryRow(ctx, `
SELECT acceptance_digest
  FROM audit_rounds
 WHERE audit_id = $1 AND round_id = $2`, params.Claim.AuditID, params.RoundID).Scan(&storedDigest); err != nil {
		return Round{}, true, err
	}
	if storedDigest == nil || *storedDigest != acceptanceDigest {
		return Round{}, true, ErrConflict
	}
	return round, true, nil
}

func roundAcceptanceDigest(params AcceptRoundParams) (string, error) {
	encoded, err := json.Marshal(struct {
		Schema          string             `json:"schema"`
		PreviousRoundID string             `json:"previousRoundId"`
		RoundID         string             `json:"roundId"`
		RoundOrdinal    int                `json:"roundOrdinal"`
		Manifest        ExactArtifact      `json:"manifest"`
		Items           []MaterializedItem `json:"items"`
	}{
		Schema: "contractor.audit.round-acceptance.v1", PreviousRoundID: params.PreviousRoundID,
		RoundID: params.RoundID, RoundOrdinal: params.RoundOrdinal,
		Manifest: params.Manifest, Items: params.Items,
	})
	if err != nil {
		return "", fmt.Errorf("encode Audit Round acceptance identity: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

func sameRoundArtifactRef(left, right contracts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name &&
		left.Revision != nil && right.Revision != nil && *left.Revision == *right.Revision
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
