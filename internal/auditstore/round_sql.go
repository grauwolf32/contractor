package auditstore

// Round statements. Each is one atomic statement whose CTE order is the
// lock order: the controller claim and the audit revision gate every write
// that follows, so an incomplete batch leaves no partial durable effects.
// Keep the CTEs in place when editing; see round.go for the callers.

// materializeRoundSQL opens the first round of an audit: it gates on the
// project being active, pins the manifest and items, and returns the
// started audit row.
var materializeRoundSQL = `
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
SELECT ` + prefixedAuditColumns("started") + ` FROM started`

// acceptNextRoundSQL closes the previous round and opens its successor
// under one controller claim, carrying proposal sources forward.
var acceptNextRoundSQL = `
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
       AND (audit.deadline_at IS NULL OR audit.deadline_at > clock_timestamp())
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
  FROM inserted_round`

// acceptedRoundDigestSQL reads the stored acceptance digest for the replay
// check that decides whether a repeated accept is idempotent.
var acceptedRoundDigestSQL = `
SELECT acceptance_digest
  FROM audit_rounds
 WHERE audit_id = $1 AND round_id = $2`

// transitionRoundSQL moves a round between states under the controller
// claim and the expected round revision.
var transitionRoundSQL = `
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
	               AND (audit.deadline_at IS NULL OR audit.deadline_at > clock_timestamp())
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
  FROM changed`
