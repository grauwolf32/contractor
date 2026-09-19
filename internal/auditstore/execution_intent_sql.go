package auditstore

// createExecutionIntentSQL keeps reservation and complete-batch submission in
// one statement. live_claim, claim_gate and round_gate preserve authority and
// lock order; eligible_members validates exact tasks, approval and attempts.
// reserved gates budgets and round state before inserted_execution, members,
// item transitions and the event can become durable. Replay is checked on both
// sides of this statement by CreateExecutionIntent.
var createExecutionIntentSQL = `
WITH member_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($12::jsonb) AS member(
        execution_item_id text, item_id text, batch_ordinal integer,
        item_attempt integer, task_ref jsonb, task_digest text, inputs jsonb
    )
), live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), claim_gate AS MATERIALIZED (
    SELECT audit.audit_id, audit.current_round_id, audit.batch_size,
           audit.reserved_run_count, audit.outstanding_run_count,
           audit.max_submitted_runs, settings.max_concurrent_runs,
	       audit.profile_snapshot,
           contractor_require_active_audit_project(audit.project_id, audit.owner_id)
      FROM audits AS audit
      JOIN live_claim USING (audit_id)
      CROSS JOIN scheduler_settings AS settings
     WHERE audit.audit_id = $1
       AND settings.singleton = true
       AND audit.state = 'active' AND audit.dispatch_state = 'open'
       AND (audit.deadline_at IS NULL OR audit.deadline_at > clock_timestamp())
	   AND audit.profile_snapshot #>> ARRAY['workflows', $13::text, 'kind'] = $5
     FOR UPDATE OF audit, settings
), round_gate AS MATERIALIZED (
    SELECT round.round_id, round.state
      FROM audit_rounds AS round
      JOIN claim_gate AS audit ON audit.audit_id = round.audit_id
     WHERE $6::text IS NOT NULL AND round.round_id = $6
     FOR UPDATE OF round
), eligible_members AS MATERIALIZED (
    SELECT count(*)::integer AS member_count,
           count(DISTINCT ROW(item.approval_kind, item.approval_subject_digest))::integer AS approval_envelope_count
      FROM audit_items AS item
      JOIN member_input AS member
        ON member.item_id = item.item_id
       AND member.task_ref = item.task_ref
       AND member.task_digest = item.task_digest
     WHERE item.audit_id = $1 AND item.round_id = $6
       AND item.state = 'ready'
	   AND item.workflow_role = $13
	   AND (
	       item.approval_kind = 'none'
	       OR EXISTS (
	           SELECT 1
	             FROM audit_review_requests AS approval
	             JOIN audit_review_decisions AS decision
	               ON decision.request_id = approval.request_id
	              AND decision.audit_id = approval.audit_id
	            WHERE approval.audit_id = item.audit_id
	              AND approval.subject_kind = 'audit-item-action'
	              AND approval.subject_id = item.item_id
	              AND approval.kind = item.approval_kind
	              AND approval.subject_revision = 1
	              AND approval.subject_digest = item.approval_subject_digest
	              AND approval.state = 'decided'
	              AND (approval.expires_at IS NULL OR approval.expires_at > clock_timestamp())
	              AND decision.action = 'approve'
	              AND decision.subject_revision = approval.subject_revision
	              AND decision.subject_digest = approval.subject_digest
	       )
	   )
	   AND member.item_attempt = COALESCE((
             SELECT max(previous.item_attempt) + 1
               FROM audit_execution_items AS previous
              WHERE previous.item_id = item.item_id
           ), 1)
), reserved AS (
    UPDATE audits AS audit
       SET reserved_run_count = audit.reserved_run_count + 1,
           outstanding_run_count = audit.outstanding_run_count + 1,
           revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM claim_gate, eligible_members
     WHERE audit.audit_id = claim_gate.audit_id
       AND audit.reserved_run_count = claim_gate.reserved_run_count
       AND audit.outstanding_run_count = claim_gate.outstanding_run_count
       AND audit.reserved_run_count < audit.max_submitted_runs
       AND audit.outstanding_run_count < claim_gate.max_concurrent_runs
       AND audit.state = 'active' AND audit.dispatch_state = 'open'
       AND (audit.deadline_at IS NULL OR audit.deadline_at > clock_timestamp())
	       AND (
           ($5 = 'check' AND $6::text IS NOT NULL
             AND jsonb_array_length($12::jsonb) > 0
             AND jsonb_array_length($12::jsonb) <= audit.batch_size
             AND eligible_members.member_count = jsonb_array_length($12::jsonb)
             AND eligible_members.approval_envelope_count = 1
             AND NOT EXISTS (
                 SELECT 1 FROM member_input
                  WHERE item_attempt > audit.max_item_run_attempts
             )
             AND audit.current_round_id = $6
             AND EXISTS (
                 SELECT 1 FROM round_gate
                  WHERE round_gate.round_id = $6 AND round_gate.state = 'executing'
             ))
	       OR ($5 = 'discovery' AND $6::text IS NOT NULL
	             AND jsonb_array_length($12::jsonb) = 0
	             AND $7::integer BETWEEN 1 AND audit.max_item_run_attempts
	             AND audit.current_round_id = $6
	             AND EXISTS (
	                 SELECT 1 FROM round_gate
	                  WHERE round_gate.round_id = $6 AND round_gate.state = 'accepted'
	             ))
	       OR ($5 = 'assessment' AND $6::text IS NOT NULL
	             AND jsonb_array_length($12::jsonb) = 0
	             AND $7::integer BETWEEN 1 AND audit.max_item_run_attempts
	             AND audit.current_round_id = $6
	             AND EXISTS (
	                 SELECT 1 FROM round_gate
	                  WHERE round_gate.round_id = $6 AND round_gate.state = 'assessing'
	             ))
       )
    RETURNING audit.*
), inserted_execution AS (
    INSERT INTO audit_executions (
        execution_id, audit_id, round_id, role, workflow_role, role_attempt,
        manifest_ref, manifest_digest, submission_key, request_digest
    )
    SELECT $4, audit_id, $6, $5, $13, $7, $8::jsonb, $9, $10, $11
      FROM reserved
    RETURNING *
), inserted_members AS (
    INSERT INTO audit_execution_items (
        execution_item_id, execution_id, audit_id, round_id, item_id,
        batch_ordinal, item_attempt, task_ref, task_digest, input_refs
    )
    SELECT member.execution_item_id, execution.execution_id,
           execution.audit_id, execution.round_id, member.item_id,
           member.batch_ordinal, member.item_attempt, member.task_ref,
           member.task_digest, member.inputs
      FROM inserted_execution AS execution CROSS JOIN member_input AS member
    RETURNING execution_item_id, item_id
), marked_items AS (
    UPDATE audit_items AS item
       SET state = 'submitted',
           last_execution_item_id = member.execution_item_id,
           updated_at = GREATEST(clock_timestamp(), item.updated_at + interval '1 microsecond')
      FROM inserted_members AS member
     WHERE item.item_id = member.item_id AND item.state = 'ready'
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, summary
    )
    SELECT execution.audit_id, reserved.next_event_sequence - 1,
           'execution.intent_created', execution.execution_id,
           jsonb_build_object(
	           'role', execution.role,
	           'workflowRole', execution.workflow_role,
	           'members', jsonb_array_length($12::jsonb)
	       )
      FROM inserted_execution AS execution JOIN reserved USING (audit_id)
)
SELECT ` + prefixedExecutionColumns("inserted_execution") + `
  FROM inserted_execution`
