package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type executionMemberJSON struct {
	ExecutionItemID string          `json:"execution_item_id"`
	ItemID          string          `json:"item_id"`
	BatchOrdinal    int             `json:"batch_ordinal"`
	ItemAttempt     int             `json:"item_attempt"`
	TaskRef         json.RawMessage `json:"task_ref"`
	TaskDigest      string          `json:"task_digest"`
	Inputs          []ExactArtifact `json:"inputs"`
}

func (s *PostgresStore) CreateExecutionIntent(
	ctx context.Context,
	params CreateExecutionIntentParams,
) (Execution, bool, error) {
	if err := validateExecutionIntent(params); err != nil {
		return Execution{}, false, err
	}
	if replay, found, err := s.lookupExecutionReplay(
		ctx, params.Claim.AuditID, params.SubmissionKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	manifestRef, _ := json.Marshal(params.Manifest.Ref)
	members := make([]executionMemberJSON, len(params.Members))
	for index, member := range params.Members {
		taskRef, _ := json.Marshal(member.Task.Ref)
		members[index] = executionMemberJSON{
			ExecutionItemID: member.ExecutionItemID, ItemID: member.ItemID,
			BatchOrdinal: member.BatchOrdinal, ItemAttempt: member.ItemAttempt,
			TaskRef: taskRef, TaskDigest: member.Task.Digest,
			Inputs: append([]ExactArtifact{}, member.Inputs...),
		}
	}
	encodedMembers, _ := json.Marshal(members)
	var roundID *string
	if params.RoundID != nil {
		value := *params.RoundID
		roundID = &value
	}
	var roleAttempt *int
	if params.RoleAttempt != nil {
		value := *params.RoleAttempt
		roleAttempt = &value
	}
	execution, err := scanExecution(s.db.QueryRow(ctx, `
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
       AND audit.deadline_at > clock_timestamp()
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
       AND audit.deadline_at > clock_timestamp()
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
SELECT `+prefixedExecutionColumns("inserted_execution")+`
  FROM inserted_execution`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExecutionID, string(params.Role), roundID, roleAttempt,
		manifestRef, params.Manifest.Digest, params.SubmissionKey,
		params.RequestDigest, encodedMembers,
		params.WorkflowRole,
	))
	if err == nil {
		return execution, true, nil
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return Execution{}, false, ErrProjectDeleting
	}
	sqlState := persistencepostgres.SQLState(err)
	if sqlState == "23505" || errors.Is(err, pgx.ErrNoRows) {
		if existing, found, replayErr := s.lookupExecutionReplay(
			ctx, params.Claim.AuditID, params.SubmissionKey, params.RequestDigest,
		); replayErr != nil || found {
			return existing, false, replayErr
		}
		if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
			return Execution{}, false, liveErr
		} else if !live {
			return Execution{}, false, ErrClaimLost
		}
		if sqlState == "23505" {
			return Execution{}, false, ErrConflict
		}
		return Execution{}, false, ErrPrecondition
	}
	return Execution{}, false, fmt.Errorf("create Audit execution intent: %w", err)
}

func prefixedExecutionColumns(prefix string) string {
	return prefix + ".execution_id, " + prefix + ".audit_id, " + prefix + ".round_id, " +
		prefix + ".role, " + prefix + ".workflow_role, " + prefix + ".role_attempt, " + prefix + ".manifest_ref, " +
		prefix + ".manifest_digest, " + prefix + ".submission_key, " + prefix + ".request_digest, " +
		prefix + ".run_id, " + prefix + ".state, " + prefix + ".terminal_outcome, " +
		prefix + ".terminal_run_generation, " + prefix + ".terminal_run_sequence, " +
		prefix + ".terminal_observed_at, " + prefix + ".run_provenance, " + prefix + ".run_deleted_at, " +
		prefix + ".created_at, " + prefix + ".updated_at"
}

func (s *PostgresStore) lookupExecutionReplay(
	ctx context.Context, auditID, submissionKey, digest string,
) (Execution, bool, error) {
	execution, err := scanExecution(s.db.QueryRow(ctx, `
SELECT `+executionColumns+` FROM audit_executions
 WHERE submission_key = $1`, submissionKey))
	if errors.Is(err, pgx.ErrNoRows) {
		return Execution{}, false, nil
	}
	if err != nil {
		return Execution{}, false, fmt.Errorf("resolve execution submission replay: %w", err)
	}
	if execution.AuditID != auditID || execution.RequestDigest != digest {
		return Execution{}, true, ErrConflict
	}
	return execution, true, nil
}

func (s *PostgresStore) claimLive(ctx context.Context, claim ControllerClaim) (bool, error) {
	var live bool
	err := s.db.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1 FROM audit_controller_claims
     WHERE audit_id = $1 AND holder_id = $2 AND epoch = $3
       AND expires_at > clock_timestamp()
)`, claim.AuditID, claim.HolderID, claim.Epoch).Scan(&live)
	if err != nil {
		return false, fmt.Errorf("check Audit claim: %w", err)
	}
	return live, nil
}

func (s *PostgresStore) BindRun(ctx context.Context, params BindRunParams) (Execution, error) {
	if err := validateBindRun(params); err != nil {
		return Execution{}, err
	}
	execution, err := scanExecution(s.db.QueryRow(ctx, `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), eligible AS MATERIALIZED (
    SELECT execution.execution_id, execution.audit_id,
           jsonb_build_object(
               'schema', 'contractor.audit.run-provenance.v1',
               'runId', run.run_id,
               'workflow', jsonb_build_object(
                   'name', run.workflow_name,
                   'version', run.workflow_version,
                   'schemaVersion', run.workflow_schema_version,
                   'configurationRef', jsonb_build_object(
                       'name', run.workflow_name,
                       'version', run.workflow_version
                   ),
                   'closureDigest', 'sha256:' || encode(
                       pg_catalog.sha256(convert_to(run.workflow_snapshot::text, 'UTF8')),
                       'hex'
                   )
               )
           ) AS run_provenance,
           contractor_require_active_audit_project(audit.project_id, audit.owner_id)
      FROM audit_executions AS execution
      JOIN audits AS audit USING (audit_id)
      JOIN live_claim USING (audit_id)
      JOIN workflow_runs AS run
        ON run.run_id = $5 AND run.owner_id = audit.owner_id
       AND run.project_id = audit.project_id AND run.state = 'initializing'
     WHERE execution.execution_id = $4 AND execution.audit_id = $1
       AND execution.state = 'intent' AND execution.run_id IS NULL
       AND audit.state = 'active' AND audit.dispatch_state = 'open'
       AND audit.deadline_at > clock_timestamp()
     FOR UPDATE OF audit, execution
), advanced_audit AS (
    UPDATE audits AS audit
       SET submitted_run_count = audit.submitted_run_count + 1,
           revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
     FROM eligible
     WHERE audit.audit_id = eligible.audit_id
       AND audit.submitted_run_count < audit.reserved_run_count
       AND audit.state = 'active' AND audit.dispatch_state = 'open'
       AND audit.deadline_at > clock_timestamp()
    RETURNING audit.audit_id, audit.next_event_sequence
), changed AS (
    UPDATE audit_executions AS execution
       SET run_id = $5, run_provenance = eligible.run_provenance,
           state = 'submitted', updated_at = clock_timestamp()
      FROM eligible JOIN advanced_audit USING (audit_id)
     WHERE execution.execution_id = eligible.execution_id
    RETURNING execution.*
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, summary)
    SELECT changed.audit_id, advanced_audit.next_event_sequence - 1,
           'execution.run_bound', changed.execution_id,
           jsonb_build_object('runId', $5::text)
      FROM changed JOIN advanced_audit USING (audit_id)
)
SELECT `+prefixedExecutionColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExecutionID, params.RunID,
	))
	if err == nil {
		return execution, nil
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return Execution{}, ErrProjectDeleting
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		if persistencepostgres.SQLState(err) == "23505" {
			return Execution{}, ErrConflict
		}
		return Execution{}, fmt.Errorf("bind Audit execution Run: %w", err)
	}
	existing, getErr := s.getExecution(ctx, params.Claim.AuditID, params.ExecutionID)
	if getErr == nil && existing.RunID != nil && *existing.RunID == params.RunID && existing.State != ExecutionIntent {
		return existing, nil
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Execution{}, liveErr
	} else if !live {
		return Execution{}, ErrClaimLost
	}
	if getErr != nil {
		return Execution{}, getErr
	}
	return Execution{}, ErrPrecondition
}

// GetRunCreationIntent locks the exact claim, Audit and Execution used by one
// trusted child-Run submission. Submitted executions remain readable so a
// response-loss replay can return the already-associated Run without touching
// mutable configuration.
func (s *PostgresStore) GetRunCreationIntent(
	ctx context.Context,
	claim ControllerClaim,
	executionID string,
) (RunCreationIntent, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return RunCreationIntent{}, err
	}
	if err := validateID("executionID", executionID); err != nil {
		return RunCreationIntent{}, err
	}
	var result RunCreationIntent
	execution, err := scanExecutionWithOwner(s.db.QueryRow(ctx, `
SELECT audit.owner_id, audit.project_id, `+prefixedExecutionColumns("execution")+`
  FROM audit_executions AS execution
  JOIN audits AS audit USING (audit_id)
  JOIN audit_controller_claims AS claim USING (audit_id)
 WHERE execution.audit_id = $1 AND execution.execution_id = $4
   AND claim.holder_id = $2 AND claim.epoch = $3
   AND claim.expires_at > clock_timestamp()
 FOR UPDATE OF claim, audit, execution`,
		claim.AuditID, claim.HolderID, claim.Epoch, executionID,
	), &result.OwnerID, &result.ProjectID)
	if errors.Is(err, pgx.ErrNoRows) {
		if live, liveErr := s.claimLive(ctx, claim); liveErr != nil {
			return RunCreationIntent{}, liveErr
		} else if !live {
			return RunCreationIntent{}, ErrClaimLost
		}
		return RunCreationIntent{}, ErrNotFound
	}
	if err != nil {
		return RunCreationIntent{}, fmt.Errorf("read Audit Run creation intent: %w", err)
	}
	result.Execution = execution
	result.Items, err = s.ListExecutionItems(ctx, executionID)
	if err != nil {
		return RunCreationIntent{}, err
	}
	return result, nil
}

func scanExecutionWithOwner(row scanner, ownerID, projectID *string) (Execution, error) {
	var execution Execution
	var encodedRef []byte
	var state string
	var role string
	var roleAttempt *int
	var outcome *string
	var terminalSequence *int64
	var encodedProvenance []byte
	if err := row.Scan(
		ownerID, projectID,
		&execution.ExecutionID, &execution.AuditID, &execution.RoundID,
		&role, &execution.WorkflowRole, &roleAttempt, &encodedRef,
		&execution.Manifest.Digest, &execution.SubmissionKey, &execution.RequestDigest,
		&execution.RunID, &state, &outcome,
		&execution.TerminalRunGeneration, &terminalSequence,
		&execution.TerminalObservedAt, &encodedProvenance, &execution.RunDeletedAt,
		&execution.CreatedAt, &execution.UpdatedAt,
	); err != nil {
		return Execution{}, err
	}
	execution.Role = ExecutionRole(role)
	execution.State = ExecutionState(state)
	execution.RoleAttempt = roleAttempt
	if err := json.Unmarshal(encodedRef, &execution.Manifest.Ref); err != nil || execution.Manifest.Ref.ValidateExact() != nil {
		return Execution{}, errors.New("stored Audit execution manifest ref is invalid")
	}
	if validateDigest("stored execution manifest digest", execution.Manifest.Digest) != nil ||
		validateDigest("stored execution request digest", execution.RequestDigest) != nil ||
		!execution.Role.Valid() || !execution.State.Valid() ||
		validateText("stored execution Workflow role", execution.WorkflowRole, 128, true) != nil {
		return Execution{}, errors.New("stored Audit execution is invalid")
	}
	if outcome != nil {
		value := TerminalOutcome(*outcome)
		if !value.Valid() {
			return Execution{}, errors.New("stored Audit execution terminal outcome is invalid")
		}
		execution.TerminalOutcome = &value
	}
	if terminalSequence != nil {
		if *terminalSequence < 0 {
			return Execution{}, errors.New("stored Audit execution terminal sequence is invalid")
		}
		value := uint64(*terminalSequence)
		execution.TerminalRunSequence = &value
	}
	if err := decodeRunProvenance(encodedProvenance, execution.RunID, &execution.RunProvenance); err != nil {
		return Execution{}, err
	}
	return execution, nil
}

func (s *PostgresStore) ObserveTerminal(
	ctx context.Context,
	params ObserveTerminalParams,
) (Execution, error) {
	if err := validateObserveTerminal(params); err != nil {
		return Execution{}, err
	}
	execution, err := scanExecution(s.db.QueryRow(ctx, `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), observation_gate AS MATERIALIZED (
    SELECT execution.execution_id, execution.audit_id,
           run.state AS terminal_outcome,
           run.run_event_generation AS terminal_run_generation,
           run.next_run_event_sequence - 1 AS terminal_run_sequence
      FROM audit_executions AS execution
      JOIN audits AS audit USING (audit_id)
      JOIN live_claim USING (audit_id)
      JOIN workflow_runs AS run ON run.run_id = execution.run_id
     WHERE execution.execution_id = $4 AND execution.audit_id = $1
       AND execution.run_id = $5 AND execution.state = 'submitted'
       AND run.state IN ('succeeded', 'failed', 'cancelled')
       AND run.run_event_generation = $6
       AND run.next_run_event_sequence - 1 = $7
     FOR UPDATE OF audit, execution
), changed AS (
    UPDATE audit_executions AS execution
       SET state = 'collecting',
           terminal_outcome = observed.terminal_outcome,
           terminal_run_generation = observed.terminal_run_generation,
           terminal_run_sequence = observed.terminal_run_sequence,
           terminal_observed_at = clock_timestamp(),
           updated_at = clock_timestamp()
      FROM observation_gate AS observed
     WHERE execution.execution_id = observed.execution_id
       AND execution.audit_id = observed.audit_id
       AND execution.state = 'submitted'
    RETURNING execution.*
), collecting_items AS (
    UPDATE audit_execution_items AS item
       SET state = 'collecting'
      FROM changed
     WHERE item.execution_id = changed.execution_id AND item.state = 'submitted'
), collecting_domain_items AS (
    UPDATE audit_items AS item
       SET state = 'collecting',
           updated_at = GREATEST(clock_timestamp(), item.updated_at + interval '1 microsecond')
      FROM audit_execution_items AS member, changed
     WHERE member.execution_id = changed.execution_id
       AND item.item_id = member.item_id AND item.state = 'submitted'
), advanced_audit AS (
    UPDATE audits AS audit
       SET outstanding_run_count = audit.outstanding_run_count - 1,
           revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM changed
     WHERE audit.audit_id = changed.audit_id
    RETURNING audit.audit_id, audit.next_event_sequence
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, summary)
    SELECT changed.audit_id, advanced_audit.next_event_sequence - 1,
           'execution.terminal_observed', changed.execution_id,
           jsonb_build_object(
               'outcome', changed.terminal_outcome,
               'generation', changed.terminal_run_generation,
               'sequence', changed.terminal_run_sequence
           )
      FROM changed JOIN advanced_audit USING (audit_id)
)
SELECT `+prefixedExecutionColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExecutionID, params.RunID, params.Generation, int64(params.Sequence),
	))
	if err == nil {
		return execution, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Execution{}, fmt.Errorf("observe terminal Audit Run: %w", err)
	}
	existing, getErr := s.getExecution(ctx, params.Claim.AuditID, params.ExecutionID)
	if getErr == nil && (existing.State == ExecutionCollecting || existing.State == ExecutionCollected) && existing.RunID != nil &&
		*existing.RunID == params.RunID && existing.TerminalRunGeneration != nil &&
		*existing.TerminalRunGeneration == params.Generation && existing.TerminalRunSequence != nil &&
		*existing.TerminalRunSequence == params.Sequence {
		return existing, nil
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Execution{}, liveErr
	} else if !live {
		return Execution{}, ErrClaimLost
	}
	if getErr != nil {
		return Execution{}, getErr
	}
	return Execution{}, ErrPrecondition
}

func (s *PostgresStore) ObserveSubmissionFailure(
	ctx context.Context,
	params ObserveSubmissionFailureParams,
) (Execution, error) {
	if err := validateSubmissionFailure(params); err != nil {
		return Execution{}, err
	}
	execution, err := scanExecution(s.db.QueryRow(ctx, `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), observation_gate AS MATERIALIZED (
    SELECT execution.execution_id, execution.audit_id
      FROM audit_executions AS execution
      JOIN audits AS audit USING (audit_id)
      JOIN live_claim USING (audit_id)
     WHERE execution.execution_id = $4 AND execution.audit_id = $1
       AND execution.state = 'intent' AND execution.run_id IS NULL
     FOR UPDATE OF audit, execution
), changed AS (
    UPDATE audit_executions AS execution
       SET state = 'collecting', terminal_outcome = 'submission-failed',
           terminal_observed_at = clock_timestamp(), updated_at = clock_timestamp()
      FROM observation_gate AS observed
     WHERE execution.execution_id = observed.execution_id
       AND execution.audit_id = observed.audit_id
       AND execution.state = 'intent' AND execution.run_id IS NULL
    RETURNING execution.*
), collecting_items AS (
    UPDATE audit_execution_items AS item SET state = 'collecting'
      FROM changed
     WHERE item.execution_id = changed.execution_id AND item.state = 'submitted'
), collecting_domain_items AS (
    UPDATE audit_items AS item
       SET state = 'collecting',
           updated_at = GREATEST(clock_timestamp(), item.updated_at + interval '1 microsecond')
      FROM audit_execution_items AS member, changed
     WHERE member.execution_id = changed.execution_id
       AND item.item_id = member.item_id AND item.state = 'submitted'
), advanced_audit AS (
    UPDATE audits AS audit
       SET outstanding_run_count = audit.outstanding_run_count - 1,
           revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM changed
     WHERE audit.audit_id = changed.audit_id
    RETURNING audit.audit_id, audit.next_event_sequence
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, summary)
    SELECT changed.audit_id, advanced_audit.next_event_sequence - 1,
           'execution.terminal_observed', changed.execution_id,
           jsonb_build_object('outcome', 'submission-failed')
      FROM changed JOIN advanced_audit USING (audit_id)
)
SELECT `+prefixedExecutionColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch, params.ExecutionID,
	))
	if err == nil {
		return execution, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Execution{}, fmt.Errorf("observe Audit submission failure: %w", err)
	}
	existing, getErr := s.getExecution(ctx, params.Claim.AuditID, params.ExecutionID)
	if getErr == nil && (existing.State == ExecutionCollecting || existing.State == ExecutionCollected) && existing.TerminalOutcome != nil &&
		*existing.TerminalOutcome == TerminalSubmissionFailed {
		return existing, nil
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Execution{}, liveErr
	} else if !live {
		return Execution{}, ErrClaimLost
	}
	if getErr != nil {
		return Execution{}, getErr
	}
	return Execution{}, ErrPrecondition
}

func (s *PostgresStore) getExecution(ctx context.Context, auditID, executionID string) (Execution, error) {
	execution, err := scanExecution(s.db.QueryRow(ctx, `
SELECT `+executionColumns+` FROM audit_executions
 WHERE audit_id = $1 AND execution_id = $2`, auditID, executionID))
	if errors.Is(err, pgx.ErrNoRows) {
		return Execution{}, ErrNotFound
	}
	if err != nil {
		return Execution{}, fmt.Errorf("read Audit execution: %w", err)
	}
	return execution, nil
}
