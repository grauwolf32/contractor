package auditstore

// SQL statements for execution.go.

// bindRunSQL binds initializing child Run $5 to intent execution $4 under the
// live controller claim ($1-$3), gated on an active, open-dispatch Audit before
// its deadline and an active Project. It takes one reserved Run slot, records
// Run provenance, marks the execution submitted and appends
// execution.run_bound; the caller completes the trailing SELECT.
// Used by PostgresStore.BindRun.
var bindRunSQL = `
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
       AND (audit.deadline_at IS NULL OR audit.deadline_at > clock_timestamp())
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
       AND (audit.deadline_at IS NULL OR audit.deadline_at > clock_timestamp())
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
SELECT `

// observeTerminalSQL records that Run $5 of submitted execution $4 finished at
// event generation $6 and last sequence $7, under the live controller claim
// ($1-$3). It moves the execution and its submitted items to collecting, stores
// the terminal outcome, decrements outstanding_run_count and appends
// execution.terminal_observed; the caller completes the trailing SELECT.
// Used by PostgresStore.ObserveTerminal.
var observeTerminalSQL = `
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
SELECT `

// observeSubmissionFailureSQL closes intent execution $4 that never got a Run,
// under the live controller claim ($1-$3): the execution and its submitted
// items move to collecting with outcome submission-failed,
// outstanding_run_count is decremented and execution.terminal_observed is
// appended. The caller completes the trailing SELECT.
// Used by PostgresStore.ObserveSubmissionFailure.
var observeSubmissionFailureSQL = `
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
SELECT `
