-- Partial indexes over the active WorkflowRun states (V367-003).
--
-- The active states are initializing, pending, running, waiting and
-- cancelling (RUN-R30). The queue and credential-reference indexes kept the
-- three-state predicate written before pending and waiting existed. A query
-- over the five active states does not imply that predicate, so PostgreSQL
-- could not use these indexes and read the whole Run history instead. They
-- are recreated over the five states that runstore defines;
-- TestPostgresRunStateSetsMatchSchema pins every workflow_runs predicate to
-- that definition.

-- The queue page selects metadata labels, so it can never be an index-only
-- scan; the old INCLUDE list only made every Run event append, which bumps
-- next_run_event_sequence, a non-HOT update of all workflow_runs indexes.
DROP INDEX workflow_runs_owner_nonterminal_queue_idx;
CREATE INDEX workflow_runs_owner_nonterminal_queue_idx
    ON workflow_runs USING btree (owner_id, created_at, run_id)
    WHERE state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling');

DROP INDEX workflow_runs_runtime_llm_credentials_idx;
CREATE INDEX workflow_runs_runtime_llm_credentials_idx
    ON workflow_runs USING gin ((runtime_config_snapshot -> 'llmCredentialIds'))
    WHERE state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling');

-- The LLM credential check also matches any credentialId in the Workflow
-- snapshot. This index holds every credentialId value and everything nested
-- in it, a superset of what that test matches, so the check can combine it
-- with workflow_runs_runtime_llm_credentials_idx instead of reading every
-- active Run's Workflow snapshot.
CREATE INDEX workflow_runs_workflow_credentials_idx
    ON workflow_runs USING gin (jsonb_path_query_array(workflow_snapshot, '$.**.credentialId.**', '{}', true))
    WHERE state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling');

DROP INDEX workflow_runs_runtime_credentials_idx;
CREATE INDEX workflow_runs_runtime_credentials_idx
    ON workflow_runs USING gin ((runtime_config_snapshot -> 'runtimeCredentialIds'))
    WHERE state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling');

DROP INDEX workflow_runs_project_http_target_credential_idx;
CREATE INDEX workflow_runs_project_http_target_credential_idx
    ON workflow_runs USING btree ((project_http_target_snapshot #>> '{credential,credentialId}'), created_at, run_id)
    WHERE state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling')
      AND project_http_target_snapshot ? 'credential';

-- workflow_runs_admission_idx and workflow_runs_skill_initialization_scheduler_idx
-- serve the Scheduler claim. No query restricts Runs to running and
-- cancelling, the predicate of this older index, so it only cost writes.
DROP INDEX workflow_runs_scheduler_idx;
