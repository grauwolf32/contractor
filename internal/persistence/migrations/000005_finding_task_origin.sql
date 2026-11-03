-- Retire role-only origin before enabling common Task finding publication.
-- Existing Audit receipts must first be retained/retired through their owner,
-- as in the established Project Finding cutover. Never invent a historical
-- Task identity or silently delete a business fact during schema migration.
DO $$ BEGIN
 IF EXISTS (SELECT 1 FROM finding_proposal_receipts WHERE audit_id IS NOT NULL) THEN
  RAISE EXCEPTION 'retire role-only Audit finding receipts before Task origin cutover' USING ERRCODE='55000';
 END IF;
END $$;

ALTER TABLE finding_proposal_receipts
 DROP CONSTRAINT finding_proposal_receipts_audit_role_check;
ALTER TABLE finding_proposal_receipts RENAME COLUMN audit_role TO audit_task_id;
ALTER TABLE finding_proposal_receipts
 ADD COLUMN audit_hypothesis jsonb,
 ADD CONSTRAINT finding_proposal_receipts_task_id_check CHECK (
   audit_task_id IS NULL OR audit_task_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'),
 ADD CONSTRAINT finding_proposal_receipts_hypothesis_check CHECK (
   audit_hypothesis IS NULL OR (
     audit_id IS NOT NULL AND jsonb_typeof(audit_hypothesis)='object'
     AND audit_hypothesis ?& ARRAY['id','revision'] AND audit_hypothesis-ARRAY['id','revision']='{}'::jsonb
     AND jsonb_typeof(audit_hypothesis->'id')='string'
     AND audit_hypothesis->>'id' ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
     AND jsonb_typeof(audit_hypothesis->'revision')='number'
     AND audit_hypothesis->>'revision' ~ '^[1-9][0-9]*$'
     AND (audit_hypothesis->>'revision')::numeric <= 9007199254740991));

CREATE INDEX finding_proposal_receipts_execution_task_idx
 ON finding_proposal_receipts(audit_execution_id,audit_task_id,created_at,receipt_id)
 WHERE audit_id IS NOT NULL;

-- Both composite FK probes must be bounded when a Task or execution is purged.
DROP INDEX audit_tasks_last_execution_idx;
CREATE INDEX audit_tasks_last_execution_idx ON audit_tasks(last_execution_id,audit_id)
 WHERE last_execution_id IS NOT NULL;
DROP INDEX audit_execution_items_task_attempt_idx;
CREATE UNIQUE INDEX audit_execution_items_task_attempt_idx
 ON audit_execution_items(task_id,audit_id,item_attempt) WHERE task_id IS NOT NULL;

-- Identity protection compares only immutable columns and permits a no-op
-- assignment. No guard fires while the controller binds/collects a child Run.
DROP TRIGGER audit_task_execution_protect_identity ON audit_executions;
CREATE TRIGGER audit_task_execution_protect_identity
 BEFORE UPDATE OF execution_id,audit_id,role,workflow_role,submission_key,request_digest,lane_class
 ON audit_executions FOR EACH ROW
 WHEN ((OLD.role='task' OR NEW.role='task') AND
  ROW(NEW.execution_id,NEW.audit_id,NEW.role,NEW.workflow_role,NEW.submission_key,NEW.request_digest,NEW.lane_class)
  IS DISTINCT FROM
  ROW(OLD.execution_id,OLD.audit_id,OLD.role,OLD.workflow_role,OLD.submission_key,OLD.request_digest,OLD.lane_class))
 EXECUTE FUNCTION contractor_protect_immutable_history();

-- Required completion is derived by the ordinary Workflow resolver and trusted
-- RunService binder, and checked again before allocation. The old SQL reader
-- understood role/workerCompletion profile metadata; common programs have no
-- such metadata. Keep immutable Run completion and exact allocation-to-Run
-- enforcement, without a second SQL interpreter for Workflow tool selections.
DROP TRIGGER workflow_runs_require_audit_completion ON workflow_runs;
DROP TRIGGER workflow_runs_require_audit_completion_on_update ON workflow_runs;
DROP FUNCTION contractor_require_audit_completion();
