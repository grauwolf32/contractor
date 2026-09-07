-- Omission preserves all existing Runs and allocation records.
ALTER TABLE workflow_runs ADD COLUMN audit_completion jsonb
 CHECK (audit_completion IS NULL OR
   (publication_mode = 'audit-managed' AND jsonb_typeof(audit_completion) = 'object'
    AND audit_completion #>> '{contract,kind}' = 'audit-check-results@1'));
ALTER TABLE stage_allocations ADD COLUMN completion_contract jsonb
 CHECK (completion_contract IS NULL OR
   (jsonb_typeof(completion_contract) = 'object'
    AND completion_contract ->> 'kind' = 'audit-check-results@1'));

CREATE FUNCTION contractor_protect_audit_completion() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.audit_completion IS DISTINCT FROM OLD.audit_completion AND
    (OLD.audit_completion IS NOT NULL OR OLD.state <> 'initializing') THEN
  RAISE EXCEPTION 'Audit completion is immutable' USING ERRCODE = '23514';
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER workflow_runs_protect_audit_completion BEFORE UPDATE ON workflow_runs
 FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_completion();

-- Read the final row so Run creation, input forks and completion pinning must
-- commit together with the existing deferred AuditExecution ownership checks.
CREATE FUNCTION contractor_require_audit_completion() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE stored workflow_runs%ROWTYPE; binding jsonb;
BEGIN
 IF TG_OP = 'UPDATE' AND OLD.audit_completion IS NULL AND NEW.audit_completion IS NULL THEN RETURN NULL; END IF;
 SELECT * INTO stored FROM workflow_runs WHERE run_id = NEW.run_id;
 IF NOT FOUND OR stored.publication_mode <> 'audit-managed' THEN RETURN NULL; END IF;
 SELECT a.profile_snapshot #> ARRAY['workflows', e.workflow_role] INTO binding
 FROM audit_executions e JOIN audits a USING (audit_id)
 WHERE e.execution_id = stored.audit_execution_id AND e.role = 'check';
 IF (binding -> 'workerCompletion') IS NOT NULL AND binding -> 'workerCompletion' <> 'null'::jsonb THEN
  IF stored.audit_completion IS NULL OR
     stored.audit_completion ->> 'stage' IS DISTINCT FROM binding #>> '{workerCompletion,stage}' OR
     stored.audit_completion ->> 'agent' IS DISTINCT FROM binding #>> '{workerCompletion,agent}' OR
     stored.audit_completion #>> '{contract,kind}' IS DISTINCT FROM binding #>> '{workerCompletion,kind}' THEN
   RAISE EXCEPTION 'Audit completion does not match owned profile' USING ERRCODE = '23514';
  END IF;
 ELSIF stored.audit_completion IS NOT NULL THEN
  RAISE EXCEPTION 'Audit completion has no owned check binding' USING ERRCODE = '23514';
 END IF;
 RETURN NULL;
END;
$$;
CREATE CONSTRAINT TRIGGER workflow_runs_require_audit_completion
 AFTER INSERT OR UPDATE ON workflow_runs DEFERRABLE INITIALLY DEFERRED
 FOR EACH ROW EXECUTE FUNCTION contractor_require_audit_completion();

CREATE FUNCTION contractor_require_allocation_completion() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE pinned jsonb; stage_name text;
BEGIN
 IF TG_OP = 'UPDATE' AND NEW.completion_contract IS DISTINCT FROM OLD.completion_contract THEN
  RAISE EXCEPTION 'Allocation completion is immutable' USING ERRCODE = '23514';
 END IF;
 SELECT r.audit_completion, s.stage_name INTO pinned, stage_name
 FROM stage_executions s JOIN workflow_runs r USING (run_id)
 WHERE s.stage_execution_id = NEW.stage_execution_id;
 IF pinned ->> 'stage' = stage_name AND pinned ->> 'agent' = NEW.logical_agent_name THEN
  IF NEW.completion_contract IS DISTINCT FROM pinned -> 'contract' THEN
   RAISE EXCEPTION 'Allocation completion differs from Run authority' USING ERRCODE = '23514';
  END IF;
 ELSIF NEW.completion_contract IS NOT NULL THEN
  RAISE EXCEPTION 'Allocation has no completion authority' USING ERRCODE = '23514';
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER stage_allocations_require_completion BEFORE INSERT OR UPDATE ON stage_allocations
 FOR EACH ROW EXECUTE FUNCTION contractor_require_allocation_completion();
