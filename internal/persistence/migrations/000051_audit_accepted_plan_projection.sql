-- The immutable prepared execution already carries the accepted plan read by
-- dispatch. Move the standalone column's shape/identity guarantees onto that
-- exact embedded JSON without normalizing or rewriting any stored content.
ALTER TABLE audit_items DROP CONSTRAINT audit_item_accepted_plan_shape;
ALTER TABLE audit_items DROP CONSTRAINT audit_item_prepared_page_shape;
ALTER TABLE audit_items ADD CONSTRAINT audit_item_prepared_plan_shape CHECK (
 prepared_execution IS NULL OR COALESCE((
  json_typeof(prepared_execution)='object'
  AND json_typeof(prepared_execution->'accepted_plan')='object'
  AND octet_length((prepared_execution->'accepted_plan')::text)<=65536
  AND prepared_execution->'accepted_plan'->>'schema'='contractor.audit.accepted-plan.v1'
  AND prepared_execution->'accepted_plan'->>'audit_id'=audit_id
  AND prepared_execution->'accepted_plan'->>'item_key'=item_key
 ),false)
);
ALTER TABLE audit_items ADD CONSTRAINT audit_item_prepared_page_shape CHECK (
 (prepared_execution IS NULL)=(prepared_page_ordinal IS NULL)
 AND (prepared_page_ordinal IS NULL OR prepared_page_ordinal BETWEEN 0 AND 4095)
 AND (prepared_related_count IS NULL)=(prepared_execution IS NULL)
 AND COALESCE(prepared_related_count BETWEEN 0 AND 512,true)
);

-- Keep the existing membership and lifecycle guard; the entire execution
-- projection remains immutable, including its embedded accepted plan.
CREATE OR REPLACE FUNCTION contractor_protect_prepared_audit_item() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='INSERT' THEN
  IF (NEW.prepared_execution IS NOT NULL OR EXISTS(SELECT 1 FROM audit_rounds
   WHERE audit_id=NEW.audit_id AND round_id=NEW.round_id AND prepared_generation IS NOT NULL))
   AND NOT EXISTS(SELECT 1 FROM audit_rounds WHERE audit_id=NEW.audit_id AND round_id=NEW.round_id
    AND state='proposed' AND prepared_page_count IS NULL AND NEW.prepared_execution IS NOT NULL) THEN
   RAISE EXCEPTION 'prepared Audit membership is sealed' USING ERRCODE='23514'; END IF;
  RETURN NEW;
 END IF;
 IF TG_OP='UPDATE' THEN
  IF OLD.prepared_execution IS NOT NULL AND ROW(NEW.item_id,NEW.audit_id,NEW.round_id,NEW.item_key,NEW.ordinal,NEW.kind,NEW.subject_key,
    NEW.workflow_role,NEW.task_ref,NEW.task_digest,NEW.origin,NEW.approval_kind,NEW.approval_subject_digest)
   IS DISTINCT FROM ROW(OLD.item_id,OLD.audit_id,OLD.round_id,OLD.item_key,OLD.ordinal,OLD.kind,OLD.subject_key,
    OLD.workflow_role,OLD.task_ref,OLD.task_digest,OLD.origin,OLD.approval_kind,OLD.approval_subject_digest) THEN
   RAISE EXCEPTION 'prepared Audit item authority is immutable' USING ERRCODE='23514'; END IF;
  IF ROW(NEW.prepared_execution::text,NEW.prepared_page_ordinal,NEW.prepared_related_count)
   IS DISTINCT FROM ROW(OLD.prepared_execution::text,OLD.prepared_page_ordinal,OLD.prepared_related_count) THEN
   RAISE EXCEPTION 'prepared Audit membership is immutable' USING ERRCODE='23514'; END IF;
  RETURN NEW;
 END IF;
 IF OLD.prepared_execution IS NOT NULL THEN
  IF EXISTS(SELECT 1 FROM audit_rounds WHERE audit_id=OLD.audit_id AND round_id=OLD.round_id AND state<>'proposed')
   AND EXISTS(SELECT 1 FROM audits WHERE audit_id=OLD.audit_id AND state<>'deleting') THEN
   RAISE EXCEPTION 'accepted prepared Audit membership cannot be deleted' USING ERRCODE='23514'; END IF;
  UPDATE audit_rounds SET prepared_staged_count=prepared_staged_count-1,
   prepared_review_count=prepared_review_count-CASE WHEN OLD.approval_kind='none' THEN 0 ELSE 1 END
  WHERE audit_id=OLD.audit_id AND round_id=OLD.round_id AND state='proposed';
 END IF;
 RETURN OLD;
END;
$$;

ALTER TABLE audit_items DROP COLUMN accepted_plan;
