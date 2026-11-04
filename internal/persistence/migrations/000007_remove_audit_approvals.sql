-- The earlier pre-release cutover retains old Audit facts as opaque Project
-- Artifacts. Refuse an incomplete cutover; never discard surviving decisions or
-- reinterpret consent as active authority. Existing Artifact bytes are untouched.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM audit_review_requests)
       OR EXISTS (SELECT 1 FROM audit_review_decisions)
       OR EXISTS (SELECT 1 FROM audits WHERE state='waiting_review')
       OR EXISTS (SELECT 1 FROM audit_items
           WHERE state='awaiting_review' OR approval_kind<>'none'
              OR approval_subject_digest IS NOT NULL) THEN
        RAISE EXCEPTION 'unarchived Audit approval records block approval removal'
            USING ERRCODE='23514';
    END IF;
END;
$$;

DROP VIEW audit_live_item_reviews;
DROP TABLE audit_review_decisions;
DROP TABLE audit_review_requests;

-- The common Start path already prohibits approval-required Projects. Preserve
-- that effective restriction; only an explicit owner edit can allow operations.
UPDATE projects SET active_checks='prohibited', revision=revision+1,
    updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond')
    WHERE active_checks='approval-required';
ALTER TABLE projects DROP CONSTRAINT projects_active_checks_policy;
ALTER TABLE projects ADD CONSTRAINT projects_active_checks_policy
    CHECK(active_checks IN ('automatic','prohibited'));

DROP INDEX audit_items_pending_approval_idx;
ALTER TABLE audit_items
    DROP COLUMN approval_kind,
    DROP COLUMN approval_subject_digest,
    DROP CONSTRAINT audit_items_state_check;
ALTER TABLE audit_items ADD CONSTRAINT audit_items_state_check
    CHECK(state IN ('pending','ready','submitted','collecting','settled'));
ALTER TABLE audit_rounds DROP COLUMN prepared_review_count;

ALTER TABLE audits DROP CONSTRAINT audits_state_check, DROP CONSTRAINT audits_time_shape;
ALTER TABLE audits ADD CONSTRAINT audits_state_check
    CHECK(state IN ('draft','active','paused','finalizing','cancelling','completed','cancelled','failed','deleting'));
ALTER TABLE audits ADD CONSTRAINT audits_time_shape CHECK(
    (state='draft' AND started_at IS NULL AND finished_at IS NULL)
    OR (state IN ('active','paused','finalizing','cancelling') AND started_at IS NOT NULL AND finished_at IS NULL)
    OR (state IN ('completed','cancelled','failed') AND started_at IS NOT NULL AND finished_at IS NOT NULL)
    OR state='deleting');
DROP INDEX audits_reconcile_idx;
CREATE INDEX audits_reconcile_idx ON audits(updated_at,audit_id)
    WHERE state IN ('active','paused','finalizing','cancelling','deleting');

CREATE OR REPLACE FUNCTION contractor_protect_prepared_audit_item() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
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
    NEW.workflow_role,NEW.task_ref,NEW.task_digest,NEW.origin)
   IS DISTINCT FROM ROW(OLD.item_id,OLD.audit_id,OLD.round_id,OLD.item_key,OLD.ordinal,OLD.kind,OLD.subject_key,
    OLD.workflow_role,OLD.task_ref,OLD.task_digest,OLD.origin) THEN
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
  UPDATE audit_rounds SET prepared_staged_count=prepared_staged_count-1
  WHERE audit_id=OLD.audit_id AND round_id=OLD.round_id AND state='proposed';
 END IF;
 RETURN OLD;
END;
$$;
