-- A discarded receipt keeps exact replay provenance, never proposal prose.
-- Audit-held reports remain immutable until the final hold is removed.
CREATE OR REPLACE FUNCTION contractor_protect_finding_proposal_receipt_retention() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='UPDATE' THEN
  IF NEW.retention_state='discarded' THEN
   NEW.report_projection:='{}';
   NEW.anchor_projection:='{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"discarded"}';
   NEW.fingerprint_projection:='{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable","reason":"discarded"}';
  END IF;
  IF NEW.comparison_projection IS DISTINCT FROM OLD.comparison_projection
   AND NEW.comparison_projection IS DISTINCT FROM contractor_finding_comparison_projection(NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state) THEN
   RAISE EXCEPTION 'finding comparison projection does not match its immutable sources' USING ERRCODE='23514';
  END IF;
  NEW.comparison_projection:=contractor_finding_comparison_projection(NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state);
  IF to_jsonb(NEW)-ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at','report_projection','fingerprint_projection','anchor_projection','comparison_projection']
   =to_jsonb(OLD)-ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at','report_projection','fingerprint_projection','anchor_projection','comparison_projection']
   AND (NEW.report_projection=OLD.report_projection OR NEW.retention_state='discarded')
   AND (NEW.anchor_projection=OLD.anchor_projection OR NEW.retention_state='discarded')
   AND (NEW.fingerprint_projection=OLD.fingerprint_projection OR NEW.retention_state='discarded') THEN RETURN NEW; END IF;
 END IF;
 IF TG_OP='DELETE' AND current_setting('contractor.lifecycle_purge',true)='project'
  AND EXISTS(SELECT 1 FROM projects WHERE project_id=OLD.project_id AND lifecycle_state='deleting') THEN RETURN OLD; END IF;
 RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE='23514';
END;
$$;

-- Use the retention trigger for historical tombstones too, so all derived
-- projections share the same discard rule without changing identity or timestamps.
UPDATE finding_proposal_receipts
   SET report_projection = '{}'
 WHERE retention_state = 'discarded' AND report_projection <> '{}';

-- The trigger normalizes UPDATEs; reject content supplied in a direct INSERT
-- of a discarded tombstone as well.
ALTER TABLE finding_proposal_receipts
 ADD CONSTRAINT finding_proposal_receipts_discarded_report_check
 CHECK (retention_state <> 'discarded' OR report_projection = '{}');
