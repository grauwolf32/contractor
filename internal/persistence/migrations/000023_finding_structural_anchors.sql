-- Retained deterministic derivations share the existing receipt lifecycle.
ALTER TABLE finding_proposal_receipts ADD COLUMN anchor_projection text NOT NULL
 DEFAULT '{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"not_recorded"}';
ALTER TABLE finding_proposal_receipts ADD CONSTRAINT finding_anchor_projection_shape CHECK ((
 octet_length(anchor_projection) BETWEEN 2 AND 32768
 AND jsonb_typeof(anchor_projection::jsonb) = 'object'
 AND anchor_projection::jsonb->>'schema' = 'contractor.finding.anchors-projection.v1'
 AND anchor_projection::jsonb->>'availability' IN ('available', 'unavailable')
) IS TRUE);

CREATE OR REPLACE FUNCTION contractor_protect_finding_proposal_receipt_retention() RETURNS trigger
 LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP = 'UPDATE' THEN
  IF NEW.retention_state = 'discarded' THEN
   NEW.anchor_projection := '{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"discarded"}';
   NEW.fingerprint_projection := '{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable","reason":"discarded"}';
  END IF;
  IF to_jsonb(NEW) - ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at','fingerprint_projection','anchor_projection']
    = to_jsonb(OLD) - ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at','fingerprint_projection','anchor_projection']
    AND (NEW.anchor_projection = OLD.anchor_projection OR NEW.retention_state = 'discarded')
    AND (NEW.fingerprint_projection = OLD.fingerprint_projection OR NEW.retention_state = 'discarded') THEN
   RETURN NEW;
  END IF;
 END IF;
 IF TG_OP = 'DELETE' AND current_setting('contractor.lifecycle_purge', true) = 'project'
  AND EXISTS (SELECT 1 FROM projects WHERE project_id = OLD.project_id AND lifecycle_state = 'deleting') THEN
  RETURN OLD;
 END IF;
 RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE = '23514';
END;
$$;
