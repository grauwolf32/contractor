-- A bounded receipt projection prevents a page read from decoding 200 full
-- proposal/report documents. Source projections remain the single authority.
CREATE FUNCTION contractor_finding_comparison_projection(report text, fingerprint text, anchors text, retention text)
 RETURNS jsonb LANGUAGE plpgsql IMMUTABLE STRICT AS $$
DECLARE
 r jsonb := report::jsonb;
 f jsonb := fingerprint::jsonb;
 a jsonb := anchors::jsonb;
 families jsonb := '[]';
 family_available boolean := true;
 result jsonb;
BEGIN
 IF retention='discarded' THEN
  RETURN '{"schema":"contractor.finding.comparison-projection.v1","availability":"unavailable","reason":"discarded"}'::jsonb;
 END IF;
 IF coalesce(r->>'title','')='' THEN
  RETURN '{"schema":"contractor.finding.comparison-projection.v1","availability":"unavailable","reason":"not_recorded"}'::jsonb;
 END IF;
 IF jsonb_array_length(coalesce(r->'standard_refs','[]'))>32 OR EXISTS (
  SELECT 1 FROM jsonb_array_elements(coalesce(r->'standard_refs','[]')) AS ref
  WHERE octet_length(ref->>'scheme')>256 OR octet_length(ref->>'version')>160 OR octet_length(ref->>'requirement_id')>256
 ) THEN family_available:=false;
 ELSE
  SELECT coalesce(jsonb_agg(ref ORDER BY (ref->>'scheme') COLLATE "C",(ref->>'version') COLLATE "C",(ref->>'requirement_id') COLLATE "C"),'[]') INTO families
  FROM jsonb_array_elements(coalesce(r->'standard_refs','[]')) AS ref;
 END IF;
 result:=jsonb_build_object('schema','contractor.finding.comparison-projection.v1','availability','available',
  'preview',jsonb_build_object('title',left(r->>'title',256),'titleTruncated',length(r->>'title')>256,
   'severitySuggestion',coalesce(r->>'severity_suggestion','')),
  'hints',jsonb_build_object('subject',r->'subject','families',families,'familiesAvailable',family_available,
   'producerKey',CASE WHEN f->>'availability'='available' AND f->>'algorithm'='contractor/producer-issue/v1'
    AND f->>'adapter'='pinned-sarif-native/v1' THEN f->>'value' ELSE '' END,
   'sites',CASE WHEN a->>'availability'='available' THEN coalesce((SELECT jsonb_agg(k->>'siteKey') FROM jsonb_array_elements(a->'keys') AS k),'[]') ELSE '[]' END,
   'repository',coalesce(f->>'repository',a->>'repository',''),
   'commit',coalesce(f->>'commit',a->>'commit','')));
 RETURN result;
END;
$$;
ALTER TABLE finding_proposal_receipts ADD COLUMN comparison_projection jsonb NOT NULL
 DEFAULT '{"schema":"contractor.finding.comparison-projection.v1","availability":"unavailable","reason":"not_recorded"}';
ALTER TABLE finding_proposal_receipts ADD CONSTRAINT finding_comparison_projection_shape CHECK (
 jsonb_typeof(comparison_projection)='object' AND octet_length(comparison_projection::text)<=32768
 AND comparison_projection->>'schema'='contractor.finding.comparison-projection.v1'
 AND comparison_projection->>'availability' IN ('available','unavailable'));

CREATE OR REPLACE FUNCTION contractor_protect_finding_proposal_receipt_retention() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='UPDATE' THEN
  IF NEW.retention_state='discarded' THEN
   NEW.anchor_projection:='{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"discarded"}';
   NEW.fingerprint_projection:='{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable","reason":"discarded"}';
  END IF;
  IF NEW.comparison_projection IS DISTINCT FROM OLD.comparison_projection
   AND NEW.comparison_projection IS DISTINCT FROM contractor_finding_comparison_projection(NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state) THEN
   RAISE EXCEPTION 'finding comparison projection does not match its immutable sources' USING ERRCODE='23514';
  END IF;
  NEW.comparison_projection:=contractor_finding_comparison_projection(NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state);
  IF to_jsonb(NEW)-ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at','fingerprint_projection','anchor_projection','comparison_projection']
   =to_jsonb(OLD)-ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at','fingerprint_projection','anchor_projection','comparison_projection']
   AND (NEW.anchor_projection=OLD.anchor_projection OR NEW.retention_state='discarded')
   AND (NEW.fingerprint_projection=OLD.fingerprint_projection OR NEW.retention_state='discarded') THEN RETURN NEW; END IF;
 END IF;
 IF TG_OP='DELETE' AND current_setting('contractor.lifecycle_purge',true)='project'
  AND EXISTS(SELECT 1 FROM projects WHERE project_id=OLD.project_id AND lifecycle_state='deleting') THEN RETURN OLD; END IF;
 RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE='23514';
END;
$$;
UPDATE finding_proposal_receipts SET comparison_projection=contractor_finding_comparison_projection(report_projection,fingerprint_projection,anchor_projection,retention_state);
CREATE FUNCTION contractor_initialize_finding_comparison_projection() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 NEW.comparison_projection:=contractor_finding_comparison_projection(NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state);
 RETURN NEW;
END;
$$;
CREATE TRIGGER finding_receipts_initialize_comparison BEFORE INSERT ON finding_proposal_receipts
 FOR EACH ROW EXECUTE FUNCTION contractor_initialize_finding_comparison_projection();

CREATE INDEX audit_findings_correlation_audit_group_idx
 ON audit_findings(audit_id,correlation_group_key COLLATE "C",finding_id COLLATE "C");
CREATE INDEX audit_findings_correlation_project_scan_idx
 ON audit_findings(correlation_project_id,created_at,finding_id COLLATE "C");
CREATE INDEX audit_findings_correlation_audit_scan_idx
 ON audit_findings(audit_id,created_at,finding_id COLLATE "C");
CREATE INDEX audit_findings_correlation_project_group_order_idx
 ON audit_findings(correlation_project_id,correlation_group_key,created_at,finding_id COLLATE "C") WHERE correlation_group_key<>'';
CREATE INDEX audit_findings_correlation_audit_group_order_idx
 ON audit_findings(audit_id,correlation_group_key COLLATE "C",created_at,finding_id COLLATE "C") WHERE correlation_group_key<>'';
