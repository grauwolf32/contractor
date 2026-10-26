-- The private finding index names byte ranges of one exact proposal, never
-- alternative proposal prose. Existing retained receipts require the explicit
-- bounded data migration, which reads either configured physical blob backend.
-- NULL is migration-pending, not permission to parse an unindexed proposal.
ALTER TABLE finding_proposal_receipts ADD COLUMN checks_projection json;
ALTER TABLE finding_proposal_receipts ADD CONSTRAINT finding_checks_projection_shape CHECK (
 checks_projection IS NULL OR COALESCE((
  octet_length(checks_projection::text)<=262144
  AND json_typeof(checks_projection)='object'
  AND checks_projection->>'schema'='contractor.audit.finding-checks-projection.v1'
  AND checks_projection->>'proposal_digest'=proposal_digest
  AND (checks_projection->>'size_bytes')::bigint=proposal_size_bytes
  AND json_typeof(checks_projection->'checks')='array'
  AND json_array_length(checks_projection->'checks')<=512
  AND json_typeof(checks_projection->'discarded')='boolean'
  AND (checks_projection->>'discarded')::boolean=(retention_state='discarded')
 ),false)
);
CREATE INDEX finding_checks_projection_pending_idx ON finding_proposal_receipts(receipt_id)
 WHERE checks_projection IS NULL;
ALTER TABLE finding_proposal_receipts ADD COLUMN checks_projection_bytes integer GENERATED ALWAYS AS (octet_length(checks_projection::text)) STORED;
ALTER TABLE finding_proposal_receipts ADD COLUMN checks_projection_count integer GENERATED ALWAYS AS (json_array_length(checks_projection->'checks')) STORED;

CREATE FUNCTION contractor_empty_finding_checks(digest text, bytes bigint) RETURNS json
 LANGUAGE sql IMMUTABLE AS $$
 SELECT json_build_object('schema','contractor.audit.finding-checks-projection.v1',
  'proposal_digest',digest,'size_bytes',bytes,'discarded',true,
  'subject',NULL,'limitations',NULL,'checks','[]'::json)
$$;

-- Only the one-time NULL-to-index derivation may change a retained receipt.
-- Discard erases all derived content, including byte indexes of proposal prose.
CREATE OR REPLACE FUNCTION contractor_protect_finding_proposal_receipt_retention() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='UPDATE' THEN
  IF NEW.retention_state='discarded' THEN
   NEW.report_projection:='{}';
   NEW.anchor_projection:='{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"discarded"}';
   NEW.fingerprint_projection:='{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable","reason":"discarded"}';
   NEW.checks_projection:=contractor_empty_finding_checks(NEW.proposal_digest,NEW.proposal_size_bytes);
  END IF;
  IF NEW.comparison_projection IS DISTINCT FROM OLD.comparison_projection
   AND NEW.comparison_projection IS DISTINCT FROM contractor_finding_comparison_projection(NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state) THEN
   RAISE EXCEPTION 'finding comparison projection does not match its immutable sources' USING ERRCODE='23514';
  END IF;
  NEW.comparison_projection:=contractor_finding_comparison_projection(NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state);
  IF to_jsonb(NEW)-ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at','report_projection','fingerprint_projection','anchor_projection','comparison_projection','checks_projection']
   =to_jsonb(OLD)-ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at','report_projection','fingerprint_projection','anchor_projection','comparison_projection','checks_projection']
   AND (NEW.report_projection=OLD.report_projection OR NEW.retention_state='discarded')
   AND (NEW.anchor_projection=OLD.anchor_projection OR NEW.retention_state='discarded')
   AND (NEW.fingerprint_projection=OLD.fingerprint_projection OR NEW.retention_state='discarded')
   AND (NEW.checks_projection::text IS NOT DISTINCT FROM OLD.checks_projection::text OR OLD.checks_projection IS NULL OR NEW.retention_state='discarded') THEN RETURN NEW; END IF;
 END IF;
 IF TG_OP='DELETE' AND current_setting('contractor.lifecycle_purge',true)='project'
  AND EXISTS(SELECT 1 FROM projects WHERE project_id=OLD.project_id AND lifecycle_state='deleting') THEN RETURN OLD; END IF;
 RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE='23514';
END;
$$;
UPDATE finding_proposal_receipts SET checks_projection=contractor_empty_finding_checks(proposal_digest,proposal_size_bytes)
 WHERE retention_state='discarded';
ALTER TABLE finding_proposal_receipts ADD CONSTRAINT finding_checks_projection_discarded CHECK (
 retention_state<>'discarded' OR COALESCE((checks_projection IS NOT NULL
 AND checks_projection->>'discarded'='true' AND json_array_length(checks_projection->'checks')=0
 AND checks_projection->>'subject' IS NULL AND checks_projection->>'limitations' IS NULL),false)
);

-- Progress belongs to the closed source Round. Proposed successors stay
-- non-current and invisible until one fixed-size acceptance publishes them.
ALTER TABLE audit_rounds ADD COLUMN admission_checkpoint json;
ALTER TABLE audit_rounds ADD CONSTRAINT audit_round_admission_checkpoint_shape CHECK (
 admission_checkpoint IS NULL OR COALESCE((json_typeof(admission_checkpoint)='object'
 AND octet_length(admission_checkpoint::text)<=65536),false)
);
ALTER TABLE audit_items ADD COLUMN accepted_plan json;
ALTER TABLE audit_items ADD COLUMN prepared_page_ordinal integer;
ALTER TABLE audit_items ADD COLUMN prepared_execution json;
ALTER TABLE audit_items ADD COLUMN prepared_related_count integer;
ALTER TABLE audit_items ADD COLUMN prepared_projection_bytes integer GENERATED ALWAYS AS (octet_length(prepared_execution::text)) STORED;
ALTER TABLE audit_items ADD CONSTRAINT audit_item_accepted_plan_shape CHECK (
 accepted_plan IS NULL OR COALESCE((json_typeof(accepted_plan)='object'
 AND octet_length(accepted_plan::text)<=65536
 AND accepted_plan->>'schema'='contractor.audit.accepted-plan.v1'
 AND accepted_plan->>'audit_id'=audit_id AND accepted_plan->>'item_key'=item_key),false)
);
ALTER TABLE audit_items ADD CONSTRAINT audit_item_prepared_page_shape CHECK (
 (accepted_plan IS NULL)=(prepared_page_ordinal IS NULL)
 AND (prepared_page_ordinal IS NULL OR prepared_page_ordinal BETWEEN 0 AND 4095)
 AND (prepared_execution IS NULL)=(accepted_plan IS NULL)
 AND (prepared_related_count IS NULL)=(accepted_plan IS NULL)
 AND COALESCE(prepared_related_count BETWEEN 0 AND 512,true)
);
CREATE INDEX finding_holds_admission_cutoff_idx ON finding_proposal_audit_holds(audit_id,created_at,receipt_id);
CREATE INDEX audit_admission_pages_idx ON audit_artifact_links
 (audit_id,(source_provenance->>'generation'),(source_provenance->>'kind'),(source_provenance->>'ordinal'))
 WHERE source_provenance->>'schema'='contractor.audit.admission-provenance.v1';
CREATE INDEX audit_admission_decisions_idx ON audit_artifact_links
 (audit_id,(source_provenance->>'source_key'),(source_provenance->>'policy_digest'),(source_provenance->>'disposition'))
 WHERE source_provenance->>'schema'='contractor.audit.admission-decision.v1';

-- A private proposed item cannot expose or exercise review authority.
CREATE OR REPLACE VIEW audit_live_item_reviews AS
 SELECT item.audit_id, item.item_id, request.request_id,
        request.state='decided' AS approved
 FROM audit_items AS item
 JOIN audit_rounds AS round USING (round_id,audit_id)
 JOIN audit_review_requests AS request ON request.audit_id=item.audit_id
  AND request.subject_kind='audit-item-action' AND request.subject_id=item.item_id
  AND request.kind=item.approval_kind AND request.subject_revision=1
  AND request.subject_digest=item.approval_subject_digest
 WHERE round.state<>'proposed'
  AND (request.expires_at IS NULL OR request.expires_at>clock_timestamp())
  AND (request.state='pending' OR (request.state='decided' AND EXISTS (
   SELECT 1 FROM audit_review_decisions AS decision
   WHERE decision.request_id=request.request_id AND decision.audit_id=request.audit_id
    AND decision.action='approve' AND decision.subject_revision=request.subject_revision
    AND decision.subject_digest=request.subject_digest)));
