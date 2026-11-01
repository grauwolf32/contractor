-- Switching requires a drained Server. Never strand an in-flight old
-- contract or discard an uncollected result; the operator finishes collection
-- with the previous revision before retrying this atomic migration.
LOCK TABLE workflow_runs,audits,audit_executions IN SHARE ROW EXCLUSIVE MODE;
DO $$ BEGIN
 IF EXISTS(SELECT 1 FROM workflow_runs WHERE state NOT IN ('succeeded','failed','cancelled'))
 OR EXISTS(SELECT 1 FROM audit_executions WHERE state<>'collected') THEN
 RAISE EXCEPTION 'finish and collect current Runs before switching to Project Findings' USING ERRCODE='23514';
 END IF;
END; $$;

-- A Finding has one Project identity and one optional current analyst decision.
-- Producer receipts are immutable sources; they own no current problem state.
CREATE INDEX artifact_bindings_project_prefix_idx ON artifact_bindings(scope_id,namespace,name COLLATE "C")
    WHERE scope_kind='project';
DROP TRIGGER finding_proposal_audit_holds_admit_finding ON finding_proposal_audit_holds;
DROP FUNCTION contractor_admit_audit_finding();
CREATE TABLE project_findings (
    finding_id text PRIMARY KEY CHECK (finding_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'),
    project_id text NOT NULL REFERENCES projects(project_id) ON DELETE RESTRICT,
    first_receipt_id text NOT NULL REFERENCES finding_proposal_receipts(receipt_id) ON DELETE RESTRICT,
    first_proposal_ref jsonb NOT NULL CHECK (jsonb_typeof(first_proposal_ref) = 'object' AND octet_length(first_proposal_ref::text) <= 4096),
    title text NOT NULL CHECK (octet_length(title) BETWEEN 1 AND 65536 AND btrim(title)<>''),
    correlation_algorithm text NOT NULL,
    correlation_value text NOT NULL,
    correlation_group_key text NOT NULL,
    correlation_reason text NOT NULL,
    state text CHECK (state IN ('proposed', 'confirmed', 'rejected', 'duplicate', 'needs-evidence')),
    analyst_verdict text CHECK (analyst_verdict IN ('true_positive', 'false_positive', 'duplicate', 'reopen', 'needs_evidence')),
    analyst_severity text CHECK (analyst_severity IN ('informational', 'low', 'medium', 'high', 'critical')),
    rationale text CHECK (octet_length(rationale) BETWEEN 1 AND 65536 AND btrim(rationale) <> ''),
    actor_id text CHECK (octet_length(actor_id) BETWEEN 1 AND 256 AND btrim(actor_id) <> ''),
    assessed_at timestamptz,
    duplicate_target_id text,
    revision bigint NOT NULL DEFAULT 1 CHECK (revision BETWEEN 1 AND 9007199254740991),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (project_id, finding_id),
    FOREIGN KEY (project_id, duplicate_target_id) REFERENCES project_findings(project_id, finding_id) DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT project_finding_primary_shape CHECK (
        (correlation_algorithm = '' AND correlation_value = '' AND correlation_group_key = ''
         AND correlation_reason IN ('not_recorded', 'no_qualified_key', 'primary_key_ambiguous'))
        OR (correlation_algorithm IN ('contractor/producer-issue/v1', 'contractor/snapshot-issue-anchor/v1')
            AND correlation_value ~ '^[0-9a-f]{64}$' AND correlation_reason = ''
            AND correlation_group_key = contractor_finding_group_key(project_id, correlation_algorithm, correlation_value))
    ),
    CONSTRAINT project_finding_assessment_shape CHECK (
        (state IS NULL AND analyst_verdict IS NULL AND analyst_severity IS NULL AND rationale IS NULL
         AND actor_id IS NULL AND assessed_at IS NULL AND duplicate_target_id IS NULL)
        OR (state IS NOT NULL AND analyst_verdict IS NOT NULL AND rationale IS NOT NULL
            AND actor_id IS NOT NULL AND assessed_at IS NOT NULL
            AND state = CASE analyst_verdict WHEN 'true_positive' THEN 'confirmed'
                WHEN 'false_positive' THEN 'rejected' WHEN 'duplicate' THEN 'duplicate'
                WHEN 'reopen' THEN 'proposed' WHEN 'needs_evidence' THEN 'needs-evidence' END
            AND ((analyst_verdict = 'true_positive') = (analyst_severity IS NOT NULL))
            AND ((analyst_verdict = 'duplicate') = (duplicate_target_id IS NOT NULL)))
    ),
    CHECK (duplicate_target_id IS NULL OR duplicate_target_id <> finding_id)
);
CREATE UNIQUE INDEX project_findings_qualified_key_idx ON project_findings(project_id, correlation_group_key)
    WHERE correlation_group_key <> '';
CREATE INDEX project_findings_page_idx ON project_findings(project_id, created_at, finding_id COLLATE "C");
CREATE INDEX project_findings_owner_page_idx ON project_findings(created_at DESC, finding_id COLLATE "C" DESC);
CREATE INDEX project_findings_duplicate_idx ON project_findings(project_id, duplicate_target_id)
    WHERE duplicate_target_id IS NOT NULL;

CREATE TABLE project_finding_sources (
    receipt_id text PRIMARY KEY REFERENCES finding_proposal_receipts(receipt_id) ON DELETE RESTRICT,
    project_id text NOT NULL,
    finding_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    FOREIGN KEY (project_id, finding_id) REFERENCES project_findings(project_id, finding_id) ON DELETE RESTRICT
);
CREATE INDEX project_finding_sources_page_idx ON project_finding_sources(finding_id, created_at, receipt_id COLLATE "C");
CREATE INDEX project_finding_sources_project_idx ON project_finding_sources(project_id, receipt_id);
CREATE TRIGGER project_finding_sources_immutable BEFORE UPDATE ON project_finding_sources
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_proposal_receipt();

-- Audit membership is an execution filter, never another Finding or state.
CREATE TABLE project_finding_audits (
 audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
 finding_id text NOT NULL,
 project_id text NOT NULL,
 finding_created_at timestamptz NOT NULL,
 correlation_group_key text NOT NULL,
 PRIMARY KEY(audit_id,finding_id),
 FOREIGN KEY(project_id,finding_id) REFERENCES project_findings(project_id,finding_id) ON DELETE CASCADE
);
CREATE INDEX project_finding_audits_page_idx ON project_finding_audits(audit_id,finding_created_at,finding_id);
CREATE INDEX project_finding_audits_finding_idx ON project_finding_audits(project_id,finding_id,audit_id);

CREATE UNIQUE INDEX project_finding_audits_group_idx ON project_finding_audits(audit_id,correlation_group_key COLLATE "C") WHERE correlation_group_key<>'';
CREATE INDEX project_finding_audits_unqualified_idx ON project_finding_audits(audit_id,finding_id COLLATE "C") WHERE correlation_group_key='';

ALTER TABLE finding_proposal_receipts
    ADD COLUMN publication_requested boolean NOT NULL DEFAULT false,
    ADD COLUMN publication_error text NOT NULL DEFAULT '',
    ADD COLUMN project_proposal jsonb,
    ADD COLUMN project_evidence jsonb,
    ADD CONSTRAINT finding_project_material_shape CHECK (
        ((project_proposal IS NULL) = (project_evidence IS NULL))
        AND (project_proposal IS NULL OR (project_id IS NOT NULL AND jsonb_typeof(project_proposal) = 'object'
            AND octet_length(project_proposal::text) <= 4096 AND jsonb_typeof(project_evidence) = 'array'
            AND octet_length(project_evidence::text) <= 1048576))
    ),
    ADD CONSTRAINT finding_publication_error_shape CHECK (
        publication_error = '' OR (publication_requested AND publication_error IN ('problem_evidence_required', 'problem_project_required'))
    );
CREATE INDEX finding_receipts_project_history_idx ON finding_proposal_receipts(project_id, created_at, receipt_id COLLATE "C")
    WHERE project_proposal IS NOT NULL;

ALTER TABLE finding_proposal_receipts
    DROP CONSTRAINT finding_proposal_receipts_retention_state_check,
    DROP CONSTRAINT finding_proposal_receipts_retention_shape,
    ADD CONSTRAINT finding_proposal_receipts_retention_state_check CHECK (retention_state IN ('source-held','audit-held','project-held','discarded')),
    ADD CONSTRAINT finding_proposal_receipts_retention_shape CHECK (
        (retention_state='source-held' AND source_run_deleted_at IS NULL AND discarded_at IS NULL)
        OR (retention_state='audit-held' AND discarded_at IS NULL)
        OR (retention_state='project-held' AND project_proposal IS NOT NULL AND discarded_at IS NULL)
        OR (retention_state='discarded' AND source_run_deleted_at IS NOT NULL AND discarded_at IS NOT NULL)
    );

DROP TRIGGER finding_proposal_audit_holds_reconcile ON finding_proposal_audit_holds;
DROP FUNCTION contractor_reconcile_finding_retention();

-- Retention may acquire an independent exact Project binding once. It never
-- changes the original receipt, explicit intent or any historical rating.
CREATE OR REPLACE FUNCTION contractor_protect_finding_proposal_receipt_retention() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    materials_unchanged boolean;
    evidence_index integer;
BEGIN
    IF TG_OP='UPDATE' THEN
        materials_unchanged := ROW(NEW.project_proposal,NEW.project_evidence)
            IS NOT DISTINCT FROM ROW(OLD.project_proposal,OLD.project_evidence);
        IF NOT materials_unchanged AND OLD.project_proposal IS NULL AND NEW.project_proposal IS NOT NULL
           AND NEW.project_proposal->>'digest'=OLD.proposal_digest
           AND NEW.project_proposal->>'mediaType'=OLD.proposal_media_type
           AND (NEW.project_proposal->>'sizeBytes')::bigint=OLD.proposal_size_bytes
           AND jsonb_array_length(NEW.project_evidence)=jsonb_array_length(OLD.evidence) THEN
            materials_unchanged := true;
            FOR evidence_index IN 0..jsonb_array_length(OLD.evidence)-1 LOOP
                IF (NEW.project_evidence->evidence_index)-'ref' IS DISTINCT FROM (OLD.evidence->evidence_index)-'ref' THEN
                    materials_unchanged := false;
                END IF;
            END LOOP;
        END IF;
        IF NEW.retention_state='discarded' THEN
            NEW.report_projection:='{}';
            NEW.anchor_projection:='{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"discarded"}';
            NEW.fingerprint_projection:='{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable","reason":"discarded"}';
            NEW.checks_projection:=contractor_empty_finding_checks(NEW.proposal_digest,NEW.proposal_size_bytes);
        END IF;
        IF NEW.comparison_projection IS DISTINCT FROM OLD.comparison_projection
           AND NEW.comparison_projection IS DISTINCT FROM contractor_finding_comparison_projection(
               NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state) THEN
            RAISE EXCEPTION 'finding comparison projection does not match its immutable sources' USING ERRCODE='23514';
        END IF;
        NEW.comparison_projection:=contractor_finding_comparison_projection(
            NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state);
        IF materials_unchanged AND
           to_jsonb(NEW)-ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at',
            'report_projection','fingerprint_projection','anchor_projection','comparison_projection',
            'checks_projection','checks_projection_bytes','checks_projection_count','project_proposal','project_evidence']
           =to_jsonb(OLD)-ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at',
            'report_projection','fingerprint_projection','anchor_projection','comparison_projection',
            'checks_projection','checks_projection_bytes','checks_projection_count','project_proposal','project_evidence']
           AND (NEW.report_projection=OLD.report_projection OR NEW.retention_state='discarded')
           AND (NEW.anchor_projection=OLD.anchor_projection OR NEW.retention_state='discarded')
           AND (NEW.fingerprint_projection=OLD.fingerprint_projection OR NEW.retention_state='discarded')
           AND (NEW.checks_projection::text IS NOT DISTINCT FROM OLD.checks_projection::text
                OR OLD.checks_projection IS NULL OR NEW.retention_state='discarded') THEN RETURN NEW;
        END IF;
    END IF;
    IF TG_OP='DELETE' AND current_setting('contractor.lifecycle_purge',true)='project'
       AND EXISTS(SELECT 1 FROM projects WHERE project_id=OLD.project_id AND lifecycle_state='deleting') THEN RETURN OLD; END IF;
    RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE='23514';
END;
$$;

-- The existing fence now also exists for Projects that have never had an Audit.
INSERT INTO finding_correlation_fences(project_id)
    SELECT project_id FROM projects WHERE kind = 'project' ON CONFLICT DO NOTHING;
CREATE FUNCTION contractor_initialize_project_finding_fence() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.kind = 'project' THEN
        INSERT INTO finding_correlation_fences(project_id) VALUES (NEW.project_id);
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER projects_initialize_finding_fence AFTER INSERT ON projects
    FOR EACH ROW EXECUTE FUNCTION contractor_initialize_project_finding_fence();

CREATE FUNCTION contractor_protect_project_finding_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.finding_id, NEW.project_id, NEW.first_receipt_id, NEW.first_proposal_ref, NEW.title,
           NEW.correlation_algorithm, NEW.correlation_value, NEW.correlation_group_key, NEW.correlation_reason,
           NEW.created_at)
       IS DISTINCT FROM ROW(OLD.finding_id, OLD.project_id, OLD.first_receipt_id, OLD.first_proposal_ref, OLD.title,
           OLD.correlation_algorithm, OLD.correlation_value, OLD.correlation_group_key, OLD.correlation_reason,
           OLD.created_at) THEN
        RAISE EXCEPTION 'Project Finding identity is immutable' USING ERRCODE = '23514';
    END IF;
    IF NEW.revision <> OLD.revision + 1 THEN
        RAISE EXCEPTION 'Project Finding revision must advance once' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER project_findings_protect_identity BEFORE UPDATE ON project_findings
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_project_finding_identity();

-- No old AuditProfile parser remains at runtime. Every old Audit becomes
-- ordinary immutable Project history, with exact materials left in place.
CREATE FUNCTION pg_temp.audit_archive_fact(p_project text,p_audit text,p_kind text,p_key text,p_original jsonb)
RETURNS jsonb LANGUAGE plpgsql AS $$
DECLARE
 body bytea := convert_to(jsonb_build_object('schema','contractor.audit.history.v1',
  'auditId',p_audit,'kind',p_kind,'original',p_original)::text,'UTF8');
 digest bytea; version text; target_name text;
BEGIN
 digest := sha256(body); version := 'audit-history-'||encode(digest,'hex');
 target_name := 'a.'||rtrim(translate(encode(sha256(convert_to(p_audit,'UTF8')),'base64'),'+/','-_'),'=')||
 '.'||p_kind||'.'||rtrim(translate(encode(sha256(convert_to(p_key,'UTF8')),'base64'),'+/','-_'),'=');
 INSERT INTO artifact_blobs(sha256,payload,size_bytes) VALUES(digest,body,octet_length(body)) ON CONFLICT DO NOTHING;
 INSERT INTO artifact_versions(version_id,blob_sha256,media_type) VALUES(version,digest,'application/json') ON CONFLICT DO NOTHING;
 INSERT INTO artifact_scopes(scope_kind,scope_id) VALUES('project',p_project) ON CONFLICT DO NOTHING;
 INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes)
 VALUES('project',p_project,'historical-audits',target_name,version,true,1,octet_length(body));
 INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id)
 VALUES('project',p_project,'historical-audits',target_name,version,version);
 RETURN jsonb_build_object('ref',jsonb_build_object('namespace','historical-audits','name',target_name,'revision',version),
 'digest','sha256:'||encode(digest,'hex'),'mediaType','application/json','sizeBytes',octet_length(body));
END; $$;

DO $$
DECLARE audit record; fact record; relation text; snapshot jsonb;
BEGIN
 FOR audit IN SELECT * FROM audits ORDER BY project_id,audit_id LOOP
  snapshot:=pg_temp.audit_archive_fact(audit.project_id,audit.audit_id,'snapshot',audit.audit_id,to_jsonb(audit));
  -- Each record is a bounded ordinary Artifact, never a giant whole-history
  -- aggregate. No old entity is read or synchronized by working code.
  FOREACH relation IN ARRAY ARRAY['audit_rounds','audit_items','audit_executions','audit_execution_items',
   'audit_artifact_links','audit_events','audit_review_requests','audit_review_decisions',
   'audit_report_pages','audit_round_result_pages','audit_findings','audit_finding_contributions',
   'audit_finding_assessments','finding_proposal_audit_holds','mutation_receipts'] LOOP
   FOR fact IN EXECUTE format('SELECT to_jsonb(row) AS original FROM %I AS row WHERE audit_id=$1',relation) USING audit.audit_id LOOP
    PERFORM pg_temp.audit_archive_fact(audit.project_id,audit.audit_id,replace(relation,'audit_',''),fact.original::text,fact.original);
   END LOOP;
  END LOOP;
  FOR fact IN SELECT receipt.receipt_id,to_jsonb(receipt) AS original FROM finding_proposal_receipts AS receipt
   WHERE receipt.audit_id=audit.audit_id OR EXISTS(SELECT 1 FROM finding_proposal_audit_holds AS hold
    WHERE hold.receipt_id=receipt.receipt_id AND hold.audit_id=audit.audit_id) LOOP
   PERFORM pg_temp.audit_archive_fact(audit.project_id,audit.audit_id,'receipt',fact.receipt_id,fact.original);
  END LOOP;
  -- Exact references inside original frozen documents continue to resolve.
  -- These bindings are now ordinary Project-owned historical materials.
  UPDATE artifact_bindings SET frozen=true WHERE scope_kind='project' AND scope_id=audit.project_id
   AND namespace='audit-'||encode(sha256(convert_to('contractor.audit.identity.v1','UTF8')||decode('00','hex')||convert_to('audit','UTF8')||decode('00','hex')||convert_to(audit.audit_id,'UTF8')),'hex');
 END LOOP;
END; $$;

-- The old model has no readers, writers or operational rows after switching.
-- Retain generic review history protection for the remaining item/report journal.
ALTER TABLE audit_findings DROP CONSTRAINT audit_findings_current_assessment_fkey;
ALTER TABLE audit_findings DROP CONSTRAINT audit_findings_current_decision_fkey;
ALTER TABLE audit_review_requests DROP CONSTRAINT audit_review_requests_finding_fkey;
ALTER TABLE audit_review_decisions DROP CONSTRAINT audit_review_decisions_duplicate_fkey;
DROP TABLE audit_finding_contributions;
DROP TABLE audit_finding_assessments;
DROP TABLE audit_findings;
DROP FUNCTION contractor_finding_correlation_deleted();
DROP FUNCTION contractor_finding_correlation_inserted();
DROP FUNCTION contractor_finding_correlation_updated();
DROP FUNCTION contractor_finding_correlation_contributed();
DROP FUNCTION contractor_finding_correlation_contribution_deleted();
DROP FUNCTION contractor_initialize_finding_correlation();
DROP FUNCTION contractor_protect_finding_correlation();

-- Release ordinary hold counters through their existing delete triggers.
-- Defer only the sealed-item deletion guard for this one-time transformation;
-- no live mutation changes a frozen task or report.
ALTER TABLE audit_items DISABLE TRIGGER audit_items_protect_prepared;
DELETE FROM audit_review_decisions;
DELETE FROM audit_review_requests;
UPDATE audit_items SET last_execution_item_id=NULL WHERE last_execution_item_id IS NOT NULL;
DELETE FROM audit_execution_items;
DELETE FROM audit_items;
DELETE FROM audit_executions;
DELETE FROM audit_rounds;
DELETE FROM audits;
ALTER TABLE audit_items ENABLE TRIGGER audit_items_protect_prepared;

-- Old Findings need no execution compatibility or live source records. Release
-- their original Run holds and start intake from an empty operational model.
DO $$ DECLARE receipt record; BEGIN
 FOR receipt IN SELECT * FROM finding_proposal_receipts WHERE source_run_deleted_at IS NULL LOOP
  PERFORM contractor_artifact_hold_apply('[]',contractor_finding_receipt_artifact_holds(receipt.run_id,receipt.proposal_ref,receipt.evidence),true);
 END LOOP;
END; $$;
ALTER TABLE finding_proposal_receipts DISABLE TRIGGER finding_proposal_receipts_protect_immutable;
DELETE FROM finding_proposal_receipts;
ALTER TABLE finding_proposal_receipts ENABLE TRIGGER finding_proposal_receipts_protect_immutable;
ALTER TABLE finding_proposal_receipts DROP CONSTRAINT finding_proposal_receipts_retention_state_check;
ALTER TABLE finding_proposal_receipts DROP CONSTRAINT finding_proposal_receipts_retention_shape;
ALTER TABLE finding_proposal_receipts ADD CONSTRAINT finding_proposal_receipts_retention_state_check
 CHECK(retention_state IN ('source-held','project-held','discarded'));
ALTER TABLE finding_proposal_receipts ADD CONSTRAINT finding_proposal_receipts_retention_shape CHECK(
 (retention_state='source-held' AND project_proposal IS NULL AND source_run_deleted_at IS NULL AND discarded_at IS NULL)
 OR (retention_state='project-held' AND project_proposal IS NOT NULL AND discarded_at IS NULL)
 OR (retention_state='discarded' AND project_proposal IS NULL AND source_run_deleted_at IS NOT NULL AND discarded_at IS NOT NULL));


ALTER TABLE audit_review_requests DROP COLUMN finding_id CASCADE;
ALTER TABLE audit_review_requests DROP CONSTRAINT audit_review_requests_kind_check;
ALTER TABLE audit_review_requests ADD CONSTRAINT audit_review_requests_kind_check
 CHECK(kind IN ('plan-approval','active-check-approval','provide-evidence','requirement-applicability','report-acceptance'));
ALTER TABLE audit_review_requests DROP CONSTRAINT audit_review_requests_subject_kind_check;
ALTER TABLE audit_review_requests ADD CONSTRAINT audit_review_requests_subject_kind_check
 CHECK(subject_kind IN ('audit-item-action','audit-report'));
ALTER TABLE audit_review_decisions DROP COLUMN finding_id CASCADE;
ALTER TABLE audit_review_decisions DROP COLUMN verdict CASCADE;
ALTER TABLE audit_review_decisions DROP COLUMN severity CASCADE;
ALTER TABLE audit_review_decisions DROP COLUMN duplicate_target_id CASCADE;
ALTER TABLE audit_review_decisions DROP CONSTRAINT audit_review_decisions_action_check;
ALTER TABLE audit_review_decisions ADD CONSTRAINT audit_review_decisions_action_check
 CHECK(action IN ('approve','reject','not_applicable'));

DROP FUNCTION pg_temp.audit_archive_fact(text,text,text,text,jsonb);

-- This one generic immutability guard serves source and action history.
ALTER FUNCTION contractor_protect_audit_finding_history() RENAME TO contractor_protect_immutable_history;

-- These are ordinary accepted-output associations, including incidental
-- discoveries. They have no current problem state and disappear with execution.
ALTER TABLE audit_execution_items ADD COLUMN finding_sources jsonb NOT NULL DEFAULT '[]'::jsonb
 CHECK(jsonb_typeof(finding_sources)='array' AND jsonb_array_length(finding_sources)<=129 AND octet_length(finding_sources::text)<=1048576);
