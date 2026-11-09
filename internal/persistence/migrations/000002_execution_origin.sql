-- Execution provenance belongs to immutable Artifact versions. A fork keeps
-- the version, so source Run purge and byte-identical independent writes remain
-- distinguishable without traversing or retaining execution history.
ALTER TABLE artifact_versions ADD COLUMN execution_origin jsonb;
ALTER TABLE artifact_versions ADD CONSTRAINT artifact_versions_execution_origin_check
CHECK (execution_origin IS NULL OR (
 jsonb_typeof(execution_origin) = 'object'
 AND octet_length(execution_origin::text) <= 2048
 AND execution_origin ?& ARRAY['runId','stageExecutionId','allocationId','logicalAgentName','artifact']
 AND execution_origin - ARRAY['runId','stageExecutionId','allocationId','logicalAgentName','artifact'] = '{}'::jsonb
 AND jsonb_typeof(execution_origin->'artifact') = 'object'
 AND jsonb_typeof(execution_origin->'runId') = 'string'
 AND jsonb_typeof(execution_origin->'stageExecutionId') = 'string'
 AND jsonb_typeof(execution_origin->'allocationId') = 'string'
 AND jsonb_typeof(execution_origin->'logicalAgentName') = 'string'
 AND execution_origin->>'runId' ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
 AND execution_origin->>'stageExecutionId' ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
 AND execution_origin->>'allocationId' ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
 AND execution_origin->>'logicalAgentName' ~ '^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$'
) IS TRUE);

-- The existing version trigger permits the one-time Git provenance attachment.
-- That exception must never permit changing execution provenance alongside it.
CREATE FUNCTION contractor_protect_execution_origin() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.execution_origin IS DISTINCT FROM OLD.execution_origin THEN
  RAISE EXCEPTION 'artifact execution origin is immutable' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER artifact_versions_execution_origin_immutable
BEFORE UPDATE OF execution_origin ON artifact_versions
FOR EACH ROW EXECUTE FUNCTION contractor_protect_execution_origin();

-- Intake has two authenticated caller lanes. Operations carry real Stage and
-- Scheduler authority; allocation/Runtime/Worker columns apply only to Workers.
ALTER TABLE finding_proposal_receipts
 ALTER COLUMN allocation_id DROP NOT NULL,
 ALTER COLUMN runtime_agent_id DROP NOT NULL,
 ALTER COLUMN runtime_instance_id DROP NOT NULL,
 ALTER COLUMN logical_agent_name DROP NOT NULL,
 ADD COLUMN operation_ref text NOT NULL DEFAULT '',
 ADD COLUMN scan_origin jsonb;
ALTER TABLE finding_proposal_receipts ADD CONSTRAINT finding_receipt_execution_lane CHECK (
 (operation_ref='' AND scan_origin IS NULL AND allocation_id IS NOT NULL AND runtime_agent_id IS NOT NULL AND runtime_instance_id IS NOT NULL AND logical_agent_name IS NOT NULL)
 OR (operation_ref='publish-findings@1' AND scan_origin IS NOT NULL AND jsonb_typeof(scan_origin)='object' AND octet_length(scan_origin::text)<=2048 AND allocation_id IS NULL AND runtime_agent_id IS NULL AND runtime_instance_id IS NULL AND logical_agent_name IS NULL)
);
CREATE FUNCTION contractor_protect_finding_execution_lane() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
 IF ROW(NEW.operation_ref,NEW.scan_origin) IS DISTINCT FROM ROW(OLD.operation_ref,OLD.scan_origin) THEN
  RAISE EXCEPTION 'finding execution origin is immutable' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER finding_receipts_execution_lane_immutable
BEFORE UPDATE OF operation_ref,scan_origin ON finding_proposal_receipts
FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_execution_lane();


CREATE OR REPLACE FUNCTION contractor_finding_comparison_projection(report text, fingerprint text, anchors text, retention text) RETURNS jsonb
    LANGUAGE plpgsql IMMUTABLE STRICT
    AS $$
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
    AND f->>'adapter' IN ('pinned-sarif-native/v1','normalized-candidate/v1') THEN f->>'value' ELSE '' END,
   'sites',CASE WHEN a->>'availability'='available' THEN coalesce((SELECT jsonb_agg(k->>'siteKey') FROM jsonb_array_elements(a->'keys') AS k),'[]') ELSE '[]' END,
   'repository',coalesce(f->>'repository',a->>'repository',''),
   'commit',coalesce(f->>'commit',a->>'commit','')));
 RETURN result;
END;
$$;


CREATE OR REPLACE FUNCTION contractor_finding_primary_key(selected_receipt text) RETURNS TABLE(algorithm text, value text, reason text)
    LANGUAGE plpgsql
    AS $$
DECLARE
 fingerprint jsonb;
 anchors jsonb;
 issue_keys text[];
BEGIN
 SELECT fingerprint_projection::jsonb, anchor_projection::jsonb INTO fingerprint, anchors
 FROM finding_proposal_receipts WHERE receipt_id = selected_receipt;
 IF NOT FOUND THEN
  RAISE EXCEPTION 'finding first receipt is missing' USING ERRCODE='23503';
 END IF;
 IF fingerprint->>'availability' = 'available' THEN
  IF fingerprint->>'algorithm' IS DISTINCT FROM 'contractor/producer-issue/v1'
   OR fingerprint->>'adapter' NOT IN ('pinned-sarif-native/v1','normalized-candidate/v1') THEN
   RAISE EXCEPTION 'unsupported primary producer policy' USING ERRCODE='23514';
  END IF;
  RETURN QUERY SELECT 'contractor/producer-issue/v1'::text, fingerprint->>'value', ''::text;
  RETURN;
 END IF;
 IF anchors->>'availability' = 'available' THEN
  SELECT array_agg(key) INTO issue_keys FROM (
   SELECT DISTINCT item->>'issueKey' AS key FROM jsonb_array_elements(anchors->'keys') AS item
   WHERE item ? 'issueKey' LIMIT 2
  ) AS qualified;
 END IF;
 IF cardinality(issue_keys) = 1 THEN
  RETURN QUERY SELECT 'contractor/snapshot-issue-anchor/v1'::text, issue_keys[1], ''::text;
 ELSIF cardinality(issue_keys) > 1 THEN
  RETURN QUERY SELECT ''::text, ''::text, 'primary_key_ambiguous'::text;
 ELSIF fingerprint->>'reason' = 'not_recorded' AND anchors->>'reason' = 'not_recorded' THEN
  RETURN QUERY SELECT ''::text, ''::text, 'not_recorded'::text;
 ELSE
  RETURN QUERY SELECT ''::text, ''::text, 'no_qualified_key'::text;
 END IF;
END;
$$;

-- Canonical Task results are bounded domain references, including an explicit
-- empty collection. They do not create report/result Artifacts in Audit scope.
ALTER TABLE audit_tasks ADD COLUMN finding_result jsonb;
ALTER TABLE audit_execution_items ADD COLUMN finding_result jsonb;
ALTER TABLE audit_tasks ADD CONSTRAINT audit_task_findings_check CHECK (
 finding_result IS NULL OR (state='settled' AND result IS NULL
 AND jsonb_typeof(finding_result)='object' AND octet_length(finding_result::text)<=49152
 AND finding_result ?& ARRAY['ids','truncated'] AND finding_result-ARRAY['ids','truncated']='{}'::jsonb
 AND jsonb_typeof(finding_result->'ids')='array' AND jsonb_array_length(finding_result->'ids')<=128
 AND jsonb_typeof(finding_result->'truncated')='boolean') IS TRUE);
ALTER TABLE audit_execution_items ADD CONSTRAINT audit_item_findings_check CHECK (
 finding_result IS NULL OR (state='settled' AND result_ref IS NULL
 AND jsonb_typeof(finding_result)='object' AND octet_length(finding_result::text)<=49152
 AND finding_result ?& ARRAY['ids','truncated'] AND finding_result-ARRAY['ids','truncated']='{}'::jsonb
 AND jsonb_typeof(finding_result->'ids')='array' AND jsonb_array_length(finding_result->'ids')<=128
 AND jsonb_typeof(finding_result->'truncated')='boolean') IS TRUE);
CREATE TRIGGER audit_tasks_protect_settled_findings BEFORE UPDATE OF finding_result ON audit_tasks
FOR EACH ROW WHEN (OLD.state='settled') EXECUTE FUNCTION contractor_protect_immutable_history();
CREATE TRIGGER audit_items_protect_settled_findings BEFORE UPDATE OF finding_result ON audit_execution_items
FOR EACH ROW WHEN (OLD.state='settled') EXECUTE FUNCTION contractor_protect_immutable_history();

-- Accepted domain-only collection has no source output Artifact. Validation
-- requires a per-member canonical Finding result before invoking the mutation.
ALTER TABLE audit_executions DROP CONSTRAINT audit_executions_collection_source_shape;
ALTER TABLE audit_executions ADD CONSTRAINT audit_executions_collection_source_shape CHECK (
 (collection_source_output_ref IS NULL)=(collection_source_output_digest IS NULL)
 AND (collection_receipt_id IS NULL OR
  (collection_disposition='accepted-result'
   OR (collection_disposition='invalid-result')=(collection_source_output_ref IS NOT NULL))));

ALTER TABLE audit_tasks DROP COLUMN metadata_bytes;
ALTER TABLE audit_tasks ADD COLUMN metadata_bytes integer GENERATED ALWAYS AS (
 49152+octet_length(document::text)+octet_length(source::text)+COALESCE(octet_length(hypothesis_snapshot::text),0)) STORED;

CREATE INDEX audit_tasks_ready_workflow_idx ON audit_tasks USING btree
 (audit_id,workflow_alias,attention DESC,ordinal) INCLUDE (task_id,metadata_bytes) WHERE state='ready';

-- Exact normalized job outputs participate in the existing session Artifact pins.
CREATE OR REPLACE FUNCTION contractor_planner_session_artifact_refs(state jsonb) RETURNS jsonb
    LANGUAGE sql IMMUTABLE
    BEGIN ATOMIC
 SELECT contractor_artifact_key_set(ARRAY( SELECT contractor_artifact_ref_key('run'::text, ''::text, (((contractor_planner_session_artifact_refs.state -> 'completion'::text) -> 'result'::text) -> 'summaryArtifact'::text)) AS contractor_artifact_ref_key
         UNION ALL
          SELECT contractor_artifact_ref_key('run'::text, ''::text, slot.slot) AS contractor_artifact_ref_key
            FROM contractor_jsonb_values((((contractor_planner_session_artifact_refs.state -> 'completion'::text) -> 'result'::text) -> 'artifacts'::text)) slot(slot)
         UNION ALL
          SELECT contractor_artifact_ref_key('run'::text, ''::text, (((contractor_planner_session_artifact_refs.state -> 'scan'::text) -> 'state'::text) -> 'plan'::text)) AS contractor_artifact_ref_key
         UNION ALL
          SELECT contractor_artifact_ref_key('run'::text, ''::text, (job.job -> 'report'::text)) AS contractor_artifact_ref_key
            FROM contractor_jsonb_elements((((contractor_planner_session_artifact_refs.state -> 'scan'::text) -> 'state'::text) -> 'jobs'::text)) job(job)
         UNION ALL
          SELECT contractor_artifact_ref_key('run'::text, ''::text, input.input) AS contractor_artifact_ref_key
            FROM contractor_jsonb_elements((((contractor_planner_session_artifact_refs.state -> 'scan'::text) -> 'state'::text) -> 'jobs'::text)) job(job),
             LATERAL contractor_jsonb_values((job.job -> 'inputArtifacts'::text)) input(input)
         UNION ALL
          SELECT contractor_artifact_ref_key('run'::text, ''::text, selected.output)
            FROM contractor_jsonb_elements(state #> '{scan,state,jobs}') job(job),
             LATERAL contractor_jsonb_values(job.job->'outputs') selected(output))) AS contractor_artifact_key_set;
END;
