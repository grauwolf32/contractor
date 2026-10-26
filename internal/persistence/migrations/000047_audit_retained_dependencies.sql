-- Exact cross-round dependencies are immutable accepted-item authority, not
-- arbitrary refs discovered by parsing artifact payloads. Each holder acquires
-- V410 exact revision obligations in the transaction publishing the item, and
-- releases them when existing bounded Audit purge deletes that item.
-- Distinct retained-revision evidence accounting is an indexed membership
-- query, independent of the containing Audit's accumulated links.
CREATE INDEX audit_artifact_links_exact_dependency_idx ON audit_artifact_links
 (audit_id,(artifact_ref->>'namespace'),(artifact_ref->>'name'),(artifact_ref->>'revision'),artifact_digest);

ALTER TABLE audit_items ADD COLUMN retained_dependencies json;
ALTER TABLE audit_items ADD CONSTRAINT audit_items_retained_dependencies_check CHECK (
 retained_dependencies IS NULL OR (
  json_typeof(retained_dependencies) = 'object'
  AND octet_length(retained_dependencies::text) <= 262144
  AND retained_dependencies->>'schema' = 'contractor.audit.retained-dependencies.v1'
  AND retained_dependencies->>'audit_id' = audit_id
  AND retained_dependencies->>'item_key' = item_key
  AND json_typeof(retained_dependencies->'dependencies') = 'array'
  AND json_array_length(retained_dependencies->'dependencies') BETWEEN 1 AND 32
 )
);

-- The same algorithm as auditdomain.ArtifactNamespace. Namespaces are derived
-- from the exact owning Audit, never inferred from a supplied source prefix.
CREATE FUNCTION contractor_audit_dependency_namespace(id text) RETURNS text
 LANGUAGE sql IMMUTABLE AS $$
SELECT 'audit-' || encode(sha256(convert_to('contractor.audit.identity.v1', 'UTF8') || decode('00','hex') ||
 convert_to('audit','UTF8') || decode('00','hex') || convert_to(id,'UTF8')), 'hex')
$$;

-- Prove each registered source from its indexed owning record. No receipt is
-- invented for baseline, preparation or accepted-item origins. Receipt refs
-- remain attribution only; the retained revision below carries the real hold.
CREATE FUNCTION contractor_audit_dependency_origin_valid(id text, dependency jsonb) RETURNS boolean
 LANGUAGE sql STABLE AS $$
SELECT CASE dependency->>'kind'
 WHEN 'baseline' THEN EXISTS (
  SELECT 1 FROM audits AS audit
  WHERE audit.audit_id=id AND audit.baseline_snapshot->'inputs'->(dependency #>> '{origin,baseline_input}') = dependency->'source'
   AND EXISTS (
    SELECT 1 FROM artifact_lineage AS lineage
    WHERE lineage.source_scope_kind='project' AND lineage.source_scope_id=audit.project_id
     AND lineage.source_namespace=dependency #>> '{source,ref,namespace}'
     AND lineage.source_name=dependency #>> '{source,ref,name}' AND lineage.source_revision=dependency #>> '{source,ref,revision}'
     AND lineage.target_scope_kind='project' AND lineage.target_scope_id=audit.project_id
     AND lineage.target_namespace=dependency #>> '{retained,ref,namespace}'
     AND lineage.target_name=dependency #>> '{retained,ref,name}' AND lineage.target_revision=dependency #>> '{retained,ref,revision}'
     AND lineage.lineage_kind='audit_import'
   )
 )
 WHEN 'prepare-output' THEN EXISTS (
  SELECT 1 FROM audit_executions AS execution,
   LATERAL jsonb_array_elements(execution.preparation_outputs) AS output
  WHERE execution.audit_id=id AND execution.execution_id=dependency #>> '{origin,execution_id}'
   AND execution.workflow_role=dependency #>> '{origin,workflow_role}' AND execution.role='prepare'
   AND execution.role_attempt=(dependency #>> '{origin,attempt}')::integer
   AND COALESCE(execution.run_id,execution.run_provenance->>'runId')=dependency #>> '{origin,run_id}'
   AND execution.collection_disposition='accepted-result'
   AND output->>'logicalName'=dependency #>> '{origin,logical_output}'
   AND output->'source'=dependency->'source' AND output->'retained'=dependency->'retained'
 )
 WHEN 'finding-receipt' THEN EXISTS (
  SELECT 1 FROM finding_proposal_audit_holds AS hold
  JOIN finding_proposal_receipts AS receipt USING (receipt_id)
  WHERE hold.audit_id=id AND hold.receipt_id=dependency #>> '{origin,receipt_id}'
   AND receipt.run_id=dependency #>> '{origin,run_id}'
   AND jsonb_array_length(hold.evidence)<=31 AND jsonb_array_length(receipt.evidence)<=31
   AND (dependency #>> '{origin,proposed_check_ordinal}')::integer >= 0
   AND (dependency #>> '{origin,proposed_check_ordinal}')::integer < 512
   AND ((receipt.proposal_ref=dependency #> '{source,ref}' AND receipt.proposal_digest=dependency #>> '{source,digest}' AND hold.proposal_ref=dependency->'retained')
    OR EXISTS (
     SELECT 1 FROM jsonb_array_elements(receipt.evidence) WITH ORDINALITY AS source(value,ordinal)
     JOIN jsonb_array_elements(hold.evidence) WITH ORDINALITY AS retained(value,ordinal) USING (ordinal)
     WHERE source.value=dependency->'source' AND retained.value=dependency->'retained'
    ))
 )
 WHEN 'accepted-item-result' THEN EXISTS (
  SELECT 1 FROM audit_items AS item
  JOIN audit_execution_items AS member ON member.item_id=item.item_id
  JOIN audit_executions AS execution ON execution.execution_id=member.execution_id
  CROSS JOIN LATERAL jsonb_array_elements(CASE
   WHEN jsonb_array_length(execution.collection_retained_refs)<=32 AND octet_length(execution.collection_retained_refs::text)<=1048576
   THEN execution.collection_retained_refs ELSE '[]'::jsonb END) AS link
  WHERE item.audit_id=id AND item.item_id=dependency #>> '{origin,item_id}'
   AND item.round_id=dependency #>> '{origin,round_id}'
   AND execution.execution_id=dependency #>> '{origin,execution_id}'
   AND member.item_attempt=(dependency #>> '{origin,attempt}')::integer
   AND COALESCE(execution.run_id,execution.run_provenance->>'runId')=dependency #>> '{origin,run_id}'
   AND member.collection_disposition='accepted-result' AND execution.collection_disposition='accepted-result'
   AND item.accepted_result_ref=member.result_ref AND item.accepted_result_digest=member.result_digest
   AND link->'artifact'=dependency->'retained'
   AND (link->>'logicalKey'='result/' || member.execution_item_id OR starts_with(link->>'logicalKey','evidence/' || execution.execution_id || '/'))
   AND ((execution.collection_source_output_ref=dependency #> '{source,ref}' AND execution.collection_source_output_digest=dependency #>> '{source,digest}')
    OR link #> '{sourceProvenance,evidence,artifact}'=dependency #> '{source,ref}')
 )
 ELSE false END
$$;

-- A declared graph is not a grant to omit the dependencies the trusted
-- producer contract requires. These bounded membership proofs use only exact
-- selected records, and never recursively follow refs embedded in payloads.
CREATE FUNCTION contractor_audit_dependency_closure_valid(id text, manifest jsonb, dependency jsonb) RETURNS boolean
 LANGUAGE sql STABLE AS $$
SELECT CASE dependency->>'kind'
 WHEN 'baseline' THEN true
 WHEN 'prepare-output' THEN NOT EXISTS (
  SELECT 1 FROM audits AS audit
  JOIN audit_executions AS execution ON execution.audit_id=audit.audit_id AND execution.execution_id=dependency #>> '{origin,execution_id}',
   LATERAL jsonb_each(audit.profile_snapshot->'workflows'->(dependency #>> '{origin,workflow_role}')->'inputs') AS input
  WHERE audit.audit_id=id AND input.value->>'source' IN ('audit-input','prepare-output')
   AND (execution.preparation_snapshot->'inputs') ? input.key
   AND NOT EXISTS (
    SELECT 1 FROM jsonb_array_elements(manifest->'dependencies') AS required
    WHERE dependency->'requires' ? (required->>'name')
     AND CASE input.value->>'source'
      WHEN 'audit-input' THEN required->>'kind'='baseline' AND required #>> '{origin,baseline_input}'=input.value->>'name'
      WHEN 'prepare-output' THEN required->>'kind'='prepare-output' AND required #>> '{origin,workflow_role}'=input.value->>'role' AND required #>> '{origin,logical_output}'=input.value->>'name'
     END
   )
 )
 WHEN 'finding-receipt' THEN NOT EXISTS (
  SELECT 1 FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_audit_holds AS hold USING (receipt_id),
   LATERAL jsonb_array_elements(CASE WHEN jsonb_array_length(hold.evidence)<=31 THEN hold.evidence ELSE '[]'::jsonb END) AS closure_evidence(value)
  WHERE hold.audit_id=id AND receipt.receipt_id=dependency #>> '{origin,receipt_id}'
   AND receipt.proposal_ref=dependency #> '{source,ref}'
   AND NOT EXISTS (
    SELECT 1 FROM jsonb_array_elements(manifest->'dependencies') AS required
    WHERE dependency->'requires' ? (required->>'name') AND required->'retained'=closure_evidence.value
     AND required->>'kind'='finding-receipt' AND required->'origin'=dependency->'origin'
   )
 )
 WHEN 'accepted-item-result' THEN NOT EXISTS (
  SELECT 1 FROM audit_executions AS execution,
   LATERAL jsonb_array_elements(CASE WHEN jsonb_array_length(execution.collection_retained_refs)<=32
    AND octet_length(execution.collection_retained_refs::text)<=1048576
    THEN execution.collection_retained_refs ELSE '[]'::jsonb END) AS closure_evidence(value)
  WHERE execution.audit_id=id AND execution.execution_id=dependency #>> '{origin,execution_id}'
   AND execution.collection_source_output_ref=dependency #> '{source,ref}'
   AND closure_evidence.value->'artifact'<>dependency->'retained'
   AND NOT EXISTS (
    SELECT 1 FROM jsonb_array_elements(manifest->'dependencies') AS required
    WHERE dependency->'requires' ? (required->>'name') AND required->'retained'=closure_evidence.value->'artifact'
     AND required->>'kind'='accepted-item-result' AND required->'origin'=dependency->'origin'
   )
 )
 ELSE false END
$$;

CREATE FUNCTION contractor_validate_audit_dependencies() RETURNS trigger
 LANGUAGE plpgsql AS $$
DECLARE dependency jsonb; project text; owner text;
BEGIN
 IF TG_OP='UPDATE' THEN
  IF NEW.retained_dependencies::text IS DISTINCT FROM OLD.retained_dependencies::text THEN
   RAISE EXCEPTION 'accepted Audit dependencies are immutable' USING ERRCODE='23514';
  END IF;
  RETURN NEW;
 END IF;
 IF NEW.retained_dependencies IS NULL THEN RETURN NEW; END IF;
 SELECT project_id,owner_id INTO project,owner FROM audits WHERE audit_id=NEW.audit_id;
 IF NEW.retained_dependencies->>'owner_id' IS DISTINCT FROM owner
  OR NEW.retained_dependencies->>'project_id' IS DISTINCT FROM project
  OR NEW.retained_dependencies->>'profile_digest' IS DISTINCT FROM (SELECT profile_digest FROM audits WHERE audit_id=NEW.audit_id)
  OR NEW.retained_dependencies->>'baseline_digest' IS DISTINCT FROM (SELECT 'sha256:' || encode(sha256(convert_to(baseline_snapshot::text,'UTF8')),'hex') FROM audits WHERE audit_id=NEW.audit_id) THEN
  RAISE EXCEPTION 'foreign Audit dependency manifest' USING ERRCODE='23514';
 END IF;
 FOR dependency IN SELECT value FROM jsonb_array_elements((NEW.retained_dependencies->'dependencies')::jsonb) LOOP
  IF ((dependency->'source') - 'ref') IS DISTINCT FROM ((dependency->'retained') - 'ref')
   OR (dependency->>'kind'='finding-receipt' AND (dependency #>> '{origin,receipt_id}' IS DISTINCT FROM NEW.proposal_receipt_id OR (dependency #>> '{origin,proposed_check_ordinal}')::integer IS DISTINCT FROM NEW.proposal_check_ordinal))
   OR dependency->>'projection' IS DISTINCT FROM 'exact'
   OR dependency #>> '{origin,kind}' IS DISTINCT FROM dependency->>'kind'
   OR dependency #>> '{retained,ref,namespace}' IS DISTINCT FROM contractor_audit_dependency_namespace(NEW.audit_id)
   OR NOT contractor_audit_dependency_origin_valid(NEW.audit_id,dependency)
   OR NOT contractor_audit_dependency_closure_valid(NEW.audit_id,NEW.retained_dependencies::jsonb,dependency)
   OR NOT EXISTS (
    SELECT 1 FROM artifact_binding_revisions AS revision
    JOIN artifact_bindings AS binding USING (scope_kind,scope_id,namespace,name)
    JOIN artifact_versions AS version USING (version_id)
    JOIN artifact_blobs AS blob ON blob.sha256=version.blob_sha256
    WHERE revision.scope_kind='project' AND revision.scope_id=project
     AND revision.namespace=dependency #>> '{retained,ref,namespace}' AND revision.name=dependency #>> '{retained,ref,name}'
     AND revision.revision=dependency #>> '{retained,ref,revision}' AND binding.frozen
     AND version.media_type=dependency #>> '{retained,mediaType}'
     AND blob.size_bytes=(dependency #>> '{retained,sizeBytes}')::bigint
     AND 'sha256:' || encode(blob.sha256,'hex')=dependency #>> '{retained,digest}'
   ) THEN RAISE EXCEPTION 'Audit dependency source is not accepted exact content' USING ERRCODE='23514'; END IF;
 END LOOP;
 RETURN NEW;
END;
$$;
CREATE TRIGGER audit_items_validate_dependencies BEFORE INSERT OR UPDATE OF retained_dependencies ON audit_items
 FOR EACH ROW EXECUTE FUNCTION contractor_validate_audit_dependencies();

CREATE FUNCTION contractor_audit_dependency_holds(project text, manifest jsonb) RETURNS jsonb
 LANGUAGE sql IMMUTABLE AS $$
SELECT contractor_artifact_key_set(ARRAY(
 SELECT contractor_artifact_ref_key('project',project,dependency #> '{retained,ref}')
 FROM contractor_jsonb_elements(manifest->'dependencies') AS dependency
))
$$;
-- Purge charges both the primary item and its distinct exact-revision
-- counter updates, without loading manifest JSON while selecting a batch.
ALTER TABLE audit_items ADD COLUMN retained_dependency_hold_count integer GENERATED ALWAYS AS
 (jsonb_array_length(contractor_audit_dependency_holds(retained_dependencies->>'project_id',retained_dependencies::jsonb))) STORED;

CREATE INDEX audit_items_dependency_purge_idx ON audit_items(audit_id,item_id) INCLUDE(retained_dependency_hold_count);

CREATE FUNCTION contractor_hold_audit_dependencies() RETURNS trigger
 LANGUAGE plpgsql AS $$
DECLARE project text;
BEGIN
 IF TG_OP='INSERT' THEN
  IF NEW.retained_dependencies IS NULL THEN RETURN NEW; END IF;
  project := NEW.retained_dependencies->>'project_id';
  PERFORM contractor_artifact_hold_apply(contractor_audit_dependency_holds(project,NEW.retained_dependencies::jsonb),'[]',true);
  RETURN NEW;
 END IF;
 IF OLD.retained_dependencies IS NULL THEN RETURN OLD; END IF;
 project := OLD.retained_dependencies->>'project_id';
 PERFORM contractor_artifact_hold_apply('[]',contractor_audit_dependency_holds(project,OLD.retained_dependencies::jsonb),true);
 RETURN OLD;
END;
$$;
CREATE TRIGGER audit_items_hold_dependencies AFTER INSERT OR DELETE ON audit_items
 FOR EACH ROW EXECUTE FUNCTION contractor_hold_audit_dependencies();
