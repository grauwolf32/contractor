-- Ordinary Audit Artifact ownership. No payload or version is duplicated.
-- Earlier pending migrations may have queued the deferred current-revision FK
-- on the referenced table. Flush that FK before altering lineage, then restore
-- its ordinary deferred insertion boundary before copying bindings/revisions.
SET CONSTRAINTS artifact_bindings_scope_kind_scope_id_namespace_name_curre_fkey IMMEDIATE;
ALTER TABLE artifact_scopes DROP CONSTRAINT artifact_scopes_scope_kind_check;
ALTER TABLE artifact_scopes ADD CONSTRAINT artifact_scopes_scope_kind_check
    CHECK(scope_kind IN ('user','project','run','audit'));
ALTER TABLE artifact_scopes ADD COLUMN audit_id text GENERATED ALWAYS AS
    (CASE WHEN scope_kind='audit' THEN scope_id END) STORED;
ALTER TABLE artifact_scopes ADD CONSTRAINT artifact_scopes_audit_id_fkey
    FOREIGN KEY(audit_id) REFERENCES audits(audit_id);
CREATE INDEX artifact_scopes_audit_id_idx ON artifact_scopes(audit_id) WHERE audit_id IS NOT NULL;
ALTER TABLE artifact_lineage DROP CONSTRAINT artifact_lineage_lineage_kind_check;
ALTER TABLE artifact_lineage ADD CONSTRAINT artifact_lineage_lineage_kind_check
    CHECK(lineage_kind IN ('input_fork','output_bind','project_output_publish','audit_import',
                          'context_projection','audit_output','audit_input','audit_publish'));
ALTER TABLE artifact_lineage DROP CONSTRAINT artifact_lineage_source_scope_kind_source_scope_id_source__fkey;

SET CONSTRAINTS artifact_bindings_scope_kind_scope_id_namespace_name_curre_fkey DEFERRED;

CREATE FUNCTION contractor_require_lineage_source() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    PERFORM 1 FROM artifact_binding_revisions
    WHERE scope_kind=NEW.source_scope_kind AND scope_id=NEW.source_scope_id
      AND namespace=NEW.source_namespace AND name=NEW.source_name AND revision=NEW.source_revision
    FOR KEY SHARE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Artifact source revision is missing'
            USING ERRCODE='23503', CONSTRAINT='artifact_exact_hold';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER artifact_lineage_require_source BEFORE INSERT ON artifact_lineage
    FOR EACH ROW EXECUTE FUNCTION contractor_require_lineage_source();


-- Persist scope on accepted references. Migrate only the declared role outputs;
-- existing unrelated package/report consumers keep their own exact bindings.
ALTER TABLE audit_artifact_links ADD COLUMN artifact_scope text NOT NULL DEFAULT 'project'
    CHECK(artifact_scope IN ('project','audit'));
CREATE INDEX audit_artifact_links_material_idx ON audit_artifact_links
    (audit_id, (artifact_ref->>'namespace'), (artifact_ref->>'name'), (artifact_ref->>'revision'), logical_key)
    WHERE artifact_scope='audit';
CREATE TEMP TABLE audit_material_migration ON COMMIT DROP AS
SELECT DISTINCT link.audit_id,audit.project_id,link.artifact_ref->>'namespace' AS namespace,
       link.artifact_ref->>'name' AS name
FROM audit_artifact_links AS link JOIN audits AS audit USING(audit_id)
WHERE link.source_provenance->>'schema' IN
 ('contractor.audit.preparation-output.v1','contractor.audit.role-output-provenance.v1');
INSERT INTO artifact_scopes(scope_kind,scope_id)
SELECT DISTINCT 'audit',audit_id FROM audit_material_migration ON CONFLICT DO NOTHING;
INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,created_at,updated_at,retained_revisions,retained_bytes)
SELECT 'audit',selected.audit_id,b.namespace,b.name,b.current_revision,b.frozen,b.created_at,b.updated_at,b.retained_revisions,b.retained_bytes
FROM audit_material_migration AS selected JOIN artifact_bindings AS b
 ON b.scope_kind='project' AND b.scope_id=selected.project_id AND b.namespace=selected.namespace AND b.name=selected.name;
INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id,created_at)
SELECT 'audit',selected.audit_id,r.namespace,r.name,r.revision,r.version_id,r.created_at
FROM audit_material_migration AS selected JOIN artifact_binding_revisions AS r
 ON r.scope_kind='project' AND r.scope_id=selected.project_id AND r.namespace=selected.namespace AND r.name=selected.name;
INSERT INTO artifact_lineage(target_scope_kind,target_scope_id,target_namespace,target_name,target_revision,
 source_scope_kind,source_scope_id,source_namespace,source_name,source_revision,lineage_kind,created_at)
SELECT 'audit',selected.audit_id,l.target_namespace,l.target_name,l.target_revision,
 l.source_scope_kind,l.source_scope_id,l.source_namespace,l.source_name,l.source_revision,'audit_output',l.created_at
FROM audit_material_migration AS selected JOIN artifact_lineage AS l
 ON l.target_scope_kind='project' AND l.target_scope_id=selected.project_id
 AND l.target_namespace=selected.namespace AND l.target_name=selected.name;
ALTER TABLE audit_artifact_links DISABLE TRIGGER USER;
UPDATE audit_artifact_links AS link SET artifact_scope='audit'
FROM audit_material_migration AS selected WHERE selected.audit_id=link.audit_id
 AND link.artifact_ref->>'namespace'=selected.namespace AND link.artifact_ref->>'name'=selected.name
 AND link.source_provenance->>'schema' IN ('contractor.audit.preparation-output.v1','contractor.audit.role-output-provenance.v1');
ALTER TABLE audit_artifact_links ENABLE TRIGGER USER;
ALTER TABLE audit_executions DISABLE TRIGGER USER;
UPDATE audit_executions AS execution SET collection_retained_refs=(
 SELECT COALESCE(jsonb_agg(link || jsonb_build_object('scope',
   CASE WHEN EXISTS(SELECT 1 FROM audit_material_migration AS selected
    WHERE selected.audit_id=execution.audit_id AND selected.namespace=link #>> '{artifact,ref,namespace}'
     AND selected.name=link #>> '{artifact,ref,name}') THEN 'audit' ELSE 'project' END)), '[]'::jsonb)
 FROM jsonb_array_elements(execution.collection_retained_refs) AS link
) WHERE collection_retained_refs IS NOT NULL;
ALTER TABLE audit_executions ENABLE TRIGGER USER;

CREATE FUNCTION contractor_guard_audit_artifact_admission() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    audit_state text;
    project text;
BEGIN
    SELECT state,project_id INTO audit_state,project FROM audits WHERE audit_id=NEW.scope_id FOR SHARE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Audit is missing' USING ERRCODE='23503';
    END IF;
    IF audit_state='deleting' THEN
        RAISE EXCEPTION 'Audit is deleting' USING ERRCODE='55000';
    END IF;
    -- Terminal collection during Project cancellation/drain retains its existing
    -- claim authority. New Run admission remains fenced by the owning service.
    PERFORM 1 FROM projects WHERE project_id=project FOR SHARE;
    RETURN NEW;
END;
$$;
CREATE TRIGGER artifact_scopes_audit_admission BEFORE INSERT ON artifact_scopes
    FOR EACH ROW WHEN (NEW.scope_kind='audit') EXECUTE FUNCTION contractor_guard_audit_artifact_admission();
CREATE TRIGGER artifact_bindings_audit_admission BEFORE INSERT OR UPDATE ON artifact_bindings
    FOR EACH ROW WHEN (NEW.scope_kind='audit') EXECUTE FUNCTION contractor_guard_audit_artifact_admission();



CREATE OR REPLACE FUNCTION contractor_validate_audit_preparation_execution() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    audit audits%ROWTYPE;
    binding jsonb;
    mapping record;
    expected jsonb;
    output jsonb;
    namespace text;
BEGIN
    IF NEW.role <> 'prepare' AND (TG_OP = 'INSERT' OR OLD.role <> 'prepare') THEN RETURN NEW; END IF;
    IF TG_OP = 'UPDATE' THEN
        IF ROW(NEW.execution_id, NEW.audit_id, NEW.round_id, NEW.role, NEW.workflow_role,
               NEW.role_attempt, NEW.manifest_ref, NEW.manifest_digest, NEW.submission_key,
               NEW.request_digest, NEW.preparation_snapshot)
           IS DISTINCT FROM ROW(OLD.execution_id, OLD.audit_id, OLD.round_id, OLD.role, OLD.workflow_role,
               OLD.role_attempt, OLD.manifest_ref, OLD.manifest_digest, OLD.submission_key,
               OLD.request_digest, OLD.preparation_snapshot)
        THEN RAISE EXCEPTION 'Audit preparation execution identity is immutable' USING ERRCODE = '23514'; END IF;
        IF OLD.collection_receipt_id IS NOT NULL AND NEW.preparation_outputs IS DISTINCT FROM OLD.preparation_outputs THEN
            RAISE EXCEPTION 'Audit preparation outputs are immutable' USING ERRCODE = '23514';
        END IF;
        -- Retained data must remain readable after its source Run is deleted.
        IF OLD.collection_receipt_id IS NOT NULL OR NEW.collection_receipt_id IS NULL THEN RETURN NEW; END IF;
    END IF;

    SELECT * INTO STRICT audit FROM audits WHERE audit_id = NEW.audit_id FOR UPDATE;
    binding := audit.profile_snapshot #> ARRAY['workflows', NEW.workflow_role];
    IF binding->>'kind' IS DISTINCT FROM 'prepare' THEN
        RAISE EXCEPTION 'Audit preparation role is invalid' USING ERRCODE = '23514';
    END IF;

    IF TG_OP = 'INSERT' THEN
        IF audit.phase <> 'preparing' OR audit.current_round_id IS NOT NULL OR audit.state <> 'active'
           OR audit.dispatch_state <> 'open' OR NEW.role_attempt > (binding->>'maxRunAttempts')::integer
           OR NEW.role_attempt <> COALESCE((SELECT max(role_attempt) + 1 FROM audit_executions
               WHERE audit_id = NEW.audit_id AND role = 'prepare' AND workflow_role = NEW.workflow_role), 1)
           OR EXISTS (SELECT 1 FROM audit_executions WHERE audit_id = NEW.audit_id AND role = 'prepare'
               AND workflow_role = NEW.workflow_role AND (state <> 'collected' OR collection_disposition = 'accepted-result'))
        THEN RAISE EXCEPTION 'Audit preparation attempt is not admissible' USING ERRCODE = '23514'; END IF;
        IF EXISTS (SELECT 1 FROM jsonb_object_keys(NEW.preparation_snapshot->'inputs') AS name
                   WHERE NOT (binding->'inputs') ? name)
           OR EXISTS (SELECT 1 FROM jsonb_object_keys(NEW.preparation_snapshot->'parameters') AS name
                   WHERE NOT (binding->'parameters') ? name)
        THEN RAISE EXCEPTION 'Audit preparation submission has undeclared mappings' USING ERRCODE = '23514'; END IF;
        FOR mapping IN SELECT * FROM jsonb_each(binding->'inputs') LOOP
            expected := NULL;
            IF mapping.value->>'source' = 'audit-input' THEN
                expected := audit.baseline_snapshot #> ARRAY['inputs', mapping.value->>'name'];
            ELSIF mapping.value->>'source' = 'prepare-output' THEN
                SELECT accepted->'retained' INTO expected
                  FROM audit_executions AS producer CROSS JOIN LATERAL jsonb_array_elements(producer.preparation_outputs) AS accepted
                 WHERE producer.audit_id = NEW.audit_id AND producer.workflow_role = mapping.value->>'role'
                   AND producer.role = 'prepare' AND producer.collection_disposition = 'accepted-result'
                   AND accepted->>'logicalName' = mapping.value->>'name';
                IF expected IS NULL THEN RAISE EXCEPTION 'Audit preparation dependency is not accepted' USING ERRCODE = '23514'; END IF;
            ELSE RAISE EXCEPTION 'Audit preparation input source is invalid' USING ERRCODE = '23514'; END IF;
            IF NEW.preparation_snapshot #> ARRAY['inputs', mapping.key] IS DISTINCT FROM expected
               OR (expected IS NULL AND binding #>> ARRAY['workflow', 'inputs', mapping.key, 'required'] = 'true')
            THEN RAISE EXCEPTION 'Audit preparation input is not pinned exactly' USING ERRCODE = '23514'; END IF;
        END LOOP;
        FOR mapping IN SELECT * FROM jsonb_each(binding->'parameters') LOOP
            IF mapping.value->>'source' = 'literal' THEN expected := to_jsonb(COALESCE(mapping.value->>'value', ''));
            ELSIF mapping.value->>'source' = 'scope-field' THEN expected := audit.baseline_snapshot #> ARRAY['scope', mapping.value->>'name'];
            ELSE RAISE EXCEPTION 'Audit preparation parameter source is invalid' USING ERRCODE = '23514'; END IF;
            IF NEW.preparation_snapshot #> ARRAY['parameters', mapping.key] IS DISTINCT FROM expected THEN
                RAISE EXCEPTION 'Audit preparation parameter is not pinned exactly' USING ERRCODE = '23514';
            END IF;
        END LOOP;
        RETURN NEW;
    END IF;

    IF NEW.collection_disposition <> 'accepted-result' THEN RETURN NEW; END IF;
    IF jsonb_array_length(NEW.preparation_outputs) <> (SELECT count(*) FROM jsonb_each(binding->'outputs'))
       OR jsonb_array_length(NEW.preparation_outputs) <> (SELECT count(DISTINCT value->>'logicalName') FROM jsonb_array_elements(NEW.preparation_outputs))
    THEN RAISE EXCEPTION 'Audit preparation must accept all mapped outputs atomically' USING ERRCODE = '23514'; END IF;
    namespace := 'audit-' || encode(pg_catalog.sha256(
        convert_to('contractor.audit.identity.v1', 'UTF8') || decode('00', 'hex') || convert_to('audit', 'UTF8') || decode('00', 'hex') || convert_to(NEW.audit_id, 'UTF8')
    ), 'hex');
    FOR output IN SELECT * FROM jsonb_array_elements(NEW.preparation_outputs) LOOP
        IF binding #>> ARRAY['outputs', output->>'logicalName'] IS DISTINCT FROM output->>'workflowOutput'
           OR NOT COALESCE((binding #> ARRAY['workflow', 'outputs', output->>'workflowOutput', 'mediaTypes']) ?| ARRAY[output #>> '{source,mediaType}', '*/*'], false)
           OR output #>> '{source,ref,namespace}' IS DISTINCT FROM 'outputs'
           OR output #>> '{source,ref,name}' IS DISTINCT FROM output->>'workflowOutput'
           OR output #>> '{retained,ref,namespace}' IS DISTINCT FROM namespace
           OR output #>> '{source,digest}' IS DISTINCT FROM output #>> '{retained,digest}'
           OR NOT EXISTS (
               SELECT 1 FROM artifact_binding_revisions AS target
               JOIN artifact_bindings AS frozen USING (scope_kind, scope_id, namespace, name)
               JOIN artifact_versions AS version ON version.version_id = target.version_id
               JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
               JOIN artifact_lineage AS lineage ON lineage.target_scope_kind = target.scope_kind
                 AND lineage.target_scope_id = target.scope_id AND lineage.target_namespace = target.namespace
                 AND lineage.target_name = target.name AND lineage.target_revision = target.revision
               JOIN artifact_binding_revisions AS source ON source.scope_kind = lineage.source_scope_kind
                 AND source.scope_id = lineage.source_scope_id AND source.namespace = lineage.source_namespace
                 AND source.name = lineage.source_name AND source.revision = lineage.source_revision
               JOIN artifact_bindings AS source_binding ON source_binding.scope_kind = source.scope_kind
                 AND source_binding.scope_id = source.scope_id AND source_binding.namespace = source.namespace AND source_binding.name = source.name
               WHERE target.scope_kind = 'audit' AND target.scope_id = audit.audit_id
                 AND target.namespace = output #>> '{retained,ref,namespace}' AND target.name = output #>> '{retained,ref,name}'
                 AND target.revision = output #>> '{retained,ref,revision}' AND frozen.frozen AND frozen.current_revision = target.revision
                 AND source.version_id = target.version_id AND source.scope_kind = 'run' AND source.scope_id = NEW.run_id
                 AND source.namespace = 'outputs' AND source.name = output->>'workflowOutput'
                 AND source.revision = output #>> '{source,ref,revision}' AND source_binding.frozen AND source_binding.current_revision = source.revision
                 AND lineage.lineage_kind = 'audit_output'
                 AND output #>> '{source,digest}' = 'sha256:' || encode(blob.sha256, 'hex')
                 AND version.media_type = output #>> '{source,mediaType}' AND version.media_type = output #>> '{retained,mediaType}'
                 AND blob.size_bytes = COALESCE((output #>> '{source,sizeBytes}')::bigint, 0)
                 AND blob.size_bytes = COALESCE((output #>> '{retained,sizeBytes}')::bigint, 0)
           ) OR NOT EXISTS (
               SELECT 1 FROM jsonb_array_elements(NEW.collection_retained_refs) AS link
                WHERE link->>'logicalKey' = 'prepare:' || NEW.execution_id || ':' || (output->>'logicalName')
                  AND link->'artifact' = output->'retained'
           )
        THEN RAISE EXCEPTION 'Audit preparation output has no exact frozen retention and lineage' USING ERRCODE = '23514'; END IF;
    END LOOP;
    IF NOT EXISTS (SELECT 1 FROM jsonb_array_elements(NEW.preparation_outputs) AS candidate
                    WHERE candidate #> '{source,ref}' = NEW.collection_source_output_ref
                      AND candidate #>> '{source,digest}' = NEW.collection_source_output_digest)
    THEN RAISE EXCEPTION 'Audit preparation receipt source is not a mapped output' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END;
$$;

-- Existing explicit dependency holders follow the migrated preparation owner.
CREATE OR REPLACE FUNCTION contractor_audit_dependency_holds(project text, manifest jsonb) RETURNS jsonb
LANGUAGE sql IMMUTABLE BEGIN ATOMIC
 SELECT contractor_artifact_key_set(ARRAY(
  SELECT contractor_artifact_ref_key(
   CASE WHEN dependency->>'kind'='prepare-output' THEN 'audit' ELSE 'project' END,
   CASE WHEN dependency->>'kind'='prepare-output' THEN manifest->>'audit_id' ELSE project END,
   dependency #> '{retained,ref}')
  FROM contractor_jsonb_elements(manifest->'dependencies') AS dependency
 ));
END;

CREATE OR REPLACE FUNCTION contractor_validate_audit_dependencies() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
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
   OR NOT (CASE dependency->>'kind'
       WHEN 'finding-receipt' THEN dependency #>> '{retained,ref,namespace}' = 'finding-materials'
       WHEN 'accepted-item-result' THEN dependency #>> '{retained,ref,namespace}' IN ('finding-materials',contractor_audit_dependency_namespace(NEW.audit_id))
       ELSE dependency #>> '{retained,ref,namespace}' = contractor_audit_dependency_namespace(NEW.audit_id) END)
   OR NOT contractor_audit_dependency_origin_valid(NEW.audit_id,dependency)
   OR NOT contractor_audit_dependency_closure_valid(NEW.audit_id,NEW.retained_dependencies::jsonb,dependency)
   OR NOT EXISTS (
    SELECT 1 FROM artifact_binding_revisions AS revision
    JOIN artifact_bindings AS binding USING (scope_kind,scope_id,namespace,name)
    JOIN artifact_versions AS version USING (version_id)
    JOIN artifact_blobs AS blob ON blob.sha256=version.blob_sha256
    WHERE revision.scope_kind=CASE WHEN dependency->>'kind'='prepare-output' THEN 'audit' ELSE 'project' END
     AND revision.scope_id=CASE WHEN dependency->>'kind'='prepare-output' THEN NEW.audit_id ELSE project END
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

-- Existing explicit item dependency holders change scope with their owner.
-- Transfer their exact counted holds before any later holder release uses the
-- new scope rule. Other holders, including accepted Run input lineage, retain
-- their original exact Project sources until their own lifecycles release them.
DO $$
DECLARE held record; previous jsonb;
BEGIN
 FOR held IN SELECT item.retained_dependencies::jsonb AS manifest,audit.project_id
  FROM audit_items AS item JOIN audits AS audit USING(audit_id)
  WHERE item.retained_dependencies IS NOT NULL AND EXISTS(
    SELECT 1 FROM contractor_jsonb_elements(item.retained_dependencies::jsonb->'dependencies') AS dependency
    WHERE dependency->>'kind'='prepare-output')
 LOOP
  SELECT contractor_artifact_key_set(ARRAY(
    SELECT contractor_artifact_ref_key('project',held.project_id,dependency #> '{retained,ref}')
    FROM contractor_jsonb_elements(held.manifest->'dependencies') AS dependency)) INTO previous;
  PERFORM contractor_artifact_hold_apply(contractor_audit_dependency_holds(held.project_id,held.manifest),previous,true);
 END LOOP;
END;
$$;
