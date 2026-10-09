-- Preparation is Audit-scoped. Neither a placeholder Round nor an item owns
-- its identity. Existing executions and receipts retain their original bytes.
ALTER TABLE audits ADD COLUMN phase text NOT NULL DEFAULT 'not-started';
UPDATE audits SET phase = 'rounds' WHERE current_round_id IS NOT NULL;
ALTER TABLE audits ADD CONSTRAINT audits_phase_shape CHECK (
    phase IN ('not-started', 'preparing', 'inventory', 'rounds')
    AND (phase NOT IN ('preparing', 'inventory') OR (current_round_id IS NULL AND baseline_snapshot IS NOT NULL))
);

ALTER TABLE audit_executions
    ADD COLUMN preparation_snapshot jsonb,
    ADD COLUMN preparation_outputs jsonb NOT NULL DEFAULT '[]'::jsonb,
    DROP CONSTRAINT audit_executions_role_check,
    DROP CONSTRAINT audit_executions_role_shape;
ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_role_check CHECK (role IN ('prepare', 'discovery', 'check', 'assessment')),
    ADD CONSTRAINT audit_executions_role_shape CHECK (
        (role = 'check' AND round_id IS NOT NULL AND role_attempt IS NULL)
        OR (role IN ('discovery', 'assessment') AND role_attempt IS NOT NULL)
        OR (role = 'prepare' AND round_id IS NULL AND role_attempt BETWEEN 1 AND 10)
    ),
    ADD CONSTRAINT audit_executions_preparation_shape CHECK (
        (role = 'prepare' AND preparation_snapshot IS NOT NULL
            AND jsonb_typeof(preparation_snapshot) = 'object'
            AND jsonb_typeof(preparation_snapshot->'inputs') = 'object'
            AND jsonb_typeof(preparation_snapshot->'parameters') = 'object'
            AND preparation_snapshot ?& ARRAY['inputs', 'parameters']
            AND octet_length(preparation_snapshot::text) <= 1048576)
        OR (role <> 'prepare' AND preparation_snapshot IS NULL)
    ),
    ADD CONSTRAINT audit_executions_preparation_outputs_shape CHECK (
        jsonb_typeof(preparation_outputs) = 'array'
        AND jsonb_array_length(preparation_outputs) <= 128
        AND octet_length(preparation_outputs::text) <= 1048576
        AND CASE WHEN role = 'prepare' AND collection_disposition = 'accepted-result'
            THEN jsonb_array_length(preparation_outputs) > 0
            ELSE preparation_outputs = '[]'::jsonb END
    );

CREATE UNIQUE INDEX audit_executions_prepare_attempt_unique
    ON audit_executions (audit_id, workflow_role, role_attempt) WHERE role = 'prepare';
CREATE UNIQUE INDEX audit_executions_prepare_accepted_unique
    ON audit_executions (audit_id, workflow_role)
    WHERE role = 'prepare' AND collection_disposition = 'accepted-result';

CREATE FUNCTION contractor_protect_audit_preparation_baseline() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.phase IN ('preparing', 'inventory') OR EXISTS (
        SELECT 1 FROM audit_executions WHERE audit_id = OLD.audit_id AND role = 'prepare'
    ) THEN
        IF ROW(NEW.baseline_snapshot, NEW.profile_snapshot, NEW.input_selection,
               NEW.profile_name, NEW.profile_version, NEW.profile_digest, NEW.owner_id, NEW.project_id)
           IS DISTINCT FROM ROW(OLD.baseline_snapshot, OLD.profile_snapshot, OLD.input_selection,
               OLD.profile_name, OLD.profile_version, OLD.profile_digest, OLD.owner_id, OLD.project_id)
        THEN RAISE EXCEPTION 'Audit preparation baseline is immutable' USING ERRCODE = '23514'; END IF;
        IF NEW.phase = 'not-started' THEN
            RAISE EXCEPTION 'Audit preparation cannot return to not-started' USING ERRCODE = '23514';
        END IF;
    END IF;
    IF NEW.phase IN ('preparing', 'inventory') THEN
        IF NEW.baseline_snapshot->>'schema' IS DISTINCT FROM 'contractor.audit.baseline.v1'
           OR jsonb_typeof(NEW.baseline_snapshot->'inputs') IS DISTINCT FROM 'object'
           OR NEW.baseline_snapshot ? 'inventory'
           OR NOT EXISTS (SELECT 1 FROM jsonb_each(NEW.profile_snapshot->'workflows') AS role WHERE role.value->>'kind' = 'prepare')
        THEN RAISE EXCEPTION 'Audit preparation baseline or profile is invalid' USING ERRCODE = '23514'; END IF;
    END IF;
    IF NEW.phase = 'inventory' AND OLD.phase IS DISTINCT FROM 'inventory' AND EXISTS (
        SELECT 1 FROM jsonb_each(NEW.profile_snapshot->'workflows') AS role
         WHERE role.value->>'kind' = 'prepare' AND NOT EXISTS (
             SELECT 1 FROM audit_executions AS execution
              WHERE execution.audit_id = NEW.audit_id AND execution.role = 'prepare'
                AND execution.workflow_role = role.key AND execution.collection_disposition = 'accepted-result'
         )
    ) THEN RAISE EXCEPTION 'Audit preparation is incomplete' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER audits_protect_preparation_baseline BEFORE UPDATE ON audits
FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_preparation_baseline();

CREATE FUNCTION contractor_validate_audit_preparation_execution() RETURNS trigger
LANGUAGE plpgsql AS $$
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
               WHERE target.scope_kind = 'project' AND target.scope_id = audit.project_id
                 AND target.namespace = output #>> '{retained,ref,namespace}' AND target.name = output #>> '{retained,ref,name}'
                 AND target.revision = output #>> '{retained,ref,revision}' AND frozen.frozen AND frozen.current_revision = target.revision
                 AND source.version_id = target.version_id AND source.scope_kind = 'run' AND source.scope_id = NEW.run_id
                 AND source.namespace = 'outputs' AND source.name = output->>'workflowOutput'
                 AND source.revision = output #>> '{source,ref,revision}' AND source_binding.frozen AND source_binding.current_revision = source.revision
                 AND lineage.lineage_kind = 'audit_import'
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
CREATE TRIGGER audit_executions_validate_preparation
BEFORE INSERT OR UPDATE ON audit_executions FOR EACH ROW
EXECUTE FUNCTION contractor_validate_audit_preparation_execution();
