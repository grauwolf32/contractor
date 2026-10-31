-- Stop the old Server before upgrading: pending sweep members are retired,
-- while admitted children remain ordinary Runs. No traffic or Audit creation.
-- These exact frozen bindings are private snapshots owned only by their sweep.
CREATE TEMP TABLE retired_sweep_inputs ON COMMIT DROP AS
SELECT b.scope_kind, b.scope_id, b.namespace, b.name
FROM artifact_bindings b
JOIN scan_sweeps s ON b.scope_id = s.owner_id
 AND b.name = 'scan-sweep.' || s.sweep_id || '.rulepack'
WHERE b.scope_kind = 'user' AND b.namespace = 'contractor-system' AND b.frozen;

CREATE TEMP TABLE retired_sweep_versions ON COMMIT DROP AS
SELECT DISTINCT r.version_id, v.blob_sha256
FROM artifact_binding_revisions r
JOIN retired_sweep_inputs b USING (scope_kind, scope_id, namespace, name)
JOIN artifact_versions v USING (version_id);

SELECT set_config('contractor.lifecycle_purge', 'project', true);
DELETE FROM artifact_lineage l USING retired_sweep_inputs b
WHERE (l.source_scope_kind, l.source_scope_id, l.source_namespace, l.source_name)
    = (b.scope_kind, b.scope_id, b.namespace, b.name)
 OR (l.target_scope_kind, l.target_scope_id, l.target_namespace, l.target_name)
    = (b.scope_kind, b.scope_id, b.namespace, b.name);
DELETE FROM artifact_bindings a USING retired_sweep_inputs b
WHERE (a.scope_kind, a.scope_id, a.namespace, a.name)
    = (b.scope_kind, b.scope_id, b.namespace, b.name);
DELETE FROM artifact_binding_revisions r USING retired_sweep_inputs b
WHERE (r.scope_kind, r.scope_id, r.namespace, r.name)
    = (b.scope_kind, b.scope_id, b.namespace, b.name);
-- Input forks and owner sources retain their own version references.
DELETE FROM artifact_versions v USING retired_sweep_versions r
WHERE v.version_id = r.version_id
 AND NOT EXISTS (SELECT 1 FROM artifact_binding_revisions b WHERE b.version_id = v.version_id);
DELETE FROM artifact_blobs b USING retired_sweep_versions r
WHERE b.sha256 = r.blob_sha256
 AND NOT EXISTS (SELECT 1 FROM artifact_versions v WHERE v.blob_sha256 = b.sha256);
-- Unreferenced filesystem objects follow the existing offline blob cleanup;
-- SQL migrations neither access host paths nor remove shared files.
SELECT set_config('contractor.lifecycle_purge', '', true);

DROP TABLE scan_sweep_members;
DROP TABLE scan_sweeps;
DROP FUNCTION contractor_scan_sweep_immutable();
DROP INDEX projects_sweep_selection_idx;

LOCK TABLE mutation_receipts IN ACCESS EXCLUSIVE MODE;
ALTER TABLE mutation_receipts DISABLE TRIGGER mutation_receipts_protect_immutable;
DELETE FROM mutation_receipts WHERE key_scope[1] = 'scan-sweep';
ALTER TABLE mutation_receipts DROP CONSTRAINT mutation_receipts_family_check;
ALTER TABLE mutation_receipts ADD CONSTRAINT mutation_receipts_family_check CHECK (
 CASE key_scope[1]
 WHEN 'audit' THEN owner_id IS NOT NULL AND audit_id IS NOT NULL AND project_id IS NULL
  AND key_scope = ARRAY['audit',owner_id,operation] AND idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
 WHEN 'eval' THEN owner_id IS NOT NULL AND project_id IS NOT NULL AND audit_id IS NULL
  AND key_scope = ARRAY['eval',owner_id,project_id,resource_id,operation]
 WHEN 'runtime-management' THEN owner_id IS NULL AND project_id IS NULL AND audit_id IS NULL
  AND key_scope = ARRAY['runtime-management'] AND idempotency_key ~ '^sha256:[0-9a-f]{64}$'
 WHEN 'project' THEN owner_id IS NOT NULL AND project_id IS NOT NULL AND audit_id IS NULL
  AND key_scope = ARRAY['project',owner_id,project_id,resource_id,operation]
  AND idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
 ELSE false END
);

ALTER TABLE mutation_receipts ENABLE TRIGGER mutation_receipts_protect_immutable;

CREATE OR REPLACE FUNCTION contractor_lifecycle_purge_enabled() RETURNS boolean
LANGUAGE sql STABLE
BEGIN ATOMIC
 SELECT COALESCE(current_setting('contractor.lifecycle_purge', true), '') IN ('run', 'project', 'audit');
END;
