-- Default Project material pages exclude migration archives. Explicit namespace
-- selection keeps the existing Project listing index and all exact reads.
CREATE INDEX artifact_bindings_project_materials_idx
ON artifact_bindings (scope_id, namespace, name)
WHERE scope_kind = 'project'
  AND namespace NOT LIKE 'audit-%'
  AND namespace NOT LIKE 'historical-%';
