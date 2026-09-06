-- Reverse FK probes used by synchronous lifecycle collection.
CREATE INDEX artifact_binding_revisions_version_idx
    ON artifact_binding_revisions (version_id);
CREATE INDEX artifact_versions_blob_idx
    ON artifact_versions (blob_sha256);

-- Exact binding history uses a stable descending keyset order.
CREATE INDEX artifact_binding_revisions_history_idx
    ON artifact_binding_revisions
       (scope_kind, scope_id, namespace, name, created_at DESC, revision DESC);

-- 000002 created this on precisely the same columns/order as the binding PK.
DROP INDEX artifact_bindings_list_idx;
