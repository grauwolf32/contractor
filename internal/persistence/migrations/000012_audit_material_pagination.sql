-- A compact stable seek position keeps cursors bounded independently of the
-- accepted logical key (up to 512 bytes). This is traversal metadata only.
ALTER TABLE audit_artifact_links
    ADD COLUMN material_ordinal bigint GENERATED ALWAYS AS IDENTITY;
CREATE UNIQUE INDEX audit_artifact_links_ordinal_idx
    ON audit_artifact_links (audit_id, material_ordinal);
