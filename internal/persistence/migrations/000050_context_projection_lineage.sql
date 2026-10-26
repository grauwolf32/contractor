-- A derived frozen context has its own bytes, retaining its exact source with
-- the existing lineage holder and ordinary Run purge lifecycle.
ALTER TABLE artifact_lineage DROP CONSTRAINT artifact_lineage_lineage_kind_check;
ALTER TABLE artifact_lineage ADD CONSTRAINT artifact_lineage_lineage_kind_check CHECK (
 lineage_kind IN ('input_fork','output_bind','project_output_publish','audit_import','context_projection')
);
