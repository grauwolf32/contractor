ALTER TABLE artifact_lineage
    DROP CONSTRAINT artifact_lineage_lineage_kind_check,
    ADD CONSTRAINT artifact_lineage_lineage_kind_check
        CHECK (lineage_kind IN (
            'input_fork', 'output_bind', 'project_output_publish', 'audit_import'
        ));

ALTER TABLE audit_coverage_rows
    DROP CONSTRAINT audit_coverage_rows_status_check,
    ADD CONSTRAINT audit_coverage_rows_status_check
        CHECK (status IN (
            'not-tested', 'inconclusive', 'satisfied', 'violated',
            'not-applicable', 'blocked', 'excluded',
            'traced-complete', 'traced-partial', 'unmapped'
        ));
