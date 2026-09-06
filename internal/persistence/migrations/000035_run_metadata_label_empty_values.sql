ALTER TABLE workflow_run_metadata_labels
    DROP CONSTRAINT workflow_run_metadata_labels_label_value_check,
    ADD CONSTRAINT workflow_run_metadata_labels_label_value_check
        CHECK (octet_length(label_value) BETWEEN 0 AND 256);
