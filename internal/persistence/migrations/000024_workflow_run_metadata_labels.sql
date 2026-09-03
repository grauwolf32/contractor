CREATE TABLE workflow_run_metadata_labels (
    run_id text NOT NULL REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
    ordinal smallint NOT NULL CHECK (ordinal BETWEEN 1 AND 32),
    label_key text NOT NULL CHECK (
        octet_length(label_key) BETWEEN 1 AND 63
        AND label_key ~ '^[a-z][a-z0-9]*([._-][a-z0-9]+)*$'
        AND label_key !~ '^contractor\.'
    ),
    label_value text NOT NULL CHECK (octet_length(label_value) BETWEEN 1 AND 256),
    PRIMARY KEY (run_id, label_key),
    UNIQUE (run_id, ordinal)
);

CREATE INDEX workflow_run_metadata_labels_exact_idx
    ON workflow_run_metadata_labels (label_key, label_value, run_id);

CREATE OR REPLACE FUNCTION contractor_protect_workflow_run_metadata_label_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    -- An FK cascade enters this trigger below the parent DELETE trigger. Keep
    -- direct child mutations closed while allowing WorkflowRun retention to
    -- remove its complete owned label set.
    IF TG_OP = 'DELETE' AND pg_trigger_depth() > 1 THEN
        RETURN OLD;
    END IF;
    RAISE EXCEPTION 'WorkflowRun metadata labels are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER workflow_run_metadata_labels_protect_immutable
BEFORE UPDATE OR DELETE ON workflow_run_metadata_labels
FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_metadata_label_immutable();
