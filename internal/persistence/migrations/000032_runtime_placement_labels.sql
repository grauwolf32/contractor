-- Runtime location selection is immutable Run data; each allocation retains
-- the selected incarnation's declaration independently of fleet liveness.
CREATE FUNCTION contractor_valid_placement_labels(labels text[]) RETURNS boolean
    LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE
    label text;
    previous text := '';
BEGIN
    IF labels IS NULL OR cardinality(labels) > 32
       OR (cardinality(labels) > 0 AND (array_ndims(labels) <> 1 OR array_lower(labels, 1) <> 1))
    THEN RETURN false; END IF;
    FOREACH label IN ARRAY labels LOOP
        IF label IS NULL OR octet_length(label) NOT BETWEEN 1 AND 63
           OR label !~ '^(cloud|local)(/[a-z][a-z0-9_-]*)*$'
           OR label ~ '[^a-z0-9_/-]'
        THEN RETURN false; END IF;
        IF previous <> '' AND left(label, length(previous) + 1) <> previous || '/'
        THEN RETURN false; END IF;
        previous := label;
    END LOOP;
    RETURN true;
END;
$$;

ALTER TABLE workflow_runs ADD COLUMN required_placement_labels text[] NOT NULL DEFAULT '{}'
    CONSTRAINT workflow_runs_placement_labels_valid CHECK (contractor_valid_placement_labels(required_placement_labels));
ALTER TABLE stage_allocations ADD COLUMN runtime_placement_labels text[] NOT NULL DEFAULT '{}'
    CONSTRAINT stage_allocations_placement_labels_valid CHECK (contractor_valid_placement_labels(runtime_placement_labels));

CREATE FUNCTION contractor_protect_placement_labels() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Runtime placement labels are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER workflow_runs_protect_placement_labels
    BEFORE UPDATE OF required_placement_labels ON workflow_runs
    FOR EACH ROW WHEN (NEW.required_placement_labels IS DISTINCT FROM OLD.required_placement_labels)
    EXECUTE FUNCTION contractor_protect_placement_labels();
CREATE TRIGGER stage_allocations_protect_placement_labels
    BEFORE UPDATE OF runtime_placement_labels ON stage_allocations
    FOR EACH ROW WHEN (NEW.runtime_placement_labels IS DISTINCT FROM OLD.runtime_placement_labels)
    EXECUTE FUNCTION contractor_protect_placement_labels();
