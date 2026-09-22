-- Reverse FK probes for artifact binding revisions. Run, Project and audit
-- namespace purges delete revisions one by one; without these both tables are
-- scanned for every deleted revision.
CREATE INDEX artifact_lineage_source_idx
    ON artifact_lineage
       (source_scope_kind, source_scope_id, source_namespace, source_name, source_revision);
CREATE INDEX artifact_pins_revision_idx
    ON artifact_pins (scope_kind, scope_id, namespace, name, revision);

-- Cascade/restrict probes for Audit and Project deletion.
CREATE INDEX audit_idempotency_audit_idx ON audit_idempotency (audit_id);
CREATE INDEX audit_collection_receipts_audit_idx ON audit_collection_receipts (audit_id);
CREATE INDEX eval_mutation_receipts_project_idx ON eval_mutation_receipts (project_id);
CREATE INDEX finding_proposal_audit_holds_project_idx ON finding_proposal_audit_holds (project_id);

-- 000059 repeated the unique partial index eval_submissions_execution_idx from 000056.
DROP INDEX eval_submissions_execution;
-- 000043 created this on precisely the same columns/order as the PK.
DROP INDEX audit_proposal_items_receipt_idx;
-- 000003 created this on precisely the same columns/order as the PK.
DROP INDEX stage_execution_reports_stage_idx;

-- Collection revisions fence list pagination. Only membership and the fields
-- that list filters, keys or summaries read can invalidate a cursor; usage
-- ticks, claims and progress writes must not.
CREATE OR REPLACE FUNCTION contractor_eval_collection_changed() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE leaving boolean := TG_OP = 'DELETE';
BEGIN
    IF TG_OP = 'UPDATE' THEN
        leaving := ROW(OLD.owner_id, OLD.project_id) IS DISTINCT FROM ROW(NEW.owner_id, NEW.project_id);
    END IF;
    -- Deleted and moved rows leave their old collections; moves also enter new ones.
    IF leaving THEN
        INSERT INTO eval_collections(owner_id, project_id) VALUES (OLD.owner_id, ''), (OLD.owner_id, OLD.project_id)
        ON CONFLICT (owner_id, project_id) DO UPDATE SET revision = eval_collections.revision + 1;
    END IF;
    IF TG_OP <> 'DELETE' THEN
        INSERT INTO eval_collections(owner_id, project_id) VALUES (NEW.owner_id, ''), (NEW.owner_id, NEW.project_id)
        ON CONFLICT (owner_id, project_id) DO UPDATE SET revision = eval_collections.revision + 1;
    END IF;
    RETURN NULL;
END; $$;
DROP TRIGGER eval_experiments_collection ON eval_experiments;
CREATE TRIGGER eval_experiments_collection AFTER INSERT OR DELETE ON eval_experiments
FOR EACH ROW EXECUTE FUNCTION contractor_eval_collection_changed();
CREATE TRIGGER eval_experiments_collection_update
AFTER UPDATE OF owner_id, project_id, name, control_mode, state, expected_count, draft, dataset_id
ON eval_experiments
FOR EACH ROW WHEN (
    ROW(OLD.owner_id, OLD.project_id, OLD.name, OLD.control_mode, OLD.state, OLD.expected_count, OLD.draft, OLD.dataset_id)
    IS DISTINCT FROM ROW(NEW.owner_id, NEW.project_id, NEW.name, NEW.control_mode, NEW.state, NEW.expected_count, NEW.draft, NEW.dataset_id)
) EXECUTE FUNCTION contractor_eval_collection_changed();
