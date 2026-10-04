-- Inserting or deleting experiments bumps each affected collection once per
-- statement. The row trigger upserted the same two collection rows twice per
-- experiment, and each upsert walked the row versions its own transaction could
-- not prune yet: inserting 20,000 experiments spent 9.4 of 10.7 s in it (1.0 s
-- without), deleting them 9.9 of 11.1 s. Cursors compare revisions only for
-- equality, so one bump per statement keeps their invalidation.
CREATE FUNCTION contractor_eval_collections_changed() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO eval_collections(owner_id, project_id)
    SELECT DISTINCT changed.owner_id, collection.project_id
    FROM changed_experiments changed
    CROSS JOIN LATERAL (VALUES (''), (changed.project_id)) AS collection(project_id)
    ORDER BY 1, 2
    ON CONFLICT (owner_id, project_id) DO UPDATE SET revision = eval_collections.revision + 1;
    RETURN NULL;
END; $$;

DROP TRIGGER eval_experiments_collection ON eval_experiments;
CREATE TRIGGER eval_experiments_collection_insert AFTER INSERT ON eval_experiments
REFERENCING NEW TABLE AS changed_experiments
FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_collections_changed();
CREATE TRIGGER eval_experiments_collection_delete AFTER DELETE ON eval_experiments
REFERENCING OLD TABLE AS changed_experiments
FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_collections_changed();
