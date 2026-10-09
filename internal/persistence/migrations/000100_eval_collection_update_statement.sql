-- Bulk state changes (including the Project deletion fence) invalidate each
-- owner/Project list once. Usage-only and no-op writes do not invalidate them.
CREATE FUNCTION contractor_eval_collections_updated() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO eval_collections(owner_id, project_id)
    WITH changed AS (
        SELECT prior.owner_id AS old_owner, prior.project_id AS old_project,
               next_row.owner_id AS new_owner, next_row.project_id AS new_project
          FROM old_experiments prior JOIN new_experiments next_row USING (experiment_id)
         WHERE ROW(prior.owner_id, prior.project_id, prior.name, prior.control_mode,
                   prior.state, prior.expected_count, prior.draft, prior.dataset_id)
               IS DISTINCT FROM
               ROW(next_row.owner_id, next_row.project_id, next_row.name, next_row.control_mode,
                   next_row.state, next_row.expected_count, next_row.draft, next_row.dataset_id)
    )
    SELECT DISTINCT collection.owner_id, collection.project_id
      FROM changed
      CROSS JOIN LATERAL (VALUES
          (old_owner, ''), (old_owner, old_project),
          (new_owner, ''), (new_owner, new_project)
      ) AS collection(owner_id, project_id)
     ORDER BY 1, 2
    ON CONFLICT (owner_id, project_id) DO UPDATE SET revision = eval_collections.revision + 1;
    RETURN NULL;
END; $$;

DROP TRIGGER eval_experiments_collection_update ON eval_experiments;
CREATE TRIGGER eval_experiments_collection_update AFTER UPDATE ON eval_experiments
REFERENCING OLD TABLE AS old_experiments NEW TABLE AS new_experiments
FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_collections_updated();
