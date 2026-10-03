-- Deleting a member's child workspace invalidates evidence but cannot reopen
-- an already terminal experiment. Deleting the Eval's own Project still fences
-- every experiment so the Project controller can purge it.
CREATE OR REPLACE FUNCTION contractor_eval_fence_project() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.lifecycle_state = 'deleting' AND OLD.lifecycle_state = 'active' THEN
        UPDATE eval_experiments
           SET deletion_requested_at = CASE
                   WHEN project_id = NEW.project_id THEN COALESCE(deletion_requested_at, clock_timestamp())
                   ELSE deletion_requested_at
               END,
               state = 'cancelling',
               revision = revision + 1,
               updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
         WHERE project_id = NEW.project_id
            OR (state NOT IN ('finished', 'cancelled') AND experiment_id IN (
                SELECT experiment_id
                  FROM eval_project_dependencies
                 WHERE project_id = NEW.project_id
            ));
    END IF;
    RETURN NEW;
END; $$;
