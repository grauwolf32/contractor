-- Execution writers own their Run/Audit locks before marking an Eval dirty.
-- Every projection writer takes the experiment queue before its member rows;
-- collectors acquire those private locks only after source/evidence reads.
CREATE OR REPLACE FUNCTION contractor_eval_dirty(selected_experiment text, selected_member text) RETURNS void
    LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO eval_projection_queue(experiment_id) VALUES (selected_experiment)
    ON CONFLICT (experiment_id)
    DO UPDATE SET revision = eval_projection_queue.revision + 1;

    INSERT INTO eval_member_projections(experiment_id, member_id)
    VALUES (selected_experiment, selected_member)
    ON CONFLICT (experiment_id, member_id)
    DO UPDATE SET revision = eval_member_projections.revision + 1;
END;
$$;

CREATE OR REPLACE FUNCTION contractor_eval_members_inserted() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO eval_projection_queue(experiment_id)
    SELECT DISTINCT experiment_id FROM inserted_members ORDER BY experiment_id
    ON CONFLICT (experiment_id)
    DO UPDATE SET revision = eval_projection_queue.revision + 1;

    INSERT INTO eval_member_projections(experiment_id, member_id)
    SELECT experiment_id, member_id FROM inserted_members ORDER BY experiment_id, member_id;
    RETURN NULL;
END;
$$;
