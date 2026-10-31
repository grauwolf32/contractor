-- The Project drain predicate needs only this Project's active Runs. The
-- full owner/Project history index and global active-Run indexes otherwise
-- form a bitmap intersection whose input grows with terminal Run history.
-- Preserve the predicate and results; give it a selective partial index.
CREATE INDEX workflow_runs_owner_project_active_idx
    ON workflow_runs (owner_id, project_id)
    WHERE project_id IS NOT NULL
      AND state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling');
