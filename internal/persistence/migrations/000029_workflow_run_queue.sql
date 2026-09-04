-- Queue is a read projection over ordinary non-terminal WorkflowRuns. This
-- partial index follows its immutable oldest-first keyset without introducing
-- a second queue table or changing Scheduler claim order.
CREATE INDEX workflow_runs_owner_nonterminal_queue_idx
    ON workflow_runs (owner_id, created_at, run_id)
    INCLUDE (project_id, workflow_name, workflow_version, state, updated_at,
             run_event_generation, next_run_event_sequence)
    WHERE state IN ('initializing', 'running', 'cancelling');
