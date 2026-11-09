-- Cost is compact operational state of existing participants and parents.
-- NULL means earlier work has no monetary observation. The new-parent default
-- is applied only to future inserts; existing business rows are not rewritten.
ALTER TABLE stage_allocations ADD COLUMN llm_cost jsonb
    CHECK (llm_cost IS NULL OR jsonb_typeof(llm_cost) = 'object');
ALTER TABLE stage_executions ADD COLUMN planner_llm_cost jsonb
    CHECK (planner_llm_cost IS NULL OR jsonb_typeof(planner_llm_cost) = 'object');
ALTER TABLE workflow_runs ADD COLUMN llm_cost jsonb
    CHECK (llm_cost IS NULL OR jsonb_typeof(llm_cost) = 'object');
ALTER TABLE audit_executions ADD COLUMN llm_cost jsonb
    CHECK (llm_cost IS NULL OR jsonb_typeof(llm_cost) = 'object');
ALTER TABLE audits ADD COLUMN llm_cost jsonb
    CHECK (llm_cost IS NULL OR jsonb_typeof(llm_cost) = 'object');
ALTER TABLE projects ADD COLUMN llm_cost jsonb
    CHECK (llm_cost IS NULL OR jsonb_typeof(llm_cost) = 'object');

-- An empty object is the compact known-empty total for a new parent. Keeping
-- it small preserves HOT space for ordinary Run lease/event updates.
ALTER TABLE workflow_runs ALTER COLUMN llm_cost SET DEFAULT '{}'::jsonb;
ALTER TABLE audits ALTER COLUMN llm_cost SET DEFAULT '{}'::jsonb;
ALTER TABLE projects ALTER COLUMN llm_cost SET DEFAULT '{}'::jsonb;
