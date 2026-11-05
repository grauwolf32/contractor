-- Existing accepted Tasks retain continuation decisions and source boundaries.
-- The latest producer lookup must not traverse the containing Audit history.
CREATE INDEX audit_tasks_workflow_ordinal ON audit_tasks(audit_id,workflow_alias,ordinal DESC);
CREATE INDEX audit_tasks_continuation_cursor ON audit_tasks(audit_id,workflow_alias,ordinal DESC)
 WHERE source->>'kind'='continuation';
