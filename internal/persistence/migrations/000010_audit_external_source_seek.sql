-- Source origins, rather than mapped output names, prevent continuation
-- decisions from observing their own retained output as new external work.
DROP INDEX audit_tasks_workflow_ordinal;
CREATE INDEX audit_tasks_workflow_external_ordinal ON audit_tasks (audit_id, workflow_alias, ordinal)
 WHERE state = 'settled' AND source->>'kind' <> 'continuation';
