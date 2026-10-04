-- planner_events duplicated every Planner fact already appended to
-- workflow_run_events, which is the Run replay authority, and no code read it.
-- Planner recovery reads planner_sessions.state. Dropping the table also drops
-- its immutability trigger; the function served no other table.
DROP TABLE planner_events;
DROP FUNCTION contractor_protect_planner_event_immutable();
