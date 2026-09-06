-- Preserve the exact named Workflow binding selected from the immutable
-- AuditProfile snapshot. The coarse execution role remains useful for phase
-- transitions, but cannot distinguish multiple bindings with the same kind.

ALTER TABLE audit_executions
    ADD COLUMN workflow_role text;

UPDATE audit_executions AS execution
   SET workflow_role = COALESCE((
       SELECT min(item.workflow_role)
         FROM audit_execution_items AS member
         JOIN audit_items AS item ON item.item_id = member.item_id
        WHERE member.execution_id = execution.execution_id
   ), execution.role);

ALTER TABLE audit_executions
    ALTER COLUMN workflow_role SET NOT NULL,
    ADD CONSTRAINT audit_executions_workflow_role_shape CHECK (
        btrim(workflow_role) <> '' AND octet_length(workflow_role) <= 128
    );

DROP INDEX audit_executions_role_attempt_unique;

CREATE UNIQUE INDEX audit_executions_role_attempt_unique
    ON audit_executions (
        audit_id, COALESCE(round_id, ''), role, workflow_role,
        COALESCE(role_attempt, 0)
    ) WHERE role IN ('discovery', 'assessment');
