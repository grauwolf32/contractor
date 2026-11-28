-- The existing immutable completion pin also admits one finding-bound ordinary Run.
ALTER TABLE workflow_runs DROP CONSTRAINT workflow_runs_check;
ALTER TABLE workflow_runs ADD CONSTRAINT workflow_runs_check CHECK (
 audit_completion IS NULL OR (
 jsonb_typeof(audit_completion) = 'object' AND
 audit_completion #>> '{contract,kind}' = 'audit-check-results@1' AND (
 (publication_mode = 'audit-managed' AND NOT audit_completion ? 'finding') OR
 (publication_mode = 'ordinary' AND project_id IS NOT NULL AND jsonb_typeof(audit_completion->'finding')='object')
 )));
