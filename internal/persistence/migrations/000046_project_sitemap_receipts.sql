-- Sitemap uses the shared immutable keyed receipt owner, not a second journal.
ALTER TABLE mutation_receipts DROP CONSTRAINT mutation_receipts_family_check;
ALTER TABLE mutation_receipts ADD CONSTRAINT mutation_receipts_family_check CHECK (
 CASE key_scope[1]
 WHEN 'audit' THEN owner_id IS NOT NULL AND audit_id IS NOT NULL AND project_id IS NULL
  AND key_scope = ARRAY['audit',owner_id,operation] AND idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
 WHEN 'eval' THEN owner_id IS NOT NULL AND project_id IS NOT NULL AND audit_id IS NULL
  AND key_scope = ARRAY['eval',owner_id,project_id,resource_id,operation]
 WHEN 'runtime-management' THEN owner_id IS NULL AND project_id IS NULL AND audit_id IS NULL
  AND key_scope = ARRAY['runtime-management'] AND idempotency_key ~ '^sha256:[0-9a-f]{64}$'
 WHEN 'scan-sweep' THEN owner_id IS NOT NULL AND project_id IS NULL AND audit_id IS NULL
  AND operation = 'scan-sweep.create' AND key_scope = ARRAY['scan-sweep',owner_id,operation]
  AND idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
 WHEN 'project-sitemap' THEN owner_id IS NOT NULL AND project_id IS NOT NULL AND audit_id IS NULL
  AND operation = 'sitemap.mutate' AND key_scope = ARRAY['project-sitemap',owner_id,project_id]
  AND resource_id = project_id AND idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
 ELSE false END
);

CREATE OR REPLACE FUNCTION contractor_protect_mutation_receipt() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP = 'DELETE' AND (
  OLD.key_scope[1] = 'audit'
  OR (OLD.key_scope[1] = 'eval' AND current_setting('contractor.eval_purge', true) = 'on')
  OR (OLD.key_scope[1] = 'project-sitemap' AND current_setting('contractor.lifecycle_purge', true) = 'project'
   AND EXISTS (SELECT 1 FROM projects WHERE project_id=OLD.project_id AND lifecycle_state='deleting'))
 ) THEN RETURN OLD; END IF;
 RAISE EXCEPTION 'mutation receipt is immutable' USING ERRCODE = '23514';
END; $$;
