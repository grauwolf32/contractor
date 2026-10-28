-- Execution manifests now transport exact task identity instead of duplicated
-- plan/dependency bodies. Finish or cancel old worklists with the old Server,
-- then stop it before applying this migration. Historical frozen bytes stay
-- evidence; neither a rewrite nor a legacy dispatch reader is introduced.
DO $$
BEGIN
    -- Keep this metadata preflight stable through the migration transaction.
    -- These locks do not authorize an old Server to resume after it commits.
    LOCK TABLE audits, audit_rounds IN SHARE MODE;
    IF EXISTS (
        SELECT 1 FROM audits AS audit
        WHERE audit.state IN ('active','waiting_review','paused','finalizing','cancelling','deleting')
          AND EXISTS (SELECT 1 FROM audit_rounds AS round WHERE round.audit_id = audit.audit_id)
    ) THEN
        RAISE EXCEPTION 'finish or cancel Audits with existing Rounds using the old Server, then stop it before migrating the execution manifest format'
            USING ERRCODE = '55000';
    END IF;
END;
$$;
