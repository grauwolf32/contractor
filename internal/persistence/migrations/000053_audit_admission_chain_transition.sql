-- One canonical admission page chain replaces the previous selection/prepared
-- chains. Drain old typed admission with the previous Server, then stop it
-- before migrating. Terminal checkpoints and frozen artifacts stay evidence.
DO $$
BEGIN
    -- Stabilize only parent metadata through this migration transaction.
    LOCK TABLE audits, audit_rounds IN SHARE MODE;
    IF EXISTS (
        SELECT 1 FROM audits AS audit
        WHERE audit.state IN ('active','waiting_review','paused','finalizing','cancelling','deleting')
          AND EXISTS (
              SELECT 1 FROM audit_rounds AS round
              WHERE round.audit_id = audit.audit_id
                AND (round.admission_checkpoint IS NOT NULL
                  OR round.prepared_generation IS NOT NULL
                  OR round.prepared_page_count IS NOT NULL)
          )
    ) THEN
        RAISE EXCEPTION 'finish or cancel Audits with typed admission metadata using the previous Server, then stop it before migrating the admission page chain'
            USING ERRCODE = '55000';
    END IF;
END;
$$;
