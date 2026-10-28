-- Retain the first proposal's applicability conditions in standalone reports.
-- Previous projection readers omitted the field. Never invent empty conditions
-- or rewrite immutable receipts/reports: the operator must stop the old writer,
-- preserve the database, and choose a fresh database or a separately authorized
-- explicit data disposition before retrying a refused upgrade (PER-R43).
DO $$
BEGIN
    LOCK TABLE finding_proposal_receipts IN SHARE MODE;
    IF EXISTS (
        SELECT 1 FROM finding_proposal_receipts
        WHERE retention_state <> 'discarded'
          AND jsonb_typeof(report_projection::jsonb -> 'preconditions') IS DISTINCT FROM 'array'
    ) THEN
        RAISE EXCEPTION 'retained finding report projections require original preconditions arrays; preserve this database and immutable reports, stop the previous Server, then use a separate fresh database or obtain an explicitly authorized data disposition before upgrading'
            USING ERRCODE = '55000';
    END IF;
END;
$$;

ALTER TABLE finding_proposal_receipts
    ADD CONSTRAINT finding_proposal_receipts_report_preconditions_check CHECK (
        retention_state = 'discarded'
        OR jsonb_typeof(report_projection::jsonb -> 'preconditions') IS NOT DISTINCT FROM 'array'
    );
