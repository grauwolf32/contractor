-- Delete finding proposal receipts with their Project (V368-009, DEL-R61).
--
-- A receipt is immutable: only its retention fields change, and the Run
-- deletion that discards it keeps it as tombstone provenance (DEL-R29).
-- Nothing removed it afterwards, so a purged Project left its receipts behind
-- for good, with owner, Project, client key, report projection and evidence
-- references. The Project purge now deletes them in bounded batches through
-- findingintake.PurgeProjectReceipts, after the Project's Runs and Audits,
-- and with the Audits every hold, contribution, finding and assessment that
-- referenced a receipt, are gone. Like the Artifact purge, it runs with
-- contractor.lifecycle_purge set to 'project'; a receipt may go only then and
-- only while its Project is deleting. A receipt of a Run outside any Project
-- names no Project and stays (DEL-R29).

CREATE OR REPLACE FUNCTION contractor_protect_finding_proposal_receipt_retention() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'UPDATE'
        AND to_jsonb(NEW) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at']
            = to_jsonb(OLD) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at']
    THEN
        RETURN NEW;
    END IF;
    IF TG_OP = 'DELETE'
        AND current_setting('contractor.lifecycle_purge', true) = 'project'
        AND EXISTS (
            SELECT 1 FROM projects
             WHERE project_id = OLD.project_id AND lifecycle_state = 'deleting'
        )
    THEN
        RETURN OLD;
    END IF;
    RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE = '23514';
END;
$$;

-- The purge reads a Project's receipts in receipt_id order, one batch at a
-- time.
CREATE INDEX finding_proposal_receipts_project_idx
    ON finding_proposal_receipts (project_id, receipt_id)
    WHERE project_id IS NOT NULL;

-- Deleting a receipt probes every foreign key that references it. These two
-- referencing columns led no index, so each probe would scan the whole table.
CREATE INDEX audit_finding_contributions_receipt_idx
    ON audit_finding_contributions (receipt_id);

CREATE INDEX audit_findings_first_receipt_idx
    ON audit_findings (first_receipt_id);
