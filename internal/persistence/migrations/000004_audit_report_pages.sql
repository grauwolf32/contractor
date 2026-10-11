-- Bounded, resumable Audit report finalization (V368-001).
--
-- Finalization used to rebuild the whole report in one attempt: every item,
-- every coverage page and one proposal artifact read per finding, all under
-- the Controller operation timeout, and a timeout restarted it from scratch.
-- Finding intake now stores the report projection of each proposal with its
-- receipt, and finalization records the report entries page by page, so a
-- later attempt resumes from the recorded pages.

-- The report fields of the proposal (auditdomain.FindingReportProjection), as
-- its canonical JSON text. A text column keeps octet_length free of
-- detoasting, which bounds report pages by bytes. A receipt is immutable, so
-- contractor_protect_finding_proposal_receipt_retention keeps this column
-- unchanged. Existing receipts cannot be projected in SQL when proposal
-- payloads live in the filesystem blob backend, so this migration requires a
-- database without receipts, like every database created from the 2026-10-11
-- baseline before this change (V367-001).
ALTER TABLE finding_proposal_receipts ADD COLUMN report_projection text NOT NULL;

ALTER TABLE finding_proposal_receipts
    ADD CONSTRAINT finding_proposal_receipts_report_projection_check CHECK (
        octet_length(report_projection) BETWEEN 2 AND 8388608
        AND jsonb_typeof(report_projection::jsonb) = 'object'
    );

-- One recorded page of a report build. A build belongs to one Audit revision;
-- pages of another revision are stale and never read. Sections follow a fixed
-- order: the items of each Round in Round order, then the findings of each
-- state. content holds the page's report entries as comma-separated compact
-- JSON values, empty when the page has none. A page that ends its section
-- (or Round) is exhausted; otherwise its last entry is the cursor of the next
-- page. Items pages also carry their coverage status counts.
CREATE TABLE audit_report_pages (
    audit_id text NOT NULL,
    audit_revision bigint NOT NULL,
    ordinal integer NOT NULL,
    section text NOT NULL,
    round_id text,
    last_item_ordinal integer,
    last_created_at timestamp with time zone,
    last_finding_id text,
    exhausted boolean NOT NULL,
    entry_count integer NOT NULL,
    content bytea NOT NULL,
    coverage_counts jsonb,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT audit_report_pages_audit_revision_check CHECK ((audit_revision > 0)),
    CONSTRAINT audit_report_pages_ordinal_check CHECK ((ordinal >= 0)),
    CONSTRAINT audit_report_pages_section_check CHECK ((section = ANY (ARRAY['items'::text, 'confirmed'::text, 'proposed'::text, 'rejected'::text, 'duplicate'::text, 'needs-evidence'::text]))),
    CONSTRAINT audit_report_pages_entries_check CHECK (((entry_count >= 0) AND ((entry_count = 0) = (octet_length(content) = 0)) AND ((entry_count > 0) OR exhausted))),
    CONSTRAINT audit_report_pages_content_check CHECK ((octet_length(content) <= 67108864)),
    CONSTRAINT audit_report_pages_shape CHECK ((CASE WHEN (section = 'items'::text) THEN ((round_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text) AND (jsonb_typeof(coverage_counts) = 'object'::text) AND (last_created_at IS NULL) AND (last_finding_id IS NULL) AND ((last_item_ordinal IS NULL) = (entry_count = 0)) AND ((last_item_ordinal IS NULL) OR (last_item_ordinal >= 0))) ELSE ((round_id IS NULL) AND (coverage_counts IS NULL) AND (last_item_ordinal IS NULL) AND ((last_created_at IS NULL) = (entry_count = 0)) AND ((last_finding_id IS NULL) = (entry_count = 0)) AND ((last_finding_id IS NULL) OR (last_finding_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text))) END))
);

ALTER TABLE audit_report_pages
    ADD CONSTRAINT audit_report_pages_pkey PRIMARY KEY (audit_id, audit_revision, ordinal);

ALTER TABLE audit_report_pages
    ADD CONSTRAINT audit_report_pages_audit_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

-- A report build only serves a finalizing Audit. Every exit from finalizing
-- (report commit, a frozen candidate, failure, cancellation or deletion)
-- discards its pages in the same transaction, so no path leaves them behind.
CREATE FUNCTION contractor_discard_audit_report_pages() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    DELETE FROM audit_report_pages WHERE audit_id = NEW.audit_id;
    RETURN NULL;
END;
$$;

CREATE TRIGGER audits_discard_report_pages AFTER UPDATE OF state ON audits FOR EACH ROW WHEN (((old.state = 'finalizing'::text) AND (new.state <> 'finalizing'::text))) EXECUTE FUNCTION contractor_discard_audit_report_pages();
