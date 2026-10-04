-- Report proposal creates the report-acceptance review request and the
-- frozen report candidate in one statement, one per Audit, and the candidate
-- repeated the request's subject revision and digest. The request now carries
-- the candidate's round and report links. They stay off audit_artifact_links:
-- Eval evidence and projections read every link there, and an unaccepted
-- report must not look like a published one.
ALTER TABLE audit_review_requests
    ADD COLUMN report_round_id text,
    ADD COLUMN report_machine_link jsonb CHECK (
        jsonb_typeof(report_machine_link) = 'object'
        AND octet_length(report_machine_link::text) <= 16384
    ),
    ADD COLUMN report_summary_link jsonb CHECK (
        jsonb_typeof(report_summary_link) = 'object'
        AND octet_length(report_summary_link::text) <= 16384
    );

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM audit_report_candidates AS candidate
          LEFT JOIN audit_review_requests AS request
            ON request.request_id = candidate.request_id AND request.audit_id = candidate.audit_id
         WHERE request.request_id IS NULL OR request.subject_kind <> 'audit-report'
            OR request.subject_revision <> candidate.subject_revision
            OR request.subject_digest <> candidate.subject_digest
    ) OR EXISTS (
        SELECT 1
          FROM audit_review_requests AS request
         WHERE request.subject_kind = 'audit-report'
           AND NOT EXISTS (
               SELECT 1 FROM audit_report_candidates AS candidate
                WHERE candidate.request_id = request.request_id
           )
    ) THEN
        RAISE EXCEPTION 'Audit report candidate disagrees with its review request';
    END IF;
END;
$$;

UPDATE audit_review_requests AS request
   SET report_round_id = candidate.round_id,
       report_machine_link = candidate.machine_link,
       report_summary_link = candidate.summary_link
  FROM audit_report_candidates AS candidate
 WHERE candidate.request_id = request.request_id AND candidate.audit_id = request.audit_id;

ALTER TABLE audit_review_requests
    ADD CONSTRAINT audit_review_requests_report_candidate_shape CHECK (
        (subject_kind = 'audit-report') = (report_round_id IS NOT NULL)
        AND (report_round_id IS NULL) = (report_machine_link IS NULL)
        AND (report_round_id IS NULL) = (report_summary_link IS NULL)
    ),
    ADD CONSTRAINT audit_review_requests_report_round_fkey
        FOREIGN KEY (report_round_id, audit_id)
        REFERENCES audit_rounds(round_id, audit_id) ON DELETE CASCADE;

-- One report candidate per Audit; a second proposal is a unique violation,
-- which the store resolves as a replay or a conflict.
CREATE UNIQUE INDEX audit_review_requests_report_candidate_unique
    ON audit_review_requests (audit_id) WHERE subject_kind = 'audit-report';

DROP TABLE audit_report_candidates;
