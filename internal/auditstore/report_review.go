package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"
)

// ProposeReport freezes exact report links and opens the existing review
// ledger instead of publishing them. It is replay-safe across Controller
// crashes and leaves the Audit without a runnable claim while a person reads
// the immutable candidate.
func (s *PostgresStore) ProposeReport(
	ctx context.Context, params ProposeReportParams,
) (Audit, bool, error) {
	if err := validateCommitReport(CommitReportParams(params)); err != nil {
		return Audit{}, false, err
	}
	machine, err := json.Marshal(params.Machine)
	if err != nil {
		return Audit{}, false, err
	}
	summary, err := json.Marshal(params.Summary)
	if err != nil {
		return Audit{}, false, err
	}
	reviewID := "review-report-" + strings.TrimPrefix(params.RequestDigest, "sha256:")
	audit, err := scanAudit(s.db.QueryRow(ctx, `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), gate AS MATERIALIZED (
    SELECT audit.*, round.revision AS round_revision
      FROM audits AS audit
      JOIN live_claim USING (audit_id)
      JOIN audit_rounds AS round
        ON round.audit_id = audit.audit_id AND round.round_id = $5
     WHERE audit.audit_id = $1 AND audit.revision = $4
       AND audit.state = 'finalizing' AND audit.dispatch_state = 'closed'
       AND audit.current_round_id = $5 AND audit.outstanding_run_count = 0
       AND round.revision = $6 AND round.state = 'closed'
       AND NOT EXISTS (
           SELECT 1 FROM audit_items AS item
            WHERE item.audit_id = audit.audit_id AND item.state <> 'settled'
       )
       AND NOT EXISTS (
           SELECT 1 FROM audit_executions AS execution
            WHERE execution.audit_id = audit.audit_id AND execution.state <> 'collected'
       )
       AND NOT EXISTS (
           SELECT 1 FROM audit_artifact_links AS existing
            WHERE existing.audit_id = audit.audit_id
              AND existing.logical_key IN ('report/machine', 'report/summary')
       )
     FOR UPDATE OF audit, round
), inserted_request AS (
    INSERT INTO audit_review_requests (
        request_id, audit_id, finding_id, subject_kind, subject_id, kind,
        subject_revision, subject_digest, requested_actions, state,
        expires_at, idempotency_key, request_digest
    )
    SELECT $7, audit_id, NULL, 'audit-report', audit_id,
           'report-acceptance', $4, $8, '["approve","reject"]'::jsonb,
           'pending', clock_timestamp() + interval '30 days',
           'auto-report:' || substr($8, 8, 64), $8
      FROM gate
    RETURNING request_id, audit_id
), inserted_candidate AS (
    INSERT INTO audit_report_candidates (
        audit_id, request_id, round_id, subject_revision, subject_digest,
        machine_link, summary_link
    )
    SELECT request.audit_id, request.request_id, $5, $4, $8,
           $9::jsonb, $10::jsonb
      FROM inserted_request AS request
    RETURNING audit_id
), changed AS (
    UPDATE audits AS audit
       SET state = 'waiting_review', revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM inserted_candidate AS candidate
     WHERE audit.audit_id = candidate.audit_id
    RETURNING audit.*
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT audit_id, next_event_sequence - 1, 'review.requested', $7,
           1, jsonb_build_object(
               'subjectKind', 'audit-report', 'subjectId', audit_id,
               'kind', 'report-acceptance'
           )
      FROM changed
)
SELECT `+prefixedAuditColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExpectedAuditRevision, params.RoundID, params.ExpectedRoundRevision,
		reviewID, params.RequestDigest, machine, summary,
	))
	if err == nil {
		return audit, true, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) && postgresState(err) != "23505" {
		return Audit{}, false, fmt.Errorf("propose Audit report: %w", err)
	}
	var storedRevision int64
	var storedDigest string
	if replayErr := s.db.QueryRow(ctx, `
SELECT candidate.subject_revision, candidate.subject_digest
  FROM audit_report_candidates AS candidate
 WHERE candidate.audit_id = $1`, params.Claim.AuditID).Scan(
		&storedRevision, &storedDigest,
	); replayErr == nil {
		if storedRevision != int64(params.ExpectedAuditRevision) || storedDigest != params.RequestDigest {
			return Audit{}, false, ErrConflict
		}
		existing, getErr := s.getAuditTrusted(ctx, params.Claim.AuditID)
		return existing, false, getErr
	} else if !errors.Is(replayErr, pgx.ErrNoRows) {
		return Audit{}, false, replayErr
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Audit{}, false, liveErr
	} else if !live {
		return Audit{}, false, ErrClaimLost
	}
	return Audit{}, false, ErrPrecondition
}

func postgresState(err error) string {
	type state interface{ SQLState() string }
	var value state
	if errors.As(err, &value) {
		return value.SQLState()
	}
	return ""
}

func (s *PostgresStore) GetReportCandidate(
	ctx context.Context, auditID string,
) (ReportCandidate, error) {
	if err := validateID("auditID", auditID); err != nil {
		return ReportCandidate{}, err
	}
	var result ReportCandidate
	var revision int64
	var machine, summary []byte
	err := s.db.QueryRow(ctx, `
SELECT request_id, round_id, subject_revision, subject_digest,
       machine_link, summary_link, created_at
  FROM audit_report_candidates
 WHERE audit_id = $1`, auditID).Scan(
		&result.RequestID, &result.RoundID, &revision, &result.SubjectDigest,
		&machine, &summary, &result.CreatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return ReportCandidate{}, ErrNotFound
	}
	if err != nil {
		return ReportCandidate{}, fmt.Errorf("read Audit report candidate: %w", err)
	}
	if revision < 1 || validateDigest("report candidate digest", result.SubjectDigest) != nil ||
		json.Unmarshal(machine, &result.Machine) != nil ||
		json.Unmarshal(summary, &result.Summary) != nil ||
		validateReportCandidateLinks(result.Machine, result.Summary) != nil {
		return ReportCandidate{}, errors.New("stored Audit report candidate is invalid")
	}
	result.SubjectRevision = uint64(revision)
	return result, nil
}

// ExpireReportReview deterministically closes an Audit whose immutable report
// candidate was not accepted within its bounded review window. The Controller
// calls it only under a live claim; the SQL rechecks database time so process
// clock skew cannot expire human authority early.
func (s *PostgresStore) ExpireReportReview(
	ctx context.Context, claim ControllerClaim, expectedAuditRevision uint64,
) (bool, error) {
	if err := validateClaimIdentity(claim); err != nil || expectedAuditRevision == 0 {
		if err != nil {
			return false, err
		}
		return false, ErrInvalid
	}
	var changed bool
	err := s.db.QueryRow(ctx, `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), gate AS MATERIALIZED (
    SELECT audit.audit_id, audit.next_event_sequence, review.request_id,
           review.state AS review_state
      FROM audits AS audit
      JOIN live_claim USING (audit_id)
      JOIN audit_report_candidates AS candidate USING (audit_id)
      JOIN audit_review_requests AS review
        ON review.request_id = candidate.request_id
       AND review.audit_id = candidate.audit_id
     WHERE audit.audit_id = $1 AND audit.revision = $4
       AND audit.state = 'waiting_review'
       AND (
           review.state = 'expired'
           OR (review.state = 'pending' AND review.expires_at <= clock_timestamp())
       )
     FOR UPDATE OF audit, review
), expired_request AS (
    UPDATE audit_review_requests AS review
       SET state = 'expired', revision = review.revision + 1,
           updated_at = GREATEST(clock_timestamp(), review.updated_at + interval '1 microsecond')
      FROM gate
     WHERE review.request_id = gate.request_id AND review.state = 'pending'
    RETURNING review.request_id
), terminal AS (
    UPDATE audits AS audit
       SET state = 'failed', dispatch_state = 'closed', hold_state = 'released',
           stop_reason_code = 'report_acceptance_expired',
           stop_reason_message = 'The exact Audit report was not accepted before its review expired.',
           revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond'),
           finished_at = clock_timestamp()
      FROM gate
     WHERE audit.audit_id = gate.audit_id
    RETURNING audit.audit_id, audit.next_event_sequence, gate.request_id
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, summary
    )
    SELECT audit_id, next_event_sequence - 1, 'review.expired', request_id,
           jsonb_build_object(
               'subjectKind', 'audit-report',
               'kind', 'report-acceptance',
               'terminalState', 'failed'
           )
      FROM terminal
)
SELECT EXISTS (SELECT 1 FROM terminal)`, claim.AuditID, claim.HolderID,
		claim.Epoch, int64(expectedAuditRevision)).Scan(&changed)
	if err != nil {
		return false, fmt.Errorf("expire Audit report review: %w", err)
	}
	if changed {
		return true, nil
	}
	live, err := s.claimLive(ctx, claim)
	if err != nil {
		return false, err
	}
	if !live {
		return false, ErrClaimLost
	}
	return false, nil
}

func validateReportCandidateLinks(machine, summary ArtifactLink) error {
	for _, pair := range []struct {
		link  ArtifactLink
		key   string
		media string
	}{
		{machine, ReportMachineLogicalKey, "application/json"},
		{summary, ReportSummaryLogicalKey, "text/markdown"},
	} {
		legacySummary := pair.key == ReportSummaryLogicalKey && pair.link.Artifact.MediaType == "text/plain"
		if pair.link.LogicalKey != pair.key || (pair.link.Artifact.MediaType != pair.media && !legacySummary) ||
			validateExactArtifact("report candidate", pair.link.Artifact, false) != nil ||
			len(pair.link.SourceProvenance) == 0 || !json.Valid(pair.link.SourceProvenance) {
			return ErrInvalid
		}
	}
	return nil
}
