package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

const (
	ReportMachineLogicalKey = "report/machine"
	ReportSummaryLogicalKey = "report/summary"
)

func (s *PostgresStore) CommitReport(
	ctx context.Context, params CommitReportParams,
) (Audit, error) {
	if err := validateCommitReport(params); err != nil {
		return Audit{}, err
	}
	links := []ArtifactLink{params.Machine, params.Summary}
	encodedLinks := make([]artifactLinkJSON, len(links))
	for index, link := range links {
		ref, _ := json.Marshal(link.Artifact.Ref)
		encodedLinks[index] = artifactLinkJSON{
			LogicalKey: link.LogicalKey, ArtifactRef: ref,
			ArtifactDigest: link.Artifact.Digest, MediaType: link.Artifact.MediaType,
			SizeBytes: link.Artifact.SizeBytes, SourceProvenance: link.SourceProvenance,
			DisplayRef: link.DisplayRef,
		}
	}
	payload, _ := json.Marshal(encodedLinks)
	audit, err := scanAudit(s.db.QueryRow(ctx, `
WITH link_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($8::jsonb) AS link(
        logical_key text, artifact_ref jsonb, artifact_digest text,
        media_type text, size_bytes bigint, source_provenance jsonb, display_ref text
    )
), live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), finalization_gate AS MATERIALIZED (
    SELECT audit.audit_id
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
), inserted_links AS (
    INSERT INTO audit_artifact_links (
        audit_id, logical_key, artifact_ref, artifact_digest,
        media_type, size_bytes, source_provenance, display_ref
    )
    SELECT gate.audit_id, link.logical_key, link.artifact_ref,
           link.artifact_digest, link.media_type, link.size_bytes,
           link.source_provenance, link.display_ref
      FROM finalization_gate AS gate CROSS JOIN link_input AS link
    RETURNING audit_id
), changed AS (
    UPDATE audits AS audit
       SET state = 'completed', revision = audit.revision + 1,
           stop_reason_code = CASE WHEN audit.stop_reason_code = 'round_complete'
                                   THEN NULL ELSE audit.stop_reason_code END,
           stop_reason_message = CASE WHEN audit.stop_reason_code = 'round_complete'
                                      THEN NULL ELSE audit.stop_reason_message END,
           finished_at = clock_timestamp(),
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM finalization_gate
     WHERE audit.audit_id = finalization_gate.audit_id
       AND (SELECT count(*) FROM inserted_links) = 2
    RETURNING audit.*
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, entity_revision, summary)
    SELECT changed.audit_id, changed.next_event_sequence - 1,
           'audit.report_committed', changed.audit_id, changed.revision,
           jsonb_build_object('requestDigest', $7::text)
      FROM changed
)
SELECT `+prefixedAuditColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExpectedAuditRevision, params.RoundID, params.ExpectedRoundRevision,
		params.RequestDigest, payload,
	))
	if err == nil {
		return audit, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, fmt.Errorf("commit Audit report: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Audit{}, liveErr
	} else if !live {
		return Audit{}, ErrClaimLost
	}
	return Audit{}, ErrPrecondition
}
