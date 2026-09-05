package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type collectionItemJSON struct {
	ExecutionItemID  string          `json:"execution_item_id"`
	Disposition      string          `json:"disposition"`
	ResultRef        json.RawMessage `json:"result_ref,omitempty"`
	ResultDigest     *string         `json:"result_digest,omitempty"`
	Retryable        bool            `json:"retryable"`
	FinalDisposition string          `json:"final_disposition"`
	Status           string          `json:"status"`
	Requested        []string        `json:"requested"`
	Completed        []string        `json:"completed"`
	Gaps             []string        `json:"gaps"`
	Rationale        string          `json:"rationale"`
}

type artifactLinkJSON struct {
	LogicalKey       string          `json:"logical_key"`
	ArtifactRef      json.RawMessage `json:"artifact_ref"`
	ArtifactDigest   string          `json:"artifact_digest"`
	MediaType        string          `json:"media_type"`
	SizeBytes        int64           `json:"size_bytes"`
	SourceProvenance json.RawMessage `json:"source_provenance"`
	DisplayRef       string          `json:"display_ref"`
}

func (s *PostgresStore) Collect(
	ctx context.Context,
	params CollectParams,
) (CollectionReceipt, bool, error) {
	if err := validateCollect(params); err != nil {
		return CollectionReceipt{}, false, err
	}
	if replay, found, err := s.lookupReceiptReplay(
		ctx, params.Claim.AuditID, params.ExecutionID, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	items := make([]collectionItemJSON, len(params.Items))
	for index, item := range params.Items {
		var resultRef json.RawMessage
		var resultDigest *string
		if item.Result != nil {
			resultRef, _ = json.Marshal(item.Result.Ref)
			value := item.Result.Digest
			resultDigest = &value
		}
		items[index] = collectionItemJSON{
			ExecutionItemID: item.ExecutionItemID, Disposition: string(item.Disposition),
			ResultRef: resultRef, ResultDigest: resultDigest, Retryable: item.Retryable,
			FinalDisposition: string(item.FinalDisposition), Status: string(item.Coverage.Status),
			Requested: nonNilStrings(item.Coverage.Requested), Completed: nonNilStrings(item.Coverage.Completed),
			Gaps: nonNilStrings(item.Coverage.Gaps), Rationale: item.Coverage.Rationale,
		}
	}
	encodedItems, _ := json.Marshal(items)
	links := make([]artifactLinkJSON, len(params.Retained))
	var retainedBytes int64
	retainedArtifacts := make(map[string]struct{}, len(params.Retained))
	for index, link := range params.Retained {
		ref, _ := json.Marshal(link.Artifact.Ref)
		links[index] = artifactLinkJSON{
			LogicalKey: link.LogicalKey, ArtifactRef: ref,
			ArtifactDigest: link.Artifact.Digest, MediaType: link.Artifact.MediaType,
			SizeBytes: link.Artifact.SizeBytes, SourceProvenance: link.SourceProvenance,
			DisplayRef: link.DisplayRef,
		}
		key := link.Artifact.Ref.Namespace + "\x00" + link.Artifact.Ref.Name + "\x00" + *link.Artifact.Ref.Revision
		if _, counted := retainedArtifacts[key]; !counted {
			retainedArtifacts[key] = struct{}{}
			retainedBytes += link.Artifact.SizeBytes
		}
	}
	encodedLinks, _ := json.Marshal(links)
	retainedSnapshot := append([]ArtifactLink{}, params.Retained...)
	encodedRetained, _ := json.Marshal(retainedSnapshot)
	var sourceRef json.RawMessage
	var sourceDigest *string
	if params.SourceOutput != nil {
		sourceRef, _ = json.Marshal(params.SourceOutput.Ref)
		value := params.SourceOutput.Digest
		sourceDigest = &value
	}
	receipt, err := scanReceipt(s.db.QueryRow(ctx, `
WITH collection_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($10::jsonb) AS item(
        execution_item_id text, disposition text, result_ref jsonb,
        result_digest text, retryable boolean, final_disposition text,
        status text, requested jsonb, completed jsonb, gaps jsonb, rationale text
    )
), live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), execution_gate AS MATERIALIZED (
    SELECT execution.*, audit.max_item_run_attempts
      FROM audit_executions AS execution
      JOIN audits AS audit USING (audit_id)
      JOIN live_claim USING (audit_id)
     WHERE execution.audit_id = $1 AND execution.execution_id = $4
       AND execution.state = 'collecting'
       AND (
           (execution.terminal_outcome = 'succeeded' AND $6 IN ('accepted-result', 'missing-output', 'invalid-result'))
           OR (execution.terminal_outcome IN ('failed', 'submission-failed') AND $6 = 'execution-failed')
           OR (execution.terminal_outcome = 'cancelled' AND $6 = 'execution-cancelled')
       )
     FOR UPDATE OF audit, execution
), member_validation AS MATERIALIZED (
    SELECT count(member.execution_item_id)::integer AS stored_count,
           count(input.execution_item_id)::integer AS matched_count
      FROM audit_execution_items AS member
      JOIN execution_gate AS execution USING (execution_id)
      LEFT JOIN collection_input AS input
        ON input.execution_item_id = member.execution_item_id
       AND input.disposition = $6
       AND member.state = 'collecting'
), advanced_audit AS (
    UPDATE audits AS audit
       SET retained_evidence_bytes = audit.retained_evidence_bytes + $12,
           revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM execution_gate, member_validation
     WHERE audit.audit_id = execution_gate.audit_id
       AND member_validation.stored_count = jsonb_array_length($10::jsonb)
       AND member_validation.matched_count = member_validation.stored_count
       AND audit.retained_evidence_bytes + $12 <= audit.max_evidence_bytes
    RETURNING audit.audit_id, audit.max_item_run_attempts, audit.next_event_sequence
), inserted_receipt AS (
    INSERT INTO audit_collection_receipts (
        receipt_id, audit_id, execution_id, run_id,
        terminal_outcome, terminal_run_generation, terminal_run_sequence,
        disposition, source_output_ref, source_output_digest, retained_refs,
        error_code, request_digest
    )
    SELECT $5, execution.audit_id, execution.execution_id, execution.run_id,
           execution.terminal_outcome, execution.terminal_run_generation,
           execution.terminal_run_sequence, $6, $7::jsonb, $8,
           $9::jsonb, $13, $14
      FROM execution_gate AS execution JOIN advanced_audit USING (audit_id)
    RETURNING *
), settled_attempts AS (
    UPDATE audit_execution_items AS member
       SET state = 'settled', collection_disposition = input.disposition,
           result_ref = input.result_ref, result_digest = input.result_digest,
           collected_at = clock_timestamp()
      FROM collection_input AS input, inserted_receipt AS receipt
     WHERE member.execution_id = receipt.execution_id
       AND member.execution_item_id = input.execution_item_id
       AND member.state = 'collecting'
    RETURNING member.execution_item_id, member.item_id, member.item_attempt
), settled_items AS (
    UPDATE audit_items AS item
       SET state = CASE
               WHEN input.retryable
                AND attempt.item_attempt < advanced.max_item_run_attempts
                AND input.disposition NOT IN ('accepted-result', 'execution-cancelled')
                 THEN 'ready'
               ELSE 'settled'
           END,
           final_disposition = CASE
               WHEN input.retryable
                AND attempt.item_attempt < advanced.max_item_run_attempts
                AND input.disposition NOT IN ('accepted-result', 'execution-cancelled')
                 THEN NULL
               ELSE input.final_disposition
           END,
           accepted_result_ref = CASE WHEN input.disposition = 'accepted-result' THEN input.result_ref ELSE NULL END,
           accepted_result_digest = CASE WHEN input.disposition = 'accepted-result' THEN input.result_digest ELSE NULL END,
           updated_at = GREATEST(clock_timestamp(), item.updated_at + interval '1 microsecond')
      FROM settled_attempts AS attempt
      JOIN collection_input AS input USING (execution_item_id)
      CROSS JOIN advanced_audit AS advanced
     WHERE item.item_id = attempt.item_id AND item.state = 'collecting'
), updated_coverage AS (
    UPDATE audit_coverage_rows AS coverage
       SET status = input.status, requested = input.requested,
           completed = input.completed, gaps = input.gaps,
           rationale = input.rationale,
           result_ref = input.result_ref, result_digest = input.result_digest,
           updated_at = clock_timestamp()
      FROM settled_attempts AS attempt
      JOIN collection_input AS input USING (execution_item_id)
     WHERE coverage.item_id = attempt.item_id
), link_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($11::jsonb) AS link(
        logical_key text, artifact_ref jsonb, artifact_digest text,
        media_type text, size_bytes bigint, source_provenance jsonb, display_ref text
    )
), inserted_links AS (
    INSERT INTO audit_artifact_links (
        audit_id, logical_key, artifact_ref, artifact_digest,
        media_type, size_bytes, source_provenance, display_ref
    )
    SELECT receipt.audit_id, link.logical_key, link.artifact_ref,
           link.artifact_digest, link.media_type, link.size_bytes,
           link.source_provenance, link.display_ref
      FROM inserted_receipt AS receipt CROSS JOIN link_input AS link
), collected_execution AS (
    UPDATE audit_executions AS execution
       SET state = 'collected', updated_at = clock_timestamp()
      FROM inserted_receipt AS receipt
     WHERE execution.execution_id = receipt.execution_id
       AND execution.state = 'collecting'
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, summary)
    SELECT receipt.audit_id, advanced.next_event_sequence - 1,
           'execution.collected', receipt.execution_id,
           jsonb_build_object('disposition', receipt.disposition)
      FROM inserted_receipt AS receipt JOIN advanced_audit AS advanced USING (audit_id)
)
SELECT `+prefixedReceiptColumns("inserted_receipt")+`
  FROM inserted_receipt`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExecutionID, params.ReceiptID, string(params.Disposition),
		sourceRef, sourceDigest, encodedRetained, encodedItems, encodedLinks,
		retainedBytes, params.ErrorCode, params.RequestDigest,
	))
	if err == nil {
		return receipt, true, nil
	}
	sqlState := persistencepostgres.SQLState(err)
	if sqlState == "23505" || errors.Is(err, pgx.ErrNoRows) {
		if existing, found, replayErr := s.lookupReceiptReplay(
			ctx, params.Claim.AuditID, params.ExecutionID, params.RequestDigest,
		); replayErr != nil || found {
			return existing, false, replayErr
		}
		if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
			return CollectionReceipt{}, false, liveErr
		} else if !live {
			return CollectionReceipt{}, false, ErrClaimLost
		}
		if sqlState == "23505" {
			return CollectionReceipt{}, false, ErrConflict
		}
		return CollectionReceipt{}, false, ErrPrecondition
	}
	return CollectionReceipt{}, false, fmt.Errorf("collect Audit execution: %w", err)
}

func prefixedReceiptColumns(prefix string) string {
	return prefix + ".receipt_id, " + prefix + ".audit_id, " + prefix + ".execution_id, " +
		prefix + ".run_id, " + prefix + ".terminal_outcome, " + prefix + ".terminal_run_generation, " +
		prefix + ".terminal_run_sequence, " + prefix + ".disposition, " + prefix + ".source_output_ref, " +
		prefix + ".source_output_digest, " + prefix + ".retained_refs, " + prefix + ".error_code, " +
		prefix + ".request_digest, " + prefix + ".created_at"
}

func (s *PostgresStore) lookupReceiptReplay(
	ctx context.Context, auditID, executionID, digest string,
) (CollectionReceipt, bool, error) {
	receipt, err := scanReceipt(s.db.QueryRow(ctx, `
SELECT `+receiptColumns+` FROM audit_collection_receipts
 WHERE execution_id = $1`, executionID))
	if errors.Is(err, pgx.ErrNoRows) {
		return CollectionReceipt{}, false, nil
	}
	if err != nil {
		return CollectionReceipt{}, false, fmt.Errorf("resolve Audit collection replay: %w", err)
	}
	if receipt.AuditID != auditID || receipt.RequestDigest != digest {
		return CollectionReceipt{}, true, ErrConflict
	}
	return receipt, true, nil
}
