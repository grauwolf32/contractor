package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/jackc/pgx/v5"
)

const roundColumns = `
round_id, audit_id, ordinal, manifest_ref, manifest_digest,
state, expected_item_count, revision, created_at, updated_at`

const itemColumns = `
item_id, audit_id, round_id, item_key, ordinal, kind, subject_key,
task_ref, task_digest, workflow_role, state, final_disposition,
accepted_result_ref, accepted_result_digest, last_execution_item_id,
created_at, updated_at`

const executionColumns = `
execution_id, audit_id, round_id, role, role_attempt,
manifest_ref, manifest_digest, submission_key, request_digest, run_id,
state, terminal_outcome, terminal_run_generation, terminal_run_sequence,
terminal_observed_at, created_at, updated_at`

const receiptColumns = `
receipt_id, audit_id, execution_id, run_id,
terminal_outcome, terminal_run_generation, terminal_run_sequence,
disposition, source_output_ref, source_output_digest, retained_refs,
error_code, request_digest, created_at`

func (s *PostgresStore) GetRound(ctx context.Context, auditID, roundID string) (Round, error) {
	if err := validateID("auditID", auditID); err != nil {
		return Round{}, err
	}
	if err := validateID("roundID", roundID); err != nil {
		return Round{}, err
	}
	result, err := scanRound(s.db.QueryRow(ctx, `
SELECT `+roundColumns+` FROM audit_rounds
 WHERE audit_id = $1 AND round_id = $2`, auditID, roundID))
	if errors.Is(err, pgx.ErrNoRows) {
		return Round{}, ErrNotFound
	}
	if err != nil {
		return Round{}, fmt.Errorf("read Audit round: %w", err)
	}
	return result, nil
}

func (s *PostgresStore) ListItems(ctx context.Context, auditID string) ([]Item, error) {
	return s.listItems(ctx, auditID, 0)
}

func (s *PostgresStore) listItems(ctx context.Context, auditID string, limit int) ([]Item, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT `+prefixedItemColumns("item")+`
  FROM audit_items AS item
  JOIN audit_rounds AS round USING (round_id, audit_id)
 WHERE item.audit_id = $1
   AND ($2::integer = 0 OR item.state <> 'settled')
 ORDER BY round.ordinal, item.ordinal, item.item_id
 LIMIT CASE WHEN $2::integer = 0 THEN 100001 ELSE $2 END`, auditID, limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit items: %w", err)
	}
	defer rows.Close()
	result := make([]Item, 0)
	for rows.Next() {
		item, scanErr := scanItem(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit item: %w", scanErr)
		}
		result = append(result, item)
	}
	return result, rows.Err()
}

func prefixedItemColumns(prefix string) string {
	return prefix + ".item_id, " + prefix + ".audit_id, " + prefix + ".round_id, " +
		prefix + ".item_key, " + prefix + ".ordinal, " + prefix + ".kind, " + prefix + ".subject_key, " +
		prefix + ".task_ref, " + prefix + ".task_digest, " + prefix + ".workflow_role, " + prefix + ".state, " +
		prefix + ".final_disposition, " + prefix + ".accepted_result_ref, " + prefix + ".accepted_result_digest, " +
		prefix + ".last_execution_item_id, " + prefix + ".created_at, " + prefix + ".updated_at"
}

func (s *PostgresStore) ListExecutions(ctx context.Context, auditID string) ([]Execution, error) {
	return s.listExecutions(ctx, auditID, 0)
}

func (s *PostgresStore) listExecutions(ctx context.Context, auditID string, limit int) ([]Execution, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT `+executionColumns+` FROM audit_executions
 WHERE audit_id = $1
   AND ($2::integer = 0 OR state IN ('intent', 'submitted', 'collecting'))
 ORDER BY created_at, execution_id
 LIMIT CASE WHEN $2::integer = 0 THEN 1000001 ELSE $2 END`, auditID, limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit executions: %w", err)
	}
	defer rows.Close()
	result := make([]Execution, 0)
	for rows.Next() {
		execution, scanErr := scanExecution(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit execution: %w", scanErr)
		}
		result = append(result, execution)
	}
	return result, rows.Err()
}

func (s *PostgresStore) ListExecutionItems(ctx context.Context, executionID string) ([]ExecutionItem, error) {
	if err := validateID("executionID", executionID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT execution_item_id, execution_id, audit_id, round_id, item_id,
       batch_ordinal, item_attempt, task_ref, task_digest, input_refs,
       state, collection_disposition, result_ref, result_digest,
       created_at, collected_at
  FROM audit_execution_items
 WHERE execution_id = $1 ORDER BY batch_ordinal`, executionID)
	if err != nil {
		return nil, fmt.Errorf("list Audit execution items: %w", err)
	}
	defer rows.Close()
	result := make([]ExecutionItem, 0)
	for rows.Next() {
		item, scanErr := scanExecutionItem(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit execution item: %w", scanErr)
		}
		result = append(result, item)
	}
	return result, rows.Err()
}

// ListEvents returns a forward, bounded cursor page. Sequence zero starts at
// the first immutable event.
func (s *PostgresStore) ListEvents(
	ctx context.Context, auditID string, after uint64, limit int,
) ([]Event, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if after > math.MaxInt64 || limit < 1 || limit > MaxPageSize {
		return nil, invalidf("event page limit is invalid")
	}
	rows, err := s.db.Query(ctx, `
SELECT audit_id, sequence_number, kind, entity_id, entity_revision, summary, created_at
  FROM audit_events
 WHERE audit_id = $1 AND sequence_number > $2
 ORDER BY sequence_number LIMIT $3`, auditID, int64(after), limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit events: %w", err)
	}
	defer rows.Close()
	result := make([]Event, 0, limit)
	for rows.Next() {
		var event Event
		var sequence int64
		var revision *int64
		if err := rows.Scan(
			&event.AuditID, &sequence, &event.Kind, &event.EntityID,
			&revision, &event.Summary, &event.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan Audit event: %w", err)
		}
		if sequence <= 0 || !validEventKind(event.Kind) {
			return nil, errors.New("stored Audit event is invalid")
		}
		event.Sequence = uint64(sequence)
		if revision != nil {
			if *revision <= 0 {
				return nil, errors.New("stored Audit event revision is invalid")
			}
			value := uint64(*revision)
			event.EntityRevision = &value
		}
		result = append(result, event)
	}
	return result, rows.Err()
}

func (s *PostgresStore) ListCoverage(
	ctx context.Context, auditID, roundID string, afterOrdinal int, limit int,
) ([]CoverageRow, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if err := validateID("roundID", roundID); err != nil {
		return nil, err
	}
	if afterOrdinal < -1 || limit < 1 || limit > MaxPageSize {
		return nil, invalidf("coverage page is invalid")
	}
	rows, err := s.db.Query(ctx, `
SELECT coverage.audit_id, coverage.round_id, coverage.item_id,
       coverage.item_key, coverage.subject_key, coverage.status,
       coverage.requested, coverage.completed, coverage.gaps,
       coverage.rationale, coverage.result_ref, coverage.result_digest,
       coverage.updated_at
  FROM audit_coverage_rows AS coverage
  JOIN audit_items AS item USING (item_id, audit_id, round_id)
 WHERE coverage.audit_id = $1 AND coverage.round_id = $2
   AND item.ordinal > $3
 ORDER BY item.ordinal, item.item_id LIMIT $4`, auditID, roundID, afterOrdinal, limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit coverage: %w", err)
	}
	defer rows.Close()
	result := make([]CoverageRow, 0, limit)
	for rows.Next() {
		var row CoverageRow
		var status string
		var requested, completed, gaps, resultRef []byte
		var resultDigest *string
		if err := rows.Scan(
			&row.AuditID, &row.RoundID, &row.ItemID, &row.ItemKey,
			&row.SubjectKey, &status, &requested, &completed, &gaps,
			&row.Coverage.Rationale, &resultRef, &resultDigest, &row.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan Audit coverage: %w", err)
		}
		row.Coverage.Status = CoverageStatus(status)
		if !row.Coverage.Status.Valid() ||
			json.Unmarshal(requested, &row.Coverage.Requested) != nil ||
			json.Unmarshal(completed, &row.Coverage.Completed) != nil ||
			json.Unmarshal(gaps, &row.Coverage.Gaps) != nil {
			return nil, errors.New("stored Audit coverage is invalid")
		}
		if resultRef != nil || resultDigest != nil {
			if resultRef == nil || resultDigest == nil {
				return nil, errors.New("stored Audit coverage result shape is invalid")
			}
			if validateDigest("stored coverage result digest", *resultDigest) != nil {
				return nil, errors.New("stored Audit coverage result digest is invalid")
			}
			row.Result = &ExactArtifact{Digest: *resultDigest}
			if json.Unmarshal(resultRef, &row.Result.Ref) != nil || row.Result.Ref.ValidateExact() != nil {
				return nil, errors.New("stored Audit coverage result ref is invalid")
			}
		}
		result = append(result, row)
	}
	return result, rows.Err()
}

func (s *PostgresStore) GetReconcileSnapshot(
	ctx context.Context,
	claim ControllerClaim,
) (ReconcileSnapshot, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return ReconcileSnapshot{}, err
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, `
SELECT `+prefixedAuditColumns("audit")+`
  FROM audits AS audit
  JOIN audit_controller_claims AS claim USING (audit_id)
 WHERE audit.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
   AND claim.expires_at > clock_timestamp()`, claim.AuditID, claim.HolderID, claim.Epoch))
	if errors.Is(err, pgx.ErrNoRows) {
		return ReconcileSnapshot{}, ErrClaimLost
	}
	if err != nil {
		return ReconcileSnapshot{}, fmt.Errorf("read claimed Audit: %w", err)
	}
	result := ReconcileSnapshot{Audit: audit}
	if audit.CurrentRoundID != nil {
		round, roundErr := s.GetRound(ctx, audit.AuditID, *audit.CurrentRoundID)
		if roundErr != nil {
			return ReconcileSnapshot{}, roundErr
		}
		result.Round = &round
	}
	result.Items, err = s.listItems(ctx, audit.AuditID, MaxReconcileRows+1)
	if err != nil {
		return ReconcileSnapshot{}, err
	}
	if len(result.Items) > MaxReconcileRows {
		result.Items, result.MoreItems = result.Items[:MaxReconcileRows], true
	}
	result.Executions, err = s.listExecutions(ctx, audit.AuditID, MaxReconcileRows+1)
	if err != nil {
		return ReconcileSnapshot{}, err
	}
	if len(result.Executions) > MaxReconcileRows {
		result.Executions, result.MoreExecutions = result.Executions[:MaxReconcileRows], true
	}
	rows, err := s.db.Query(ctx, `
SELECT receipt_id, audit_id, execution_id, run_id,
       terminal_outcome, terminal_run_generation, terminal_run_sequence,
       disposition, error_code, request_digest, created_at
  FROM audit_collection_receipts
 WHERE audit_id = $1 ORDER BY created_at DESC, receipt_id DESC
 LIMIT $2`, audit.AuditID, MaxReconcileRows+1)
	if err != nil {
		return ReconcileSnapshot{}, fmt.Errorf("list Audit collection receipts: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		receipt, scanErr := scanReceiptSummary(rows)
		if scanErr != nil {
			return ReconcileSnapshot{}, fmt.Errorf("scan Audit collection receipt: %w", scanErr)
		}
		result.Receipts = append(result.Receipts, receipt)
	}
	if err := rows.Err(); err != nil {
		return ReconcileSnapshot{}, err
	}
	if len(result.Receipts) > MaxReconcileRows {
		result.Receipts, result.MoreReceipts = result.Receipts[:MaxReconcileRows], true
	}
	return result, nil
}

func scanRound(row scanner) (Round, error) {
	var result Round
	var encodedRef []byte
	var state string
	var revision int64
	if err := row.Scan(
		&result.RoundID, &result.AuditID, &result.Ordinal, &encodedRef,
		&result.Manifest.Digest, &state, &result.ExpectedItemCount,
		&revision, &result.CreatedAt, &result.UpdatedAt,
	); err != nil {
		return Round{}, err
	}
	if err := json.Unmarshal(encodedRef, &result.Manifest.Ref); err != nil || result.Manifest.Ref.ValidateExact() != nil {
		return Round{}, errors.New("stored Audit round manifest ref is invalid")
	}
	if validateDigest("stored round manifest digest", result.Manifest.Digest) != nil {
		return Round{}, errors.New("stored Audit round manifest digest is invalid")
	}
	result.State = RoundState(state)
	if revision <= 0 || !result.State.Valid() {
		return Round{}, errors.New("stored Audit round state is invalid")
	}
	result.Revision = uint64(revision)
	return result, nil
}

func scanItem(row scanner) (Item, error) {
	var result Item
	var taskRef, acceptedRef []byte
	var state string
	var final *string
	var acceptedDigest *string
	if err := row.Scan(
		&result.ItemID, &result.AuditID, &result.RoundID, &result.ItemKey,
		&result.Ordinal, &result.Kind, &result.SubjectKey, &taskRef,
		&result.Task.Digest, &result.WorkflowRole, &state, &final,
		&acceptedRef, &acceptedDigest, &result.LastExecutionItemID,
		&result.CreatedAt, &result.UpdatedAt,
	); err != nil {
		return Item{}, err
	}
	if err := json.Unmarshal(taskRef, &result.Task.Ref); err != nil || result.Task.Ref.ValidateExact() != nil {
		return Item{}, errors.New("stored Audit item task ref is invalid")
	}
	if validateDigest("stored item task digest", result.Task.Digest) != nil {
		return Item{}, errors.New("stored Audit item task digest is invalid")
	}
	result.State = ItemState(state)
	if !result.State.Valid() {
		return Item{}, errors.New("stored Audit item state is invalid")
	}
	if final != nil {
		value := FinalDisposition(*final)
		if !value.Valid() {
			return Item{}, errors.New("stored Audit item disposition is invalid")
		}
		result.FinalDisposition = &value
	}
	if acceptedRef != nil {
		if acceptedDigest == nil {
			return Item{}, errors.New("stored Audit item result shape is invalid")
		}
		result.AcceptedResult = &ExactArtifact{Digest: *acceptedDigest}
		if err := json.Unmarshal(acceptedRef, &result.AcceptedResult.Ref); err != nil || result.AcceptedResult.Ref.ValidateExact() != nil {
			return Item{}, errors.New("stored Audit item result ref is invalid")
		}
	} else if acceptedDigest != nil {
		return Item{}, errors.New("stored Audit item result shape is invalid")
	}
	return result, nil
}

func scanExecution(row scanner) (Execution, error) {
	var result Execution
	var encodedRef []byte
	var role, state string
	var roleAttempt *int
	var outcome *string
	var terminalSequence *int64
	if err := row.Scan(
		&result.ExecutionID, &result.AuditID, &result.RoundID, &role, &roleAttempt,
		&encodedRef, &result.Manifest.Digest, &result.SubmissionKey, &result.RequestDigest,
		&result.RunID, &state, &outcome, &result.TerminalRunGeneration,
		&terminalSequence, &result.TerminalObservedAt, &result.CreatedAt, &result.UpdatedAt,
	); err != nil {
		return Execution{}, err
	}
	if err := json.Unmarshal(encodedRef, &result.Manifest.Ref); err != nil || result.Manifest.Ref.ValidateExact() != nil {
		return Execution{}, errors.New("stored Audit execution manifest ref is invalid")
	}
	if validateDigest("stored execution manifest digest", result.Manifest.Digest) != nil ||
		validateDigest("stored execution request digest", result.RequestDigest) != nil {
		return Execution{}, errors.New("stored Audit execution digest is invalid")
	}
	result.Role, result.State, result.RoleAttempt = ExecutionRole(role), ExecutionState(state), roleAttempt
	if !result.Role.Valid() || !result.State.Valid() {
		return Execution{}, errors.New("stored Audit execution state is invalid")
	}
	if outcome != nil {
		value := TerminalOutcome(*outcome)
		if !value.Valid() {
			return Execution{}, errors.New("stored Audit terminal outcome is invalid")
		}
		result.TerminalOutcome = &value
	}
	if terminalSequence != nil {
		if *terminalSequence < 0 {
			return Execution{}, errors.New("stored Audit terminal sequence is invalid")
		}
		value := uint64(*terminalSequence)
		result.TerminalRunSequence = &value
	}
	return result, nil
}

func scanExecutionItem(row scanner) (ExecutionItem, error) {
	var result ExecutionItem
	var taskRef, encodedInputs, resultRef []byte
	var state string
	var disposition *string
	var resultDigest *string
	if err := row.Scan(
		&result.ExecutionItemID, &result.ExecutionID, &result.AuditID,
		&result.RoundID, &result.ItemID, &result.BatchOrdinal, &result.ItemAttempt,
		&taskRef, &result.Task.Digest, &encodedInputs, &state, &disposition,
		&resultRef, &resultDigest, &result.CreatedAt, &result.CollectedAt,
	); err != nil {
		return ExecutionItem{}, err
	}
	if err := json.Unmarshal(taskRef, &result.Task.Ref); err != nil || result.Task.Ref.ValidateExact() != nil ||
		json.Unmarshal(encodedInputs, &result.Inputs) != nil {
		return ExecutionItem{}, errors.New("stored Audit execution item inputs are invalid")
	}
	if validateDigest("stored execution task digest", result.Task.Digest) != nil {
		return ExecutionItem{}, errors.New("stored Audit execution item task digest is invalid")
	}
	for _, input := range result.Inputs {
		if validateExactArtifact("stored execution input", input, false) != nil {
			return ExecutionItem{}, errors.New("stored Audit execution input is invalid")
		}
	}
	result.State = ItemState(state)
	if !result.State.Valid() {
		return ExecutionItem{}, errors.New("stored Audit execution item state is invalid")
	}
	if disposition != nil {
		value := CollectionDisposition(*disposition)
		if !value.Valid() {
			return ExecutionItem{}, errors.New("stored Audit execution item disposition is invalid")
		}
		result.CollectionDisposition = &value
	}
	if resultRef != nil || resultDigest != nil {
		if resultRef == nil || resultDigest == nil {
			return ExecutionItem{}, errors.New("stored Audit execution item result shape is invalid")
		}
		if validateDigest("stored execution item result digest", *resultDigest) != nil {
			return ExecutionItem{}, errors.New("stored Audit execution item result digest is invalid")
		}
		result.Result = &ExactArtifact{Digest: *resultDigest}
		if err := json.Unmarshal(resultRef, &result.Result.Ref); err != nil || result.Result.Ref.ValidateExact() != nil {
			return ExecutionItem{}, errors.New("stored Audit execution item result ref is invalid")
		}
	}
	return result, nil
}

func scanReceipt(row scanner) (CollectionReceipt, error) {
	var result CollectionReceipt
	var outcome, disposition string
	var sequence *int64
	var sourceRef, retained []byte
	var sourceDigest *string
	if err := row.Scan(
		&result.ReceiptID, &result.AuditID, &result.ExecutionID, &result.RunID,
		&outcome, &result.TerminalRunGeneration, &sequence, &disposition,
		&sourceRef, &sourceDigest, &retained, &result.ErrorCode,
		&result.RequestDigest, &result.CreatedAt,
	); err != nil {
		return CollectionReceipt{}, err
	}
	result.TerminalOutcome, result.Disposition = TerminalOutcome(outcome), CollectionDisposition(disposition)
	if !result.TerminalOutcome.Valid() || !result.Disposition.Valid() ||
		validateDigest("stored receipt request digest", result.RequestDigest) != nil {
		return CollectionReceipt{}, errors.New("stored Audit receipt state is invalid")
	}
	if sequence != nil {
		if *sequence < 0 {
			return CollectionReceipt{}, errors.New("stored Audit receipt sequence is invalid")
		}
		value := uint64(*sequence)
		result.TerminalRunSequence = &value
	}
	if sourceRef != nil || sourceDigest != nil {
		if sourceRef == nil || sourceDigest == nil {
			return CollectionReceipt{}, errors.New("stored Audit receipt source shape is invalid")
		}
		if validateDigest("stored receipt source digest", *sourceDigest) != nil {
			return CollectionReceipt{}, errors.New("stored Audit receipt source digest is invalid")
		}
		result.SourceOutput = &ExactArtifact{Digest: *sourceDigest}
		if err := json.Unmarshal(sourceRef, &result.SourceOutput.Ref); err != nil || result.SourceOutput.Ref.ValidateExact() != nil {
			return CollectionReceipt{}, errors.New("stored Audit receipt source ref is invalid")
		}
	}
	if err := json.Unmarshal(retained, &result.Retained); err != nil {
		return CollectionReceipt{}, errors.New("stored Audit retained refs are invalid")
	}
	for _, link := range result.Retained {
		if validateExactArtifact("stored retained artifact", link.Artifact, true) != nil ||
			validateJSONObject("stored retained provenance", link.SourceProvenance, 1<<20) != nil {
			return CollectionReceipt{}, errors.New("stored Audit retained artifact is invalid")
		}
	}
	return result, nil
}

func scanReceiptSummary(row scanner) (CollectionReceiptSummary, error) {
	var result CollectionReceiptSummary
	var outcome, disposition string
	var sequence *int64
	if err := row.Scan(
		&result.ReceiptID, &result.AuditID, &result.ExecutionID, &result.RunID,
		&outcome, &result.TerminalRunGeneration, &sequence, &disposition,
		&result.ErrorCode, &result.RequestDigest, &result.CreatedAt,
	); err != nil {
		return CollectionReceiptSummary{}, err
	}
	result.TerminalOutcome = TerminalOutcome(outcome)
	result.Disposition = CollectionDisposition(disposition)
	if !result.TerminalOutcome.Valid() || !result.Disposition.Valid() ||
		validateDigest("stored receipt request digest", result.RequestDigest) != nil {
		return CollectionReceiptSummary{}, errors.New("stored Audit receipt summary is invalid")
	}
	if sequence != nil {
		if *sequence < 0 {
			return CollectionReceiptSummary{}, errors.New("stored Audit receipt summary sequence is invalid")
		}
		value := uint64(*sequence)
		result.TerminalRunSequence = &value
	}
	return result, nil
}
