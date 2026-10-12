package pgstore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5"
)

func (s *store) getRound(ctx context.Context, auditID, roundID string) (auditstore.Round, error) {
	if err := validateID("auditID", auditID); err != nil {
		return auditstore.Round{}, err
	}
	if err := validateID("roundID", roundID); err != nil {
		return auditstore.Round{}, err
	}
	result, err := scanRound(s.db.QueryRow(ctx, `
SELECT `+prefixedRoundColumns("audit_rounds")+` FROM audit_rounds
 WHERE audit_id = $1 AND round_id = $2`, auditID, roundID))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Round{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.Round{}, fmt.Errorf("read Audit round: %w", err)
	}
	return result, nil
}

// ListRounds returns every Round of one Audit in ordinal order.
func (s *QueryStore) ListRounds(ctx context.Context, auditID string) ([]auditstore.Round, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT `+prefixedRoundColumns("audit_rounds")+` FROM audit_rounds
 WHERE audit_id = $1 ORDER BY ordinal`, auditID)
	if err != nil {
		return nil, fmt.Errorf("list Audit rounds: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Round, 0)
	for rows.Next() {
		round, scanErr := scanRound(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit round: %w", scanErr)
		}
		result = append(result, round)
	}
	return result, rows.Err()
}

func (s *QueryStore) ListItems(ctx context.Context, auditID string) ([]auditstore.Item, error) {
	return s.listItems(ctx, auditID, 0)
}

// ListItemsByIDs reads the exact bounded member set of one execution. Unlike
// the reconcile snapshot, its result does not depend on item ordinal or state.
func (s *QueryStore) ListItemsByIDs(ctx context.Context, auditID string, itemIDs []string) ([]auditstore.Item, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if len(itemIDs) == 0 || len(itemIDs) > auditstore.MaxCollectionItems {
		return nil, invalidf("Audit item selection is invalid")
	}
	seen := make(map[string]struct{}, len(itemIDs))
	for _, itemID := range itemIDs {
		if err := validateID("itemID", itemID); err != nil {
			return nil, err
		}
		if _, exists := seen[itemID]; exists {
			return nil, invalidf("Audit item selection contains duplicates")
		}
		seen[itemID] = struct{}{}
	}
	rows, err := s.db.Query(ctx, `
SELECT `+prefixedItemColumns("item")+`
  FROM audit_items AS item
 WHERE item.audit_id = $1 AND item.item_id = ANY($2::text[])`, auditID, itemIDs)
	if err != nil {
		return nil, fmt.Errorf("list selected Audit items: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Item, 0, len(itemIDs))
	for rows.Next() {
		item, scanErr := scanItem(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan selected Audit item: %w", scanErr)
		}
		result = append(result, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate selected Audit items: %w", err)
	}
	return result, nil
}

// ListItemsPage is the owner-safe, filter-before-keyset projection used by the
// public API. Controller reconciliation deliberately keeps its separate
// bounded non-settled scan below.
func (s *QueryStore) ListItemsPage(
	ctx context.Context, params auditstore.ListItemsParams,
) ([]auditstore.Item, error) {
	if err := validateListItems(params); err != nil {
		return nil, err
	}
	var state *string
	if params.State != nil {
		value := string(*params.State)
		state = &value
	}
	rows, err := s.db.Query(ctx, `
SELECT `+prefixedItemColumns("item")+`
  FROM audit_items AS item
  JOIN audit_rounds AS round USING (round_id, audit_id)
  JOIN audits AS audit USING (audit_id)
 WHERE audit.owner_id = $1 AND item.audit_id = $2
   AND ($3::text IS NULL OR item.round_id = $3)
   AND ($4::text IS NULL OR item.state = $4)
   AND ($5::text IS NULL OR item.subject_key = $5)
   AND ($6::integer IS NULL OR (round.ordinal, item.ordinal, item.item_id) > ($6, $7, $8))
 ORDER BY round.ordinal, item.ordinal, item.item_id
 LIMIT $9`, params.OwnerID, params.AuditID, params.RoundID, state,
		params.SubjectKey, params.AfterRoundOrdinal, params.AfterItemOrdinal,
		params.AfterItemID, params.Limit)
	if err != nil {
		return nil, fmt.Errorf("list owner Audit items: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Item, 0, params.Limit)
	for rows.Next() {
		item, scanErr := scanItem(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan owner Audit item: %w", scanErr)
		}
		result = append(result, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate owner Audit items: %w", err)
	}
	if len(result) == 0 {
		if _, err := s.get(ctx, params.OwnerID, params.AuditID); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func (s *store) listItems(ctx context.Context, auditID string, limit int) ([]auditstore.Item, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT `+prefixedItemColumns("item")+`
  FROM audit_items AS item
  JOIN audit_rounds AS round USING (round_id, audit_id)
 WHERE item.audit_id = $1
   AND ($2::integer = 0 OR item.state <> 'settled')
 ORDER BY ($2::integer <> 0 AND item.state = 'awaiting_review'),
          round.ordinal, item.ordinal, item.item_id
 LIMIT CASE WHEN $2::integer = 0 THEN 100001 ELSE $2 END`, auditID, limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit items: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Item, 0)
	for rows.Next() {
		item, scanErr := scanItem(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit item: %w", scanErr)
		}
		result = append(result, item)
	}
	return result, rows.Err()
}

func (s *ClaimStore) listReadyItems(ctx context.Context, auditID, roundID string) ([]auditstore.Item, error) {
	rows, err := s.db.Query(ctx, `
SELECT `+prefixedItemColumns("item")+`
  FROM audit_items AS item
 WHERE item.audit_id = $1 AND item.round_id = $2 AND item.state = 'ready'
 ORDER BY item.ordinal, item.item_id
 LIMIT $3`, auditID, roundID, auditstore.MaxReconcileRows)
	if err != nil {
		return nil, fmt.Errorf("list ready Audit items: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Item, 0)
	for rows.Next() {
		item, scanErr := scanItem(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan ready Audit item: %w", scanErr)
		}
		result = append(result, item)
	}
	return result, rows.Err()
}

// onlyAwaitingReview reports whether at least one item is unsettled and every
// unsettled item awaits human review.
func (s *store) onlyAwaitingReview(ctx context.Context, auditID string) (bool, error) {
	var unsettled, awaiting int64
	if err := s.db.QueryRow(ctx, `
SELECT count(*), count(*) FILTER (WHERE state = 'awaiting_review')
  FROM audit_items
 WHERE audit_id = $1 AND state <> 'settled'`, auditID).Scan(&unsettled, &awaiting); err != nil {
		return false, fmt.Errorf("count Audit items awaiting review: %w", err)
	}
	return unsettled != 0 && unsettled == awaiting, nil
}

func (s *QueryStore) ListExecutions(ctx context.Context, auditID string) ([]auditstore.Execution, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT `+prefixedExecutionColumns("audit_executions")+` FROM audit_executions
 WHERE audit_id = $1
 ORDER BY created_at, execution_id
 LIMIT 1000001`, auditID)
	if err != nil {
		return nil, fmt.Errorf("list Audit executions: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Execution, 0)
	for rows.Next() {
		execution, scanErr := scanExecution(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit execution: %w", scanErr)
		}
		result = append(result, execution)
	}
	return result, rows.Err()
}

// listWindowExecutions reads the reconcile window of unsettled executions,
// one more than MaxReconcileRows so the caller can detect more, together with
// the lifecycle state of each bound Run keyed by Run ID. The join replaces one
// Run read per submitted execution; a Run whose row is gone has no state.
func (s *ClaimStore) listWindowExecutions(
	ctx context.Context, auditID string,
) ([]auditstore.Execution, map[string]runstore.WorkflowRunState, error) {
	rows, err := s.db.Query(ctx, `
SELECT `+prefixedExecutionColumns("execution")+`, run.state
  FROM audit_executions AS execution
  LEFT JOIN workflow_runs AS run ON run.run_id = execution.run_id
 WHERE execution.audit_id = $1 AND execution.state IN ('intent', 'submitted', 'collecting')
 ORDER BY execution.created_at, execution.execution_id
 LIMIT $2`, auditID, auditstore.MaxReconcileRows+1)
	if err != nil {
		return nil, nil, fmt.Errorf("list Audit executions: %w", err)
	}
	defer rows.Close()
	executions := make([]auditstore.Execution, 0)
	states := make(map[string]runstore.WorkflowRunState)
	for rows.Next() {
		var runState *string
		execution, scanErr := scanExecution(withColumns{row: rows, after: []any{&runState}})
		if scanErr != nil {
			return nil, nil, fmt.Errorf("scan Audit execution: %w", scanErr)
		}
		if runState != nil {
			state := runstore.WorkflowRunState(*runState)
			if execution.RunID == nil || !state.Valid() {
				return nil, nil, errors.New("stored Audit child Run state is invalid")
			}
			states[*execution.RunID] = state
		}
		executions = append(executions, execution)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("iterate Audit executions: %w", err)
	}
	return executions, states, nil
}

func (s *store) listExecutionItems(ctx context.Context, executionID string) ([]auditstore.ExecutionItem, error) {
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
	result := make([]auditstore.ExecutionItem, 0)
	for rows.Next() {
		item, scanErr := scanExecutionItem(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit execution item: %w", scanErr)
		}
		result = append(result, item)
	}
	return result, rows.Err()
}

// ListItemAttempts returns bounded tombstone-safe attempt history for one
// already owner-authorized Audit page. Filters are applied in SQL before the
// result is grouped, so unrelated Audit data is never loaded for projection.
func (s *QueryStore) ListItemAttempts(
	ctx context.Context, ownerID, auditID string, itemIDs []string,
) (map[string][]auditstore.ItemAttempt, error) {
	if err := validateText("ownerID", ownerID, 256, true); err != nil {
		return nil, err
	}
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if len(itemIDs) == 0 || len(itemIDs) > pagelimit.MaxRows {
		return nil, invalidf("Audit attempt item selection is invalid")
	}
	seen := make(map[string]struct{}, len(itemIDs))
	for _, itemID := range itemIDs {
		if err := validateID("itemID", itemID); err != nil {
			return nil, err
		}
		if _, exists := seen[itemID]; exists {
			return nil, invalidf("Audit attempt item selection contains duplicates")
		}
		seen[itemID] = struct{}{}
	}
	rows, err := s.db.Query(ctx, listItemAttemptsSQL, ownerID, auditID, itemIDs, auditstore.MaxAttemptProjection+1)
	if err != nil {
		return nil, fmt.Errorf("list Audit item attempts: %w", err)
	}
	defer rows.Close()
	result := make(map[string][]auditstore.ItemAttempt, len(itemIDs))
	count := 0
	for rows.Next() {
		count++
		if count > auditstore.MaxAttemptProjection {
			return nil, errors.New("stored Audit attempt projection exceeds its bound")
		}
		var attempt auditstore.ItemAttempt
		var role, state string
		var disposition, outcome *string
		var resultRef, provenance []byte
		var resultDigest *string
		if err := rows.Scan(
			&attempt.ExecutionItemID, &attempt.ExecutionID, &attempt.ItemID,
			&attempt.ItemAttempt, &role, &state, &disposition, &resultRef, &resultDigest,
			&outcome, &attempt.RunID, &provenance, &attempt.RunDeleted,
			&attempt.CreatedAt, &attempt.CollectedAt,
		); err != nil {
			return nil, fmt.Errorf("scan Audit item attempt: %w", err)
		}
		attempt.Role, attempt.State = auditstore.ExecutionRole(role), auditstore.ItemState(state)
		if !attempt.Role.Valid() || !attempt.State.Valid() || attempt.ItemAttempt < 1 {
			return nil, errors.New("stored Audit item attempt is invalid")
		}
		if disposition != nil {
			value := auditstore.CollectionDisposition(*disposition)
			if !value.Valid() {
				return nil, errors.New("stored Audit item attempt disposition is invalid")
			}
			attempt.CollectionDisposition = &value
		}
		if outcome != nil {
			value := auditstore.TerminalOutcome(*outcome)
			if !value.Valid() {
				return nil, errors.New("stored Audit item attempt outcome is invalid")
			}
			attempt.TerminalOutcome = &value
		}
		if resultRef != nil {
			if resultDigest == nil {
				return nil, errors.New("stored Audit item attempt result is invalid")
			}
			attempt.Result = &auditstore.ExactArtifact{Digest: *resultDigest}
			if json.Unmarshal(resultRef, &attempt.Result.Ref) != nil || attempt.Result.Ref.ValidateExact() != nil ||
				validateDigest("stored Audit item attempt result", *resultDigest) != nil {
				return nil, errors.New("stored Audit item attempt result is invalid")
			}
		} else if resultDigest != nil {
			return nil, errors.New("stored Audit item attempt result is invalid")
		}
		if err := decodeRunProvenance(provenance, attempt.RunID, &attempt.RunProvenance); err != nil {
			return nil, err
		}
		result[attempt.ItemID] = append(result[attempt.ItemID], attempt)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit item attempts: %w", err)
	}
	if count == 0 {
		if _, err := s.get(ctx, ownerID, auditID); err != nil {
			return nil, err
		}
	}
	for _, itemID := range itemIDs {
		if result[itemID] == nil {
			result[itemID] = []auditstore.ItemAttempt{}
		}
	}
	return result, nil
}

// NextItemAttempts returns the next policy attempt of each listed item that is
// still ready, read for the whole candidate batch in one statement. An item
// that is no longer ready is absent from the result. The values are advisory
// until CreateExecutionIntent atomically verifies them again; binding the
// read to a live claim keeps Controller recovery from inspecting another
// Controller's in-flight Audit.
func (s *ClaimStore) NextItemAttempts(
	ctx context.Context, claim auditstore.ControllerClaim, itemIDs []string,
) (map[string]int, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return nil, err
	}
	if len(itemIDs) == 0 || len(itemIDs) > auditstore.MaxCollectionItems {
		return nil, invalidf("Audit attempt item selection is invalid")
	}
	seen := make(map[string]struct{}, len(itemIDs))
	for _, itemID := range itemIDs {
		if err := validateID("itemID", itemID); err != nil {
			return nil, err
		}
		if _, exists := seen[itemID]; exists {
			return nil, invalidf("Audit attempt item selection contains duplicates")
		}
		seen[itemID] = struct{}{}
	}
	rows, err := s.db.Query(ctx, nextItemAttemptsSQL, claim.AuditID, claim.HolderID, claim.Epoch, itemIDs)
	if err != nil {
		return nil, fmt.Errorf("read next Audit item attempts: %w", err)
	}
	defer rows.Close()
	result := make(map[string]int, len(itemIDs))
	for rows.Next() {
		var itemID string
		var attempt int
		if err := rows.Scan(&itemID, &attempt); err != nil {
			return nil, fmt.Errorf("scan next Audit item attempt: %w", err)
		}
		if attempt < 1 || attempt > 11 {
			return nil, errors.New("stored Audit item attempt is invalid")
		}
		result[itemID] = attempt
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate next Audit item attempts: %w", err)
	}
	rows.Close()
	if len(result) == len(itemIDs) {
		return result, nil
	}
	if live, liveErr := s.claimLive(ctx, claim); liveErr != nil {
		return nil, liveErr
	} else if !live {
		return nil, auditstore.ErrClaimLost
	}
	return result, nil
}

// ListEvents returns a forward, bounded cursor page. Sequence zero starts at
// the first immutable event.
func (s *QueryStore) ListEvents(
	ctx context.Context, auditID string, after uint64, limit int,
) ([]auditstore.Event, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if after > math.MaxInt64 || !pagelimit.ValidFetch(limit) {
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
	result := make([]auditstore.Event, 0, limit)
	for rows.Next() {
		var event auditstore.Event
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

func (s *QueryStore) ListCoverage(
	ctx context.Context, auditID, roundID string, afterOrdinal int, limit int,
) ([]auditstore.CoverageRow, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if err := validateID("roundID", roundID); err != nil {
		return nil, err
	}
	if afterOrdinal < -1 || !pagelimit.ValidFetch(limit) {
		return nil, invalidf("coverage page is invalid")
	}
	rows, err := s.db.Query(ctx, listCoverageSQL, auditID, roundID, afterOrdinal, limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit coverage: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.CoverageRow, 0, limit)
	for rows.Next() {
		var row auditstore.CoverageRow
		var status string
		var requested, completed, gaps, resultRef, taskRef []byte
		var resultDigest *string
		if err := rows.Scan(
			&row.AuditID, &row.RoundID, &row.ItemID, &row.Ordinal, &row.ItemKey,
			&row.SubjectKey, &status, &requested, &completed, &gaps,
			&row.Coverage.Rationale, &resultRef, &resultDigest, &row.UpdatedAt,
			&taskRef, &row.Task.Digest,
		); err != nil {
			return nil, fmt.Errorf("scan Audit coverage: %w", err)
		}
		row.Coverage.Status = auditstore.CoverageStatus(status)
		if json.Unmarshal(taskRef, &row.Task.Ref) != nil || row.Task.Ref.ValidateExact() != nil ||
			validateDigest("stored coverage task digest", row.Task.Digest) != nil {
			return nil, errors.New("stored Audit coverage task is invalid")
		}
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
			row.Result = &auditstore.ExactArtifact{Digest: *resultDigest}
			if json.Unmarshal(resultRef, &row.Result.Ref) != nil || row.Result.Ref.ValidateExact() != nil {
				return nil, errors.New("stored Audit coverage result ref is invalid")
			}
		}
		result = append(result, row)
	}
	return result, rows.Err()
}

func (s *QueryStore) GetArtifactLink(
	ctx context.Context, auditID, logicalKey string,
) (auditstore.ArtifactLink, error) {
	if err := validateID("auditID", auditID); err != nil {
		return auditstore.ArtifactLink{}, err
	}
	if err := validateText("artifact logical key", logicalKey, 512, true); err != nil {
		return auditstore.ArtifactLink{}, err
	}
	var result auditstore.ArtifactLink
	var encodedRef []byte
	err := s.db.QueryRow(ctx, `
SELECT logical_key, artifact_ref, artifact_digest, media_type, size_bytes,
       source_provenance, display_ref, created_at
  FROM audit_artifact_links
 WHERE audit_id = $1 AND logical_key = $2`, auditID, logicalKey).Scan(
		&result.LogicalKey, &encodedRef, &result.Artifact.Digest,
		&result.Artifact.MediaType, &result.Artifact.SizeBytes,
		&result.SourceProvenance, &result.DisplayRef, &result.CreatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ArtifactLink{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.ArtifactLink{}, fmt.Errorf("read Audit artifact link: %w", err)
	}
	if json.Unmarshal(encodedRef, &result.Artifact.Ref) != nil ||
		validateExactArtifact("stored Audit artifact link", result.Artifact, true) != nil ||
		validateJSONObject("stored Audit artifact provenance", result.SourceProvenance, 1<<20) != nil {
		return auditstore.ArtifactLink{}, errors.New("stored Audit artifact link is invalid")
	}
	return result, nil
}

func (s *QueryStore) CollectionDispositionCounts(
	ctx context.Context, auditID string,
) (auditstore.CollectionDispositionCounts, error) {
	if err := validateID("auditID", auditID); err != nil {
		return auditstore.CollectionDispositionCounts{}, err
	}
	var result auditstore.CollectionDispositionCounts
	err := s.db.QueryRow(ctx, `
SELECT count(*) FILTER (WHERE collection_disposition = 'accepted-result'),
       count(*) FILTER (WHERE collection_disposition = 'missing-output'),
       count(*) FILTER (WHERE collection_disposition = 'invalid-result'),
       count(*) FILTER (WHERE collection_disposition = 'execution-failed'),
       count(*) FILTER (WHERE collection_disposition = 'execution-cancelled'),
       count(*) FILTER (WHERE collection_disposition = 'collection-contract-invalid')
  FROM audit_executions WHERE audit_id = $1 AND collection_receipt_id IS NOT NULL`, auditID).Scan(
		&result.AcceptedResult, &result.MissingOutput, &result.InvalidResult,
		&result.ExecutionFailed, &result.ExecutionCancelled,
		&result.ContractInvalid,
	)
	if err != nil {
		return auditstore.CollectionDispositionCounts{}, fmt.Errorf("count Audit collection dispositions: %w", err)
	}
	return result, nil
}

// snapshotTransaction reads every part of a reconcile snapshot, or of an
// event page, from one commit. On a pool, persistencepostgres.Within replays
// a transaction conflict, but a read-only REPEATABLE READ transaction takes
// no row locks and cannot fail with a serialization conflict, so that retry
// never triggers and the snapshot is never read twice.
var snapshotTransaction = pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly}

// GetReconcileSnapshot reads the claimed Audit's reconcile snapshot. A
// pool-backed store reads all of it in one read-only REPEATABLE READ
// transaction, so an owner mutation or child Run transition is visible in
// every part of the snapshot or in none (AUX-R95); a transaction-backed store
// reads it in the caller's transaction, whose isolation level the caller
// owns (PER-R18). The number of statements does not depend on the Audit's
// items, executions or child Runs.
func (s *ClaimStore) GetReconcileSnapshot(
	ctx context.Context,
	claim auditstore.ControllerClaim,
) (auditstore.ReconcileSnapshot, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return auditstore.ReconcileSnapshot{}, err
	}
	var result auditstore.ReconcileSnapshot
	err := persistencepostgres.Within(ctx, s.db, snapshotTransaction, func(tx pgx.Tx) error {
		var err error
		result, err = NewClaimStore(tx).readReconcileSnapshot(ctx, claim)
		return err
	})
	return result, err
}

func (s *ClaimStore) readReconcileSnapshot(
	ctx context.Context,
	claim auditstore.ControllerClaim,
) (auditstore.ReconcileSnapshot, error) {
	audit, err := scanAudit(s.db.QueryRow(ctx, `
SELECT `+prefixedAuditColumns("audit")+`
  FROM audits AS audit
  JOIN audit_controller_claims AS claim USING (audit_id)
 WHERE audit.audit_id = $1 AND `+auditClaims.LiveCondition("claim", "$2", "$3")+``, claim.AuditID, claim.HolderID, claim.Epoch))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ReconcileSnapshot{}, auditstore.ErrClaimLost
	}
	if err != nil {
		return auditstore.ReconcileSnapshot{}, fmt.Errorf("read claimed Audit: %w", err)
	}
	result := auditstore.ReconcileSnapshot{Audit: audit}
	settings, err := settingsstore.NewPostgresStore(s.db).GetSchedulerSettings(ctx)
	if err != nil {
		return auditstore.ReconcileSnapshot{}, fmt.Errorf("read Audit dispatch window: %w", err)
	}
	result.MaxConcurrentRuns = settings.MaxConcurrentRuns
	if audit.CurrentRoundID != nil {
		round, roundErr := s.getRound(ctx, audit.AuditID, *audit.CurrentRoundID)
		if roundErr != nil {
			return auditstore.ReconcileSnapshot{}, roundErr
		}
		result.Round = &round
	}
	result.Items, err = s.listItems(ctx, audit.AuditID, auditstore.MaxReconcileRows+1)
	if err != nil {
		return auditstore.ReconcileSnapshot{}, err
	}
	if len(result.Items) > auditstore.MaxReconcileRows {
		result.Items, result.MoreItems = result.Items[:auditstore.MaxReconcileRows], true
	}
	if result.Round != nil {
		result.ReadyItems, err = s.listReadyItems(ctx, audit.AuditID, result.Round.RoundID)
		if err != nil {
			return auditstore.ReconcileSnapshot{}, err
		}
	}
	result.OnlyAwaitingReview, err = s.onlyAwaitingReview(ctx, audit.AuditID)
	if err != nil {
		return auditstore.ReconcileSnapshot{}, err
	}
	result.Executions, result.RunStates, err = s.listWindowExecutions(ctx, audit.AuditID)
	if err != nil {
		return auditstore.ReconcileSnapshot{}, err
	}
	if len(result.Executions) > auditstore.MaxReconcileRows {
		result.Executions, result.MoreExecutions = result.Executions[:auditstore.MaxReconcileRows], true
	}
	var currentRoundID *string
	if result.Round != nil {
		currentRoundID = &result.Round.RoundID
	}
	result.RoleExecutions, err = s.listRoleExecutions(ctx, audit.AuditID, currentRoundID)
	if err != nil {
		return auditstore.ReconcileSnapshot{}, err
	}
	result.RoleReceipts, err = s.listRoleReceipts(ctx, audit.AuditID, currentRoundID)
	if err != nil {
		return auditstore.ReconcileSnapshot{}, err
	}
	rows, err := s.db.Query(ctx, listCollectionReceiptsSQL, audit.AuditID, auditstore.MaxReconcileRows+1)
	if err != nil {
		return auditstore.ReconcileSnapshot{}, fmt.Errorf("list Audit collection receipts: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		receipt, scanErr := scanReceiptSummary(rows)
		if scanErr != nil {
			return auditstore.ReconcileSnapshot{}, fmt.Errorf("scan Audit collection receipt: %w", scanErr)
		}
		result.Receipts = append(result.Receipts, receipt)
	}
	if err := rows.Err(); err != nil {
		return auditstore.ReconcileSnapshot{}, err
	}
	if len(result.Receipts) > auditstore.MaxReconcileRows {
		result.Receipts, result.MoreReceipts = result.Receipts[:auditstore.MaxReconcileRows], true
	}
	return result, nil
}

func (s *ClaimStore) listRoleReceipts(
	ctx context.Context, auditID string, roundID *string,
) ([]auditstore.CollectionReceiptSummary, error) {
	rows, err := s.db.Query(ctx, listRoleReceiptsSQL, auditID, roundID, auditstore.MaxAuditRoleExecutionsPerRound+1)
	if err != nil {
		return nil, fmt.Errorf("list Audit role collection receipts: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.CollectionReceiptSummary, 0)
	for rows.Next() {
		receipt, scanErr := scanReceiptSummary(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit role collection receipt: %w", scanErr)
		}
		result = append(result, receipt)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(result) > auditstore.MaxAuditRoleExecutionsPerRound {
		return nil, errors.New("stored Audit role receipts exceed their per-Round bound")
	}
	return result, nil
}

func (s *store) listRoleExecutions(
	ctx context.Context, auditID string, roundID *string,
) ([]auditstore.Execution, error) {
	rows, err := s.db.Query(ctx, `
SELECT `+prefixedExecutionColumns("audit_executions")+` FROM audit_executions
 WHERE audit_id = $1 AND (role = 'prepare' OR (
       role IN ('discovery', 'assessment') AND round_id IS NOT DISTINCT FROM $2))
 ORDER BY role, workflow_role, role_attempt, execution_id
 LIMIT $3`, auditID, roundID, auditstore.MaxAuditRoleExecutionsPerRound+1)
	if err != nil {
		return nil, fmt.Errorf("list Audit role executions: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Execution, 0)
	for rows.Next() {
		execution, scanErr := scanExecution(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit role execution: %w", scanErr)
		}
		result = append(result, execution)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(result) > auditstore.MaxAuditRoleExecutionsPerRound {
		return nil, errors.New("stored Audit role executions exceed their per-Round bound")
	}
	return result, nil
}

func scanRound(row scanner) (auditstore.Round, error) {
	var result auditstore.Round
	var encodedRef []byte
	var state string
	var revision int64
	if err := row.Scan(
		&result.RoundID, &result.AuditID, &result.Ordinal, &encodedRef,
		&result.Manifest.Digest, &state, &result.ExpectedItemCount,
		&revision, &result.CreatedAt, &result.UpdatedAt,
	); err != nil {
		return auditstore.Round{}, err
	}
	if err := json.Unmarshal(encodedRef, &result.Manifest.Ref); err != nil || result.Manifest.Ref.ValidateExact() != nil {
		return auditstore.Round{}, errors.New("stored Audit round manifest ref is invalid")
	}
	if validateDigest("stored round manifest digest", result.Manifest.Digest) != nil {
		return auditstore.Round{}, errors.New("stored Audit round manifest digest is invalid")
	}
	result.State = auditstore.RoundState(state)
	if revision <= 0 || !result.State.Valid() {
		return auditstore.Round{}, errors.New("stored Audit round state is invalid")
	}
	result.Revision = uint64(revision)
	return result, nil
}

func scanItem(row scanner) (auditstore.Item, error) {
	var result auditstore.Item
	var taskRef, origin, acceptedRef []byte
	var state, approvalKind string
	var approvalDigest *string
	var final *string
	var acceptedDigest *string
	if err := row.Scan(
		&result.ItemID, &result.AuditID, &result.RoundID, &result.ItemKey,
		&result.Ordinal, &result.Kind, &result.SubjectKey, &taskRef,
		&result.Task.Digest, &origin, &result.WorkflowRole, &state,
		&approvalKind, &approvalDigest, &final,
		&acceptedRef, &acceptedDigest, &result.LastExecutionItemID,
		&result.CreatedAt, &result.UpdatedAt,
	); err != nil {
		return auditstore.Item{}, err
	}
	if err := json.Unmarshal(taskRef, &result.Task.Ref); err != nil || result.Task.Ref.ValidateExact() != nil {
		return auditstore.Item{}, errors.New("stored Audit item task ref is invalid")
	}
	if validateDigest("stored item task digest", result.Task.Digest) != nil {
		return auditstore.Item{}, errors.New("stored Audit item task digest is invalid")
	}
	if err := decodeStoredProvenance(origin, &result.Origin); err != nil ||
		validateItemOrigin(result.Origin, result.ItemKey) != nil {
		return auditstore.Item{}, errors.New("stored Audit item origin is invalid")
	}
	result.State = auditstore.ItemState(state)
	if !result.State.Valid() {
		return auditstore.Item{}, errors.New("stored Audit item state is invalid")
	}
	result.ApprovalKind = auditstore.ItemApprovalKind(approvalKind)
	if !result.ApprovalKind.Valid() ||
		(result.ApprovalKind == auditstore.ItemApprovalNone) != (approvalDigest == nil) {
		return auditstore.Item{}, errors.New("stored Audit item approval is invalid")
	}
	if approvalDigest != nil {
		if validateDigest("stored item approval digest", *approvalDigest) != nil {
			return auditstore.Item{}, errors.New("stored Audit item approval digest is invalid")
		}
		result.ApprovalDigest = *approvalDigest
	}
	if final != nil {
		value := auditstore.FinalDisposition(*final)
		if !value.Valid() {
			return auditstore.Item{}, errors.New("stored Audit item disposition is invalid")
		}
		result.FinalDisposition = &value
	}
	if acceptedRef != nil {
		if acceptedDigest == nil {
			return auditstore.Item{}, errors.New("stored Audit item result shape is invalid")
		}
		result.AcceptedResult = &auditstore.ExactArtifact{Digest: *acceptedDigest}
		if err := json.Unmarshal(acceptedRef, &result.AcceptedResult.Ref); err != nil || result.AcceptedResult.Ref.ValidateExact() != nil {
			return auditstore.Item{}, errors.New("stored Audit item result ref is invalid")
		}
	} else if acceptedDigest != nil {
		return auditstore.Item{}, errors.New("stored Audit item result shape is invalid")
	}
	return result, nil
}

func scanExecution(row scanner) (auditstore.Execution, error) {
	var result auditstore.Execution
	var encodedRef []byte
	var role, state string
	var roleAttempt *int
	var outcome *string
	var terminalSequence *int64
	var encodedProvenance []byte
	var preparation, outputs []byte
	if err := row.Scan(
		&result.ExecutionID, &result.AuditID, &result.RoundID, &role, &result.WorkflowRole, &roleAttempt,
		&encodedRef, &result.Manifest.Digest, &result.SubmissionKey, &result.RequestDigest,
		&result.RunID, &state, &outcome, &result.TerminalRunGeneration,
		&terminalSequence, &result.TerminalObservedAt, &encodedProvenance, &result.RunDeletedAt,
		&result.CreatedAt, &result.UpdatedAt, &preparation, &outputs,
	); err != nil {
		return auditstore.Execution{}, err
	}
	if err := json.Unmarshal(encodedRef, &result.Manifest.Ref); err != nil || result.Manifest.Ref.ValidateExact() != nil {
		return auditstore.Execution{}, errors.New("stored Audit execution manifest ref is invalid")
	}
	if validateDigest("stored execution manifest digest", result.Manifest.Digest) != nil ||
		validateDigest("stored execution request digest", result.RequestDigest) != nil {
		return auditstore.Execution{}, errors.New("stored Audit execution digest is invalid")
	}
	result.Role, result.State, result.RoleAttempt = auditstore.ExecutionRole(role), auditstore.ExecutionState(state), roleAttempt
	if !result.Role.Valid() || !result.State.Valid() ||
		validateText("stored execution Workflow role", result.WorkflowRole, 128, true) != nil {
		return auditstore.Execution{}, errors.New("stored Audit execution state is invalid")
	}
	if outcome != nil {
		value := auditstore.TerminalOutcome(*outcome)
		if !value.Valid() {
			return auditstore.Execution{}, errors.New("stored Audit terminal outcome is invalid")
		}
		result.TerminalOutcome = &value
	}
	if terminalSequence != nil {
		if *terminalSequence < 0 {
			return auditstore.Execution{}, errors.New("stored Audit terminal sequence is invalid")
		}
		value := uint64(*terminalSequence)
		result.TerminalRunSequence = &value
	}
	if err := decodeRunProvenance(encodedProvenance, result.RunID, &result.RunProvenance); err != nil {
		return auditstore.Execution{}, err
	}
	if err := decodePreparation(&result, preparation, outputs); err != nil {
		return auditstore.Execution{}, err
	}
	return result, nil
}

// Historical or mixed provenance must fail without reinterpreting stored bytes.
func decodeStoredProvenance(encoded []byte, target any) error {
	if !json.Valid(encoded) {
		return errors.New("stored Audit provenance is not valid JSON")
	}
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	return decoder.Decode(target)
}

func decodeRunProvenance(encoded []byte, runID *string, target **auditstore.RunProvenance) error {
	if runID == nil {
		if encoded != nil {
			return errors.New("stored Audit execution unexpectedly has Run provenance")
		}
		return nil
	}
	if encoded == nil {
		return errors.New("stored Audit execution has no Run provenance")
	}
	var value auditstore.RunProvenance
	if decodeStoredProvenance(encoded, &value) != nil || value.Schema != "contractor.audit.run-provenance.v1" ||
		value.RunID != *runID || value.Workflow == nil {
		return errors.New("stored Audit execution Run provenance is invalid")
	}
	workflow := value.Workflow
	if validateText("stored Workflow name", workflow.Name, 128, true) != nil ||
		validateText("stored Workflow version", workflow.Version, 128, true) != nil ||
		validateText("stored Workflow schema version", workflow.SchemaVersion, 128, true) != nil ||
		validateDigest("stored Workflow closure digest", workflow.ClosureDigest) != nil ||
		workflow.ConfigurationRef.Name != workflow.Name || workflow.ConfigurationRef.Version != workflow.Version {
		return errors.New("stored Audit execution Workflow provenance is invalid")
	}
	*target = &value
	return nil
}

func scanExecutionItem(row scanner) (auditstore.ExecutionItem, error) {
	var result auditstore.ExecutionItem
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
		return auditstore.ExecutionItem{}, err
	}
	if err := json.Unmarshal(taskRef, &result.Task.Ref); err != nil || result.Task.Ref.ValidateExact() != nil ||
		json.Unmarshal(encodedInputs, &result.Inputs) != nil {
		return auditstore.ExecutionItem{}, errors.New("stored Audit execution item inputs are invalid")
	}
	if validateDigest("stored execution task digest", result.Task.Digest) != nil {
		return auditstore.ExecutionItem{}, errors.New("stored Audit execution item task digest is invalid")
	}
	for _, input := range result.Inputs {
		if validateExactArtifact("stored execution input", input, false) != nil {
			return auditstore.ExecutionItem{}, errors.New("stored Audit execution input is invalid")
		}
	}
	result.State = auditstore.ItemState(state)
	if !result.State.Valid() {
		return auditstore.ExecutionItem{}, errors.New("stored Audit execution item state is invalid")
	}
	if disposition != nil {
		value := auditstore.CollectionDisposition(*disposition)
		if !value.Valid() {
			return auditstore.ExecutionItem{}, errors.New("stored Audit execution item disposition is invalid")
		}
		result.CollectionDisposition = &value
	}
	if resultRef != nil || resultDigest != nil {
		if resultRef == nil || resultDigest == nil {
			return auditstore.ExecutionItem{}, errors.New("stored Audit execution item result shape is invalid")
		}
		if validateDigest("stored execution item result digest", *resultDigest) != nil {
			return auditstore.ExecutionItem{}, errors.New("stored Audit execution item result digest is invalid")
		}
		result.Result = &auditstore.ExactArtifact{Digest: *resultDigest}
		if err := json.Unmarshal(resultRef, &result.Result.Ref); err != nil || result.Result.Ref.ValidateExact() != nil {
			return auditstore.ExecutionItem{}, errors.New("stored Audit execution item result ref is invalid")
		}
	}
	return result, nil
}

func scanReceipt(row scanner) (auditstore.CollectionReceipt, error) {
	var result auditstore.CollectionReceipt
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
		return auditstore.CollectionReceipt{}, err
	}
	result.TerminalOutcome, result.Disposition = auditstore.TerminalOutcome(outcome), auditstore.CollectionDisposition(disposition)
	if !result.TerminalOutcome.Valid() || !result.Disposition.Valid() ||
		validateDigest("stored receipt request digest", result.RequestDigest) != nil {
		return auditstore.CollectionReceipt{}, errors.New("stored Audit receipt state is invalid")
	}
	if sequence != nil {
		if *sequence < 0 {
			return auditstore.CollectionReceipt{}, errors.New("stored Audit receipt sequence is invalid")
		}
		value := uint64(*sequence)
		result.TerminalRunSequence = &value
	}
	if sourceRef != nil || sourceDigest != nil {
		if sourceRef == nil || sourceDigest == nil {
			return auditstore.CollectionReceipt{}, errors.New("stored Audit receipt source shape is invalid")
		}
		if validateDigest("stored receipt source digest", *sourceDigest) != nil {
			return auditstore.CollectionReceipt{}, errors.New("stored Audit receipt source digest is invalid")
		}
		result.SourceOutput = &auditstore.ExactArtifact{Digest: *sourceDigest}
		if err := json.Unmarshal(sourceRef, &result.SourceOutput.Ref); err != nil || result.SourceOutput.Ref.ValidateExact() != nil {
			return auditstore.CollectionReceipt{}, errors.New("stored Audit receipt source ref is invalid")
		}
	}
	if err := json.Unmarshal(retained, &result.Retained); err != nil {
		return auditstore.CollectionReceipt{}, errors.New("stored Audit retained refs are invalid")
	}
	for _, link := range result.Retained {
		if validateExactArtifact("stored retained artifact", link.Artifact, true) != nil ||
			validateJSONObject("stored retained provenance", link.SourceProvenance, 1<<20) != nil {
			return auditstore.CollectionReceipt{}, errors.New("stored Audit retained artifact is invalid")
		}
	}
	return result, nil
}

func scanReceiptSummary(row scanner) (auditstore.CollectionReceiptSummary, error) {
	var result auditstore.CollectionReceiptSummary
	var outcome, disposition string
	var sequence *int64
	if err := row.Scan(
		&result.ReceiptID, &result.AuditID, &result.ExecutionID, &result.RunID,
		&outcome, &result.TerminalRunGeneration, &sequence, &disposition,
		&result.ErrorCode, &result.RequestDigest, &result.CreatedAt,
	); err != nil {
		return auditstore.CollectionReceiptSummary{}, err
	}
	result.TerminalOutcome = auditstore.TerminalOutcome(outcome)
	result.Disposition = auditstore.CollectionDisposition(disposition)
	if !result.TerminalOutcome.Valid() || !result.Disposition.Valid() ||
		validateDigest("stored receipt request digest", result.RequestDigest) != nil {
		return auditstore.CollectionReceiptSummary{}, errors.New("stored Audit receipt summary is invalid")
	}
	if sequence != nil {
		if *sequence < 0 {
			return auditstore.CollectionReceiptSummary{}, errors.New("stored Audit receipt summary sequence is invalid")
		}
		value := uint64(*sequence)
		result.TerminalRunSequence = &value
	}
	return result, nil
}
