package pgstore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5"
)

// listWindowExecutions reads the reconcile window of unsettled executions,
// one more than MaxReconcileRows so the caller can detect more, together with
// the lifecycle state of each bound Run keyed by Run ID. The join replaces one
// Run read per submitted execution; a Run whose row is gone has no state.
func (s *ClaimStore) listWindowExecutions(
	ctx context.Context, auditID string,
) ([]auditstore.Execution, map[string]runstore.WorkflowRunState, error) {
	rows, err := s.db.Query(ctx, `
SELECT `+ExecutionColumns("execution")+`, run.state
  FROM audit_executions AS execution
  LEFT JOIN workflow_runs AS run ON run.run_id = execution.run_id
 WHERE execution.audit_id = $1 AND execution.state IN `+uncollectedExecutionStatesSQL+`
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
		execution, scanErr := ScanExecution(withColumns{row: rows, after: []any{&runState}})
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
SELECT artifact_scope, logical_key, artifact_ref, artifact_digest, media_type, size_bytes,
       source_provenance, display_ref, created_at
  FROM audit_artifact_links
 WHERE audit_id = $1 AND logical_key = $2`, auditID, logicalKey).Scan(
		&result.Scope, &result.LogicalKey, &encodedRef, &result.Artifact.Digest,
		&result.Artifact.MediaType, &result.Artifact.SizeBytes,
		&result.SourceProvenance, &result.DisplayRef, &result.CreatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ArtifactLink{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.ArtifactLink{}, fmt.Errorf("read Audit artifact link: %w", err)
	}
	if result.Scope != "project" && result.Scope != "audit" {
		return auditstore.ArtifactLink{}, invalidf("stored Audit artifact scope is invalid")
	}
	if json.Unmarshal(encodedRef, &result.Artifact.Ref) != nil ||
		validateExactArtifact("stored Audit artifact link", result.Artifact, true) != nil ||
		validateJSONObject("stored Audit artifact provenance", result.SourceProvenance, 1<<20) != nil {
		return auditstore.ArtifactLink{}, errors.New("stored Audit artifact link is invalid")
	}
	return result, nil
}

// snapshotTransaction reads every part of a reconcile snapshot, of an event
// page, of an Audit list page and its preparation or of an Audit's detail,
// from one commit. On a pool, persistencepostgres.Within replays
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
		result, err = NewClaimStore(tx, s.storage).readReconcileSnapshot(ctx, claim)
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
	result.MaxConcurrentToolRuns = settings.MaxConcurrentToolRuns
	return s.readTaskReconcileSnapshot(ctx, result)
}

func scanItem(row scanner) (auditstore.Item, error) {
	var result auditstore.Item
	var taskRef, origin, acceptedRef, dependencies []byte
	var state string
	var final *string
	var acceptedDigest *string
	if err := row.Scan(
		&result.ItemID, &result.AuditID, &result.RoundID, &result.ItemKey,
		&result.Ordinal, &result.Kind, &result.SubjectKey, &taskRef,
		&result.Task.Digest, &origin, &result.WorkflowRole, &state,
		&final,
		&acceptedRef, &acceptedDigest, &result.LastExecutionItemID,
		&result.CreatedAt, &result.UpdatedAt, &dependencies,
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
	if dependencies != nil {
		value, err := auditdomain.DecodeRetainedDependencies(dependencies)
		if err != nil || value.AuditID != result.AuditID || value.ItemKey != result.ItemKey {
			return auditstore.Item{}, errors.New("stored Audit retained dependencies are invalid")
		}
		result.Dependencies = &value
	}
	result.State = auditstore.ItemState(state)
	if !result.State.Valid() {
		return auditstore.Item{}, errors.New("stored Audit item state is invalid")
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

// ScanExecution reads one execution that ExecutionColumns selected and
// rejects a stored value that the store would not write: an inexact manifest
// ref or an invalid digest, an unknown role, state or terminal outcome, a
// negative terminal sequence, or Run provenance or a preparation snapshot
// that does not decode strictly.
func ScanExecution(row scanner) (auditstore.Execution, error) {
	var result auditstore.Execution
	var encodedRef []byte
	var manifestDigest *string
	var role, state string
	var roleAttempt *int
	var outcome *string
	var terminalSequence *int64
	var encodedProvenance []byte
	var preparation, outputs []byte
	if err := row.Scan(
		&result.ExecutionID, &result.AuditID, &result.RoundID, &role, &result.WorkflowRole, &roleAttempt,
		&encodedRef, &manifestDigest, &result.SubmissionKey, &result.RequestDigest,
		&result.RunID, &state, &outcome, &result.TerminalRunGeneration,
		&terminalSequence, &result.TerminalObservedAt, &encodedProvenance, &result.RunDeletedAt,
		&result.CreatedAt, &result.UpdatedAt, &preparation, &outputs,
	); err != nil {
		return auditstore.Execution{}, err
	}
	if role == string(auditstore.ExecutionTask) {
		if encodedRef != nil || manifestDigest != nil || result.RoundID != nil || roleAttempt != nil {
			return auditstore.Execution{}, errors.New("stored common Task execution contains retired manifest authority")
		}
	} else {
		if err := strictjson.Decode(encodedRef, &result.Manifest.Ref); err != nil || result.Manifest.Ref.ValidateExact() != nil || manifestDigest == nil || validateDigest("stored execution manifest digest", *manifestDigest) != nil {
			return auditstore.Execution{}, errors.New("stored Audit execution manifest ref is invalid")
		}
		result.Manifest.Digest = *manifestDigest
	}
	if validateDigest("stored execution request digest", result.RequestDigest) != nil {
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
