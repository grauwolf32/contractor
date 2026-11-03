package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *QueryStore) ListPreparationExecutions(ctx context.Context, auditID string) ([]auditstore.Execution, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	return s.listRoleExecutions(ctx, auditID, nil)
}

// ListPreparedPage reads one page of Audits, as List does, and the
// preparation of the listed Audits whose phase precedes Rounds (ASC-R150). A
// pool-backed store reads both in one read-only REPEATABLE READ transaction,
// so the page shows every Audit, and its preparation, as of one commit: a
// concurrent mutation or purge is in neither read or, if it committed first,
// in both. A transaction-backed store reads them in the caller's transaction,
// whose isolation level the caller owns (PER-R18). It issues the list
// statement and, only when a listed Audit is preparing or inventory, one
// preparation statement, whatever the number of such Audits. Between the two
// it only checks each scanned phase: profiles are decoded after the
// transaction, so it is not held open for per-Audit work. The preparation
// maps each such Audit to its pinned prepare roles in role order, each with
// only its latest attempt; an Audit pins at most
// auditprofile.MaxAuditProfileWorkflows roles.
func (s *QueryStore) ListPreparedPage(
	ctx context.Context, params auditstore.ListParams,
) ([]auditstore.Audit, map[string][]auditstore.ListedPreparationRole, error) {
	if err := validateList(params); err != nil {
		return nil, nil, err
	}
	var audits []auditstore.Audit
	var roles map[string][]auditstore.ListedPreparationRole
	err := persistencepostgres.Within(ctx, s.db, snapshotTransaction, func(tx pgx.Tx) error {
		var err error
		audits, err = NewQueryStore(tx, s.storage).List(ctx, params)
		if err != nil {
			return err
		}
		selected := make([]string, 0, len(audits))
		for _, audit := range audits {
			if audit.Phase.PrecedesRounds() {
				selected = append(selected, audit.AuditID)
			}
		}
		roles = map[string][]auditstore.ListedPreparationRole{}
		if len(selected) == 0 {
			return nil
		}
		roles, err = listPreparationPage(ctx, tx, params.OwnerID, selected)
		return err
	})
	if err != nil {
		return nil, nil, err
	}
	return audits, roles, nil
}

// GetPreparedDetail reads owner ownerID's Audit auditID and the rows of its
// detail projection (ASC-R149). A pool-backed store reads them in one
// read-only REPEATABLE READ transaction, so the Audit, its revision and every
// row come from one commit: a write that commits during the read is in none
// of them or, if it committed first, in all, and never fails the read. A
// transaction-backed store reads them in the caller's transaction, whose
// isolation level the caller owns (PER-R18). It issues the Audit statement
// and, unless the Audit has not started, the preparation statement of the
// Audit lists (ASC-R150), so a list item and the detail project the same
// rows. An Audit in the rounds phase that is not deleting adds the
// initial-inventory link and Round 1 statements, and currentRound adds the
// current Round, unless that is Round 1, and its items (at most
// auditprofile.MaxAuditItemsPerRound). No statement count depends on the
// Audit's roles, attempts, Rounds or items, and the transaction reads no
// Artifact payload: the caller reads the exact snapshot the link names after
// it commits.
func (s *QueryStore) GetPreparedDetail(
	ctx context.Context, ownerID, auditID string, currentRound bool,
) (auditstore.PreparedDetail, error) {
	var result auditstore.PreparedDetail
	err := persistencepostgres.Within(ctx, s.db, snapshotTransaction, func(tx pgx.Tx) error {
		reads := NewQueryStore(tx, s.storage)
		audit, err := reads.get(ctx, ownerID, auditID)
		if err != nil {
			return err
		}
		result = auditstore.PreparedDetail{Audit: audit}
		if audit.Phase == auditdomain.AuditPhaseNotStarted {
			return nil
		}
		roles, err := listPreparationPage(ctx, tx, ownerID, []string{auditID})
		if err != nil {
			return err
		}
		result.Roles = roles[auditID]
		// Only generated inventory has the link, and only an Audit that pins
		// a prepare role generates its inventory.
		if len(result.Roles) != 0 && audit.Phase == auditdomain.AuditPhaseRounds && audit.State != auditstore.AuditDeleting {
			link, err := reads.GetArtifactLink(ctx, auditID, auditstore.InitialInventoryLogicalKey)
			if err != nil {
				return err
			}
			first, err := reads.getRound(ctx, auditID, auditdomain.DeterministicID("round", auditID, "1"))
			if err != nil {
				return err
			}
			result.InventoryLink, result.FirstRound = &link, &first
		}
		if !currentRound || audit.CurrentRoundID == nil {
			return nil
		}
		if result.FirstRound != nil && result.FirstRound.RoundID == *audit.CurrentRoundID {
			current := *result.FirstRound
			result.CurrentRound = &current
		} else {
			current, err := reads.getRound(ctx, auditID, *audit.CurrentRoundID)
			if err != nil {
				return err
			}
			result.CurrentRound = &current
		}
		result.CurrentItems, err = reads.ListRoundItems(ctx, auditID, *audit.CurrentRoundID)
		return err
	})
	if err != nil {
		return auditstore.PreparedDetail{}, err
	}
	return result, nil
}

// listPreparationPage reads the preparation of owner ownerID's Audits
// auditIDs, which a list or a detail read in the same snapshot, in one
// statement. A list fetches at most pagelimit.MaxFetchRows Audits.
func listPreparationPage(
	ctx context.Context, db persistencepostgres.DBTX, ownerID string, auditIDs []string,
) (map[string][]auditstore.ListedPreparationRole, error) {
	if err := validateText("ownerID", ownerID, 256, true); err != nil {
		return nil, err
	}
	if len(auditIDs) == 0 || len(auditIDs) > pagelimit.MaxFetchRows {
		return nil, invalidf("listed Audit selection is invalid")
	}
	result := make(map[string][]auditstore.ListedPreparationRole, len(auditIDs))
	for _, auditID := range auditIDs {
		if validateID("auditID", auditID) != nil {
			return nil, invalidf("listed Audit identity is invalid")
		}
		if _, exists := result[auditID]; exists {
			return nil, invalidf("listed Audit selection contains duplicates")
		}
		result[auditID] = []auditstore.ListedPreparationRole{}
	}
	rows, err := db.Query(ctx, listPreparationPageSQL, ownerID, auditIDs)
	if err != nil {
		return nil, fmt.Errorf("list Audit page preparation: %w", err)
	}
	defer rows.Close()
	seen := make(map[string]bool, len(auditIDs))
	for rows.Next() {
		auditID, role, scanErr := scanListedPreparationRole(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit page preparation: %w", scanErr)
		}
		roles, listed := result[auditID]
		if !listed {
			return nil, errors.New("stored Audit page preparation names an unlisted Audit")
		}
		seen[auditID] = true
		if role == nil {
			continue
		}
		if len(roles) == auditprofile.MaxAuditProfileWorkflows {
			return nil, errors.New("stored Audit preparation roles exceed their profile bound")
		}
		result[auditID] = append(roles, *role)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit page preparation: %w", err)
	}
	if len(seen) != len(auditIDs) {
		return nil, errors.New("stored Audit page preparation omitted a listed Audit")
	}
	return result, nil
}

// scanListedPreparationRole reads one row of listPreparationPageSQL. Role is
// nil when the Audit pins no prepare role.
func scanListedPreparationRole(row scanner) (string, *auditstore.ListedPreparationRole, error) {
	var auditID, status string
	var key, executionID, runID, state *string
	var attempts int
	var maxAttempts, roleAttempt *int
	var outputs []byte
	if err := row.Scan(&auditID, &key, &attempts, &maxAttempts, &executionID,
		&roleAttempt, &runID, &state, &outputs, &status); err != nil {
		return "", nil, err
	}
	if key == nil {
		return auditID, nil, nil
	}
	role := auditstore.ListedPreparationRole{PreparationRole: auditstore.PreparationRole{
		WorkflowRole: *key, Status: auditdomain.PreparationStatus(status), Attempts: attempts, ExecutionID: executionID,
	}}
	if validateText("Workflow role", *key, 128, true) != nil || maxAttempts == nil || !role.Status.Valid() {
		return "", nil, errors.New("derived Audit preparation status is invalid")
	}
	role.MaxRunAttempts = *maxAttempts
	if executionID == nil {
		if roleAttempt != nil || runID != nil || state != nil || outputs != nil {
			return "", nil, errors.New("stored Audit preparation role has an attempt without identity")
		}
		return auditID, &role, nil
	}
	if roleAttempt == nil || state == nil || !auditstore.ExecutionState(*state).Valid() {
		return "", nil, errors.New("stored preparation role has no matching bounded execution")
	}
	latest := auditstore.PreparationAttempt{
		ExecutionID: *executionID, RoleAttempt: *roleAttempt, RunID: runID, State: auditstore.ExecutionState(*state),
	}
	if decodeStoredProvenance(outputs, &latest.Outputs) != nil || len(latest.Outputs) > auditprofile.MaxAuditWorkflowMappings {
		return "", nil, errors.New("stored Audit preparation outputs are invalid")
	}
	role.Latest = &latest
	return auditID, &role, nil
}

// StartPreparation pins the original baseline before any preparation Run or
// generated inventory exists.
func (s *OwnerStore) StartPreparation(ctx context.Context, p auditstore.StartPreparationParams) (auditstore.Audit, bool, error) {
	if validateText("ownerID", p.OwnerID, 256, true) != nil || validateID("auditID", p.AuditID) != nil ||
		p.ExpectedRevision == 0 || p.ExpectedRevision > math.MaxInt64 {
		return auditstore.Audit{}, false, invalidf("preparation start identity is invalid")
	}
	if err := validateIdempotency(p.IdempotencyKey, p.RequestDigest); err != nil {
		return auditstore.Audit{}, false, err
	}
	if err := validateJSONObject("preparation baseline", p.BaselineSnapshot, auditstore.MaxSnapshotBytes); err != nil {
		return auditstore.Audit{}, false, err
	}
	if len(p.InitialRetained) > auditstore.MaxArtifactLinksPerCall {
		return auditstore.Audit{}, false, invalidf("too many initial preparation links")
	}
	retainedBytes, err := validateArtifactLinks(p.InitialRetained)
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	return s.keyedAuditMutation(ctx, p.OwnerID, auditstore.MutationStart, p.IdempotencyKey, p.RequestDigest,
		func(tx pgx.Tx) (auditMutationResult, error) {
			started, err := startPreparation(ctx, tx, p, retainedBytes)
			if err != nil {
				return auditMutationResult{}, err
			}
			return auditMutationResult{
				audit: started, resourceID: started.AuditID, response: map[string]string{"auditId": started.AuditID},
			}, nil
		})
}

// startPreparation checks the draft's pinned profile against the start's
// baseline and enters preparation in tx.
func startPreparation(
	ctx context.Context, tx pgx.Tx, p auditstore.StartPreparationParams, retainedBytes int64,
) (auditstore.Audit, error) {
	audit, err := (&store{db: tx}).get(ctx, p.OwnerID, p.AuditID)
	if err != nil {
		return auditstore.Audit{}, err
	}
	profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil || !profile.HasPreparation() || profile.Ref.Name != audit.Profile.Name ||
		profile.Ref.Version != audit.Profile.Version || profile.Ref.Digest != audit.Profile.Digest {
		return auditstore.Audit{}, invalidf("preparation requires a valid pinned current-schema profile")
	}
	var baseline struct {
		Schema    string                              `json:"schema"`
		Inputs    map[string]auditstore.ExactArtifact `json:"inputs"`
		Inventory json.RawMessage                     `json:"inventory"`
	}
	if json.Unmarshal(p.BaselineSnapshot, &baseline) != nil || baseline.Schema != "contractor.audit.baseline.v1" ||
		baseline.Inputs == nil || baseline.Inventory != nil {
		return auditstore.Audit{}, invalidf("preparation baseline requires original inputs and forbids inventory")
	}
	for name, input := range profile.Inputs {
		artifact, exists := baseline.Inputs[name]
		if input.Required && !exists {
			return auditstore.Audit{}, invalidf("preparation baseline is missing input %s", name)
		}
		if exists && (validateExactArtifact("baseline input", artifact, true) != nil || !contracts.AcceptsMediaType(input.MediaTypes, artifact.MediaType)) {
			return auditstore.Audit{}, invalidf("preparation baseline input %s is invalid", name)
		}
	}
	for name := range baseline.Inputs {
		if _, exists := profile.Inputs[name]; !exists {
			return auditstore.Audit{}, invalidf("preparation baseline names an undeclared input")
		}
	}
	links, _ := prepareRetainedLinks(auditstore.CollectParams{Retained: p.InitialRetained})
	result, err := scanAudit(persistencepostgres.SavepointQueryRow(ctx, tx, startPreparationSQL,
		p.OwnerID, p.AuditID, p.ExpectedRevision, []byte(p.BaselineSnapshot), optionalDeadline(p.DeadlineAt),
		links, retainedBytes,
	))
	if err == nil {
		return result, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting {
		return auditstore.Audit{}, auditstore.ErrProjectDeleting
	}
	if errors.Is(err, pgx.ErrNoRows) || persistencepostgres.Classify(err) == persistencepostgres.ClassDuplicate {
		return auditstore.Audit{}, auditstore.ErrPrecondition
	}
	return auditstore.Audit{}, writeError("start Audit preparation", err)
}

// CompletePreparation advances only after every prepare role has an accepted
// receipt. Pause/cancel/delete preserve receipts and prevent this transition.
func (s *ClaimStore) CompletePreparation(ctx context.Context, claim auditstore.ControllerClaim, expectedRevision uint64) (auditstore.Audit, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return auditstore.Audit{}, err
	}
	if expectedRevision == 0 || expectedRevision > math.MaxInt64 {
		return auditstore.Audit{}, invalidf("preparation revision is invalid")
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, completePreparationSQL,
		claim.AuditID, claim.HolderID, claim.Epoch, expectedRevision,
	))
	if err == nil {
		return audit, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting {
		return auditstore.Audit{}, auditstore.ErrProjectDeleting
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, fmt.Errorf("complete Audit preparation: %w", err)
	}
	if live, err := s.claimLive(ctx, claim); err != nil {
		return auditstore.Audit{}, err
	} else if !live {
		return auditstore.Audit{}, auditstore.ErrClaimLost
	}
	return auditstore.Audit{}, auditstore.ErrPrecondition
}

func (s *QueryStore) GetPreparationOutput(ctx context.Context, auditID, workflowRole, logicalName string) (auditstore.AcceptedPreparationOutput, error) {
	if validateID("auditID", auditID) != nil || validateText("Workflow role", workflowRole, 128, true) != nil || validateText("output name", logicalName, 128, true) != nil {
		return auditstore.AcceptedPreparationOutput{}, invalidf("preparation output identity is invalid")
	}
	var result auditstore.AcceptedPreparationOutput
	var encoded, inputs []byte
	err := s.db.QueryRow(ctx, `
SELECT execution.execution_id, execution.workflow_role, execution.role_attempt, COALESCE(execution.run_id, execution.run_provenance->>'runId'), output, execution.preparation_snapshot->'inputs'
  FROM audit_executions AS execution CROSS JOIN LATERAL jsonb_array_elements(execution.preparation_outputs) AS output
 WHERE execution.audit_id = $1 AND execution.workflow_role = $2 AND execution.role = 'prepare'
   AND execution.collection_disposition = 'accepted-result' AND output->>'logicalName' = $3`, auditID, workflowRole, logicalName).
		Scan(&result.ExecutionID, &result.WorkflowRole, &result.RoleAttempt, &result.RunID, &encoded, &inputs)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.AcceptedPreparationOutput{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.AcceptedPreparationOutput{}, fmt.Errorf("read accepted preparation output: %w", err)
	}
	if err := decodeStoredProvenance(encoded, &result.Output); err != nil {
		return auditstore.AcceptedPreparationOutput{}, err
	}
	if err := decodeStoredProvenance(inputs, &result.Inputs); err != nil || result.Inputs == nil || len(result.Inputs) > auditprofile.MaxAuditWorkflowMappings {
		return auditstore.AcceptedPreparationOutput{}, invalidf("accepted preparation inputs are invalid")
	}
	return result, nil
}

func decodePreparation(execution *auditstore.Execution, snapshot, outputs []byte) error {
	if snapshot != nil {
		var value auditstore.PreparationSnapshot
		if decodeStoredProvenance(snapshot, &value) != nil || validatePreparationSnapshot(&value) != nil || execution.Role != auditstore.ExecutionPrepare {
			return errors.New("stored Audit preparation snapshot is invalid")
		}
		execution.Preparation = &value
	} else if execution.Role == auditstore.ExecutionPrepare {
		return errors.New("stored Audit preparation snapshot is missing")
	}
	if decodeStoredProvenance(outputs, &execution.PreparationOutputs) != nil {
		return errors.New("stored Audit preparation outputs are invalid")
	}
	return nil
}

func validatePreparationSnapshot(snapshot *auditstore.PreparationSnapshot) error {
	if snapshot == nil || snapshot.Inputs == nil || snapshot.Parameters == nil ||
		len(snapshot.Inputs) > auditprofile.MaxAuditWorkflowMappings || len(snapshot.Parameters) > auditprofile.MaxAuditWorkflowMappings {
		return invalidf("preparation inputs and parameters are required and bounded")
	}
	for name, artifact := range snapshot.Inputs {
		if validateText("input name", name, 128, true) != nil || validateExactArtifact("preparation input", artifact, true) != nil {
			return invalidf("preparation input is invalid")
		}
	}
	for name, value := range snapshot.Parameters {
		if validateText("parameter name", name, 128, true) != nil || validateText("parameter value", value, auditprofile.MaxAuditLiteralParamBytes, false) != nil {
			return invalidf("preparation parameter is invalid")
		}
	}
	encoded, err := json.Marshal(snapshot)
	if err != nil || len(encoded) > auditstore.MaxExecutionInputsBytes {
		return invalidf("preparation snapshot exceeds bounds")
	}
	return nil
}

// Outputs become ordinary immutable Audit links in the same receipt write.
// The DB additionally proves their frozen Project binding and import lineage.
func withPreparationLinks(p auditstore.CollectParams) (auditstore.CollectParams, error) {
	if len(p.PreparationOutputs) > auditprofile.MaxAuditWorkflowMappings ||
		(len(p.PreparationOutputs) != 0 && (p.Disposition != auditstore.CollectionAccepted || len(p.Items) != 0)) {
		return p, invalidf("preparation output collection shape is invalid")
	}
	seen := map[string]bool{}
	p.Retained = append([]auditstore.ArtifactLink{}, p.Retained...)
	for _, output := range p.PreparationOutputs {
		if validateText("logical output name", output.LogicalName, 128, true) != nil ||
			validateText("Workflow output name", output.WorkflowOutput, 128, true) != nil || seen[output.LogicalName] ||
			validateExactArtifact("preparation source", output.Source, true) != nil ||
			validateExactArtifact("preparation retained output", output.Retained, true) != nil ||
			output.Source.Ref.Namespace != "outputs" || output.Source.Ref.Name != output.WorkflowOutput ||
			output.Retained.Ref.Namespace != auditdomain.ArtifactNamespace(p.Claim.AuditID) ||
			output.Source.Digest != output.Retained.Digest || output.Source.MediaType != output.Retained.MediaType || output.Source.SizeBytes != output.Retained.SizeBytes {
			return p, invalidf("preparation output identity or exact retention is invalid")
		}
		seen[output.LogicalName] = true
		provenance, _ := json.Marshal(struct {
			Schema      string                       `json:"schema"`
			ExecutionID string                       `json:"executionId"`
			Output      auditstore.PreparationOutput `json:"output"`
		}{"contractor.audit.preparation-output.v1", p.ExecutionID, output})
		p.Retained = append(p.Retained, auditstore.ArtifactLink{Scope: "audit", LogicalKey: "prepare:" + p.ExecutionID + ":" + output.LogicalName,
			Artifact: output.Retained, SourceProvenance: provenance,
		})
	}
	return p, nil
}
