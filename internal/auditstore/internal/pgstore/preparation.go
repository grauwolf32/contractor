package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *QueryStore) ListPreparationExecutions(ctx context.Context, auditID string) ([]auditstore.Execution, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	return s.listRoleExecutions(ctx, auditID, nil)
}

// ListPreparationRoles is independent of the paged Round/execution history;
// even an Audit with no executions exposes each pinned role as pending.
func (s *QueryStore) ListPreparationRoles(ctx context.Context, auditID string) ([]auditstore.PreparationRole, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, listPreparationRolesSQL, auditID)
	if err != nil {
		return nil, fmt.Errorf("list Audit preparation roles: %w", err)
	}
	defer rows.Close()
	result := []auditstore.PreparationRole{}
	for rows.Next() {
		var role auditstore.PreparationRole
		if err := rows.Scan(&role.WorkflowRole, &role.Attempts, &role.MaxRunAttempts, &role.ExecutionID, &role.Status); err != nil {
			return nil, err
		}
		result = append(result, role)
	}
	return result, rows.Err()
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
	if replay, found, err := s.lookupAuditReplay(ctx, p.OwnerID, auditstore.MutationStart, p.IdempotencyKey, p.RequestDigest); err != nil || found {
		return replay, false, err
	}
	audit, err := s.get(ctx, p.OwnerID, p.AuditID)
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil || !profile.HasPreparation() || profile.Ref.Name != audit.Profile.Name ||
		profile.Ref.Version != audit.Profile.Version || profile.Ref.Digest != audit.Profile.Digest {
		return auditstore.Audit{}, false, invalidf("preparation requires a valid pinned current-schema profile")
	}
	var baseline struct {
		Schema    string                              `json:"schema"`
		Inputs    map[string]auditstore.ExactArtifact `json:"inputs"`
		Inventory json.RawMessage                     `json:"inventory"`
	}
	if json.Unmarshal(p.BaselineSnapshot, &baseline) != nil || baseline.Schema != "contractor.audit.baseline.v1" ||
		baseline.Inputs == nil || baseline.Inventory != nil {
		return auditstore.Audit{}, false, invalidf("preparation baseline requires original inputs and forbids inventory")
	}
	for name, input := range profile.Inputs {
		artifact, exists := baseline.Inputs[name]
		if input.Required && !exists {
			return auditstore.Audit{}, false, invalidf("preparation baseline is missing input %s", name)
		}
		if exists && (validateExactArtifact("baseline input", artifact, true) != nil || !contracts.AcceptsMediaType(input.MediaTypes, artifact.MediaType)) {
			return auditstore.Audit{}, false, invalidf("preparation baseline input %s is invalid", name)
		}
	}
	for name := range baseline.Inputs {
		if _, exists := profile.Inputs[name]; !exists {
			return auditstore.Audit{}, false, invalidf("preparation baseline names an undeclared input")
		}
	}
	links, _ := prepareRetainedLinks(auditstore.CollectParams{Retained: p.InitialRetained})
	result, err := scanAudit(persistencepostgres.SavepointQueryRow(ctx, s.db, startPreparationSQL,
		p.OwnerID, p.AuditID, p.ExpectedRevision, []byte(p.BaselineSnapshot), optionalDeadline(p.DeadlineAt),
		links, retainedBytes, p.IdempotencyKey, p.RequestDigest,
	))
	if err == nil {
		return result, true, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting {
		return auditstore.Audit{}, false, auditstore.ErrProjectDeleting
	}
	if errors.Is(err, pgx.ErrNoRows) || persistencepostgres.Classify(err) == persistencepostgres.ClassDuplicate {
		if replay, found, replayErr := s.lookupAuditReplay(ctx, p.OwnerID, auditstore.MutationStart, p.IdempotencyKey, p.RequestDigest); replayErr != nil || found {
			return replay, false, replayErr
		}
		return auditstore.Audit{}, false, auditstore.ErrPrecondition
	}
	return auditstore.Audit{}, false, fmt.Errorf("start Audit preparation: %w", err)
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
	var encoded []byte
	err := s.db.QueryRow(ctx, `
SELECT execution.execution_id, execution.workflow_role, execution.role_attempt, execution.run_id, output
  FROM audit_executions AS execution CROSS JOIN LATERAL jsonb_array_elements(execution.preparation_outputs) AS output
 WHERE execution.audit_id = $1 AND execution.workflow_role = $2 AND execution.role = 'prepare'
   AND execution.collection_disposition = 'accepted-result' AND output->>'logicalName' = $3`, auditID, workflowRole, logicalName).
		Scan(&result.ExecutionID, &result.WorkflowRole, &result.RoleAttempt, &result.RunID, &encoded)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.AcceptedPreparationOutput{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.AcceptedPreparationOutput{}, fmt.Errorf("read accepted preparation output: %w", err)
	}
	if err := decodeStoredProvenance(encoded, &result.Output); err != nil {
		return auditstore.AcceptedPreparationOutput{}, err
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
		len(snapshot.Inputs) > config.MaxAuditWorkflowMappings || len(snapshot.Parameters) > config.MaxAuditWorkflowMappings {
		return invalidf("preparation inputs and parameters are required and bounded")
	}
	for name, artifact := range snapshot.Inputs {
		if validateText("input name", name, 128, true) != nil || validateExactArtifact("preparation input", artifact, true) != nil {
			return invalidf("preparation input is invalid")
		}
	}
	for name, value := range snapshot.Parameters {
		if validateText("parameter name", name, 128, true) != nil || validateText("parameter value", value, config.MaxAuditLiteralParamBytes, false) != nil {
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
	if len(p.PreparationOutputs) > config.MaxAuditWorkflowMappings ||
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
		p.Retained = append(p.Retained, auditstore.ArtifactLink{
			LogicalKey: "prepare:" + p.ExecutionID + ":" + output.LogicalName,
			Artifact:   output.Retained, SourceProvenance: provenance,
		})
	}
	return p, nil
}
