package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// PreparationSnapshot pins the complete resolved submission inputs and
// parameters before a Run exists. Recovery never resolves mutable bindings.
type PreparationSnapshot struct {
	Inputs     map[string]ExactArtifact `json:"inputs"`
	Parameters map[string]string        `json:"parameters"`
}

// PreparationOutput is stored on the immutable accepted execution receipt.
// Its containing execution supplies the Audit, role, attempt and source Run.
type PreparationOutput struct {
	LogicalName    string        `json:"logicalName"`
	WorkflowOutput string        `json:"workflowOutput"`
	Source         ExactArtifact `json:"source"`
	Retained       ExactArtifact `json:"retained"`
}

type AcceptedPreparationOutput struct {
	ExecutionID  string
	WorkflowRole string
	RoleAttempt  int
	RunID        string
	Output       PreparationOutput
}

type PreparationRole struct {
	WorkflowRole   string
	Status         auditdomain.PreparationStatus
	Attempts       int
	MaxRunAttempts int
	ExecutionID    *string
}

func (s *PostgresStore) ListPreparationExecutions(ctx context.Context, auditID string) ([]Execution, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	return s.listRoleExecutions(ctx, auditID, nil)
}

// ListPreparationRoles is independent of the paged Round/execution history;
// even an Audit with no executions exposes each pinned role as pending.
func (s *PostgresStore) ListPreparationRoles(ctx context.Context, auditID string) ([]PreparationRole, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, listPreparationRolesSQL, auditID)
	if err != nil {
		return nil, fmt.Errorf("list Audit preparation roles: %w", err)
	}
	defer rows.Close()
	result := []PreparationRole{}
	for rows.Next() {
		var role PreparationRole
		if err := rows.Scan(&role.WorkflowRole, &role.Attempts, &role.MaxRunAttempts, &role.ExecutionID, &role.Status); err != nil {
			return nil, err
		}
		result = append(result, role)
	}
	return result, rows.Err()
}

type StartPreparationParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	BaselineSnapshot json.RawMessage
	DeadlineAt       time.Time
	InitialRetained  []ArtifactLink
	IdempotencyKey   string
	RequestDigest    string
}

// StartPreparation pins the original baseline before any preparation Run or
// generated inventory exists.
func (s *PostgresStore) StartPreparation(ctx context.Context, p StartPreparationParams) (Audit, bool, error) {
	if validateText("ownerID", p.OwnerID, 256, true) != nil || validateID("auditID", p.AuditID) != nil ||
		p.ExpectedRevision == 0 || p.ExpectedRevision > math.MaxInt64 {
		return Audit{}, false, invalidf("preparation start identity is invalid")
	}
	if err := validateIdempotency(p.IdempotencyKey, p.RequestDigest); err != nil {
		return Audit{}, false, err
	}
	if err := validateJSONObject("preparation baseline", p.BaselineSnapshot, MaxSnapshotBytes); err != nil {
		return Audit{}, false, err
	}
	if len(p.InitialRetained) > MaxArtifactLinksPerCall {
		return Audit{}, false, invalidf("too many initial preparation links")
	}
	retainedBytes, err := validateArtifactLinks(p.InitialRetained)
	if err != nil {
		return Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(ctx, p.OwnerID, "audit.start", p.IdempotencyKey, p.RequestDigest); err != nil || found {
		return replay, false, err
	}
	audit, err := s.Get(ctx, p.OwnerID, p.AuditID)
	if err != nil {
		return Audit{}, false, err
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil || !profile.HasPreparation() || profile.Ref.Name != audit.Profile.Name ||
		profile.Ref.Version != audit.Profile.Version || profile.Ref.Digest != audit.Profile.Digest {
		return Audit{}, false, invalidf("preparation requires a valid pinned current-schema profile")
	}
	var baseline struct {
		Schema    string                   `json:"schema"`
		Inputs    map[string]ExactArtifact `json:"inputs"`
		Inventory json.RawMessage          `json:"inventory"`
	}
	if json.Unmarshal(p.BaselineSnapshot, &baseline) != nil || baseline.Schema != "contractor.audit.baseline.v1" ||
		baseline.Inputs == nil || baseline.Inventory != nil {
		return Audit{}, false, invalidf("preparation baseline requires original inputs and forbids inventory")
	}
	for name, input := range profile.Inputs {
		artifact, exists := baseline.Inputs[name]
		if input.Required && !exists {
			return Audit{}, false, invalidf("preparation baseline is missing input %s", name)
		}
		if exists && (validateExactArtifact("baseline input", artifact, true) != nil || !contracts.AcceptsMediaType(input.MediaTypes, artifact.MediaType)) {
			return Audit{}, false, invalidf("preparation baseline input %s is invalid", name)
		}
	}
	for name := range baseline.Inputs {
		if _, exists := profile.Inputs[name]; !exists {
			return Audit{}, false, invalidf("preparation baseline names an undeclared input")
		}
	}
	links, _ := prepareRetainedLinks(CollectParams{Retained: p.InitialRetained})
	result, err := scanAudit(s.db.QueryRow(ctx, startPreparationSQL,
		p.OwnerID, p.AuditID, p.ExpectedRevision, []byte(p.BaselineSnapshot), optionalDeadline(p.DeadlineAt),
		links, retainedBytes, p.IdempotencyKey, p.RequestDigest,
	))
	if err == nil {
		return result, true, nil
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return Audit{}, false, ErrProjectDeleting
	}
	if errors.Is(err, pgx.ErrNoRows) || persistencepostgres.SQLState(err) == persistencepostgres.SQLStateUniqueViolation {
		if replay, found, replayErr := s.lookupAuditReplay(ctx, p.OwnerID, "audit.start", p.IdempotencyKey, p.RequestDigest); replayErr != nil || found {
			return replay, false, replayErr
		}
		return Audit{}, false, ErrPrecondition
	}
	return Audit{}, false, fmt.Errorf("start Audit preparation: %w", err)
}

// CompletePreparation advances only after every prepare role has an accepted
// receipt. Pause/cancel/delete preserve receipts and prevent this transition.
func (s *PostgresStore) CompletePreparation(ctx context.Context, claim ControllerClaim, expectedRevision uint64) (Audit, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return Audit{}, err
	}
	if expectedRevision == 0 || expectedRevision > math.MaxInt64 {
		return Audit{}, invalidf("preparation revision is invalid")
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, completePreparationSQL,
		claim.AuditID, claim.HolderID, claim.Epoch, expectedRevision,
	))
	if err == nil {
		return audit, nil
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return Audit{}, ErrProjectDeleting
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, fmt.Errorf("complete Audit preparation: %w", err)
	}
	if live, err := s.claimLive(ctx, claim); err != nil {
		return Audit{}, err
	} else if !live {
		return Audit{}, ErrClaimLost
	}
	return Audit{}, ErrPrecondition
}

func (s *PostgresStore) GetPreparationOutput(ctx context.Context, auditID, workflowRole, logicalName string) (AcceptedPreparationOutput, error) {
	if validateID("auditID", auditID) != nil || validateText("Workflow role", workflowRole, 128, true) != nil || validateText("output name", logicalName, 128, true) != nil {
		return AcceptedPreparationOutput{}, invalidf("preparation output identity is invalid")
	}
	var result AcceptedPreparationOutput
	var encoded []byte
	err := s.db.QueryRow(ctx, `
SELECT execution.execution_id, execution.workflow_role, execution.role_attempt, execution.run_id, output
  FROM audit_executions AS execution CROSS JOIN LATERAL jsonb_array_elements(execution.preparation_outputs) AS output
 WHERE execution.audit_id = $1 AND execution.workflow_role = $2 AND execution.role = 'prepare'
   AND execution.collection_disposition = 'accepted-result' AND output->>'logicalName' = $3`, auditID, workflowRole, logicalName).
		Scan(&result.ExecutionID, &result.WorkflowRole, &result.RoleAttempt, &result.RunID, &encoded)
	if errors.Is(err, pgx.ErrNoRows) {
		return AcceptedPreparationOutput{}, ErrNotFound
	}
	if err != nil {
		return AcceptedPreparationOutput{}, fmt.Errorf("read accepted preparation output: %w", err)
	}
	if err := decodeStoredProvenance(encoded, &result.Output); err != nil {
		return AcceptedPreparationOutput{}, err
	}
	return result, nil
}

func decodePreparation(execution *Execution, snapshot, outputs []byte) error {
	if snapshot != nil {
		var value PreparationSnapshot
		if decodeStoredProvenance(snapshot, &value) != nil || validatePreparationSnapshot(&value) != nil || execution.Role != ExecutionPrepare {
			return errors.New("stored Audit preparation snapshot is invalid")
		}
		execution.Preparation = &value
	} else if execution.Role == ExecutionPrepare {
		return errors.New("stored Audit preparation snapshot is missing")
	}
	if decodeStoredProvenance(outputs, &execution.PreparationOutputs) != nil {
		return errors.New("stored Audit preparation outputs are invalid")
	}
	return nil
}

func validatePreparationSnapshot(snapshot *PreparationSnapshot) error {
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
	if err != nil || len(encoded) > MaxExecutionInputsBytes {
		return invalidf("preparation snapshot exceeds bounds")
	}
	return nil
}

// Outputs become ordinary immutable Audit links in the same receipt write.
// The DB additionally proves their frozen Project binding and import lineage.
func withPreparationLinks(p CollectParams) (CollectParams, error) {
	if len(p.PreparationOutputs) > config.MaxAuditWorkflowMappings ||
		(len(p.PreparationOutputs) != 0 && (p.Disposition != CollectionAccepted || len(p.Items) != 0)) {
		return p, invalidf("preparation output collection shape is invalid")
	}
	seen := map[string]bool{}
	p.Retained = append([]ArtifactLink{}, p.Retained...)
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
			Schema      string            `json:"schema"`
			ExecutionID string            `json:"executionId"`
			Output      PreparationOutput `json:"output"`
		}{"contractor.audit.preparation-output.v1", p.ExecutionID, output})
		p.Retained = append(p.Retained, ArtifactLink{
			LogicalKey: "prepare:" + p.ExecutionID + ":" + output.LogicalName,
			Artifact:   output.Retained, SourceProvenance: provenance,
		})
	}
	return p, nil
}
