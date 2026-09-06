package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	maxProposalsPerRun       = 4096
	maxProposalEvidenceBytes = 64 << 20
)

type Service struct {
	pool *pgxpool.Pool
}

func New(pool *pgxpool.Pool) (*Service, error) {
	if pool == nil {
		return nil, errors.New("finding intake PostgreSQL pool is required")
	}
	return &Service{pool: pool}, nil
}

// FindReplay returns an already committed exact receipt without consulting the
// allocation write fence. The caller must still authenticate the live grant.
func (s *Service) FindReplay(
	ctx context.Context,
	grant controlplane.AllocationGrant,
	input Submission,
) (Receipt, bool, error) {
	canonical, err := canonicalize(input)
	if err != nil {
		return Receipt{}, false, err
	}
	receipt, err := readReceiptBySubmission(
		ctx, s.pool, grant.AllocationID, input.InvocationID, input.SubmissionID,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return Receipt{}, false, nil
	}
	if err != nil {
		return Receipt{}, false, err
	}
	if receipt.RequestDigest != canonical.digest || receipt.Origin.RunID != grant.RunID ||
		receipt.Origin.StageExecutionID != grant.StageExecutionID ||
		receipt.Origin.LogicalAgentName != grant.LogicalAgentName {
		return Receipt{}, false, ErrConflict
	}
	return receipt, true, nil
}

// Submit commits proposal bytes, exact evidence pins, and the replay receipt
// in one transaction. Its caller holds the Registry write-grant critical
// section for the entire call, serializing this commit with SetWriteFence.
func (s *Service) Submit(
	ctx context.Context,
	grant controlplane.AllocationGrant,
	input Submission,
) (Receipt, bool, error) {
	canonical, err := canonicalize(input)
	if err != nil {
		return Receipt{}, false, err
	}
	var result Receipt
	var replayed bool
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		// Serialize the initially absent idempotency row as well as ordinary
		// replays. A transaction advisory lock avoids a race in which two
		// byte-identical requests both advance the proposal Artifact binding
		// before either receipt becomes visible.
		lockKey := deterministicID(
			"submission-lock", grant.AllocationID, input.InvocationID, input.SubmissionID,
		)
		if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, lockKey); err != nil {
			return fmt.Errorf("lock finding proposal submission: %w", err)
		}
		existing, readErr := readReceiptBySubmission(
			ctx, tx, grant.AllocationID, input.InvocationID, input.SubmissionID,
		)
		if readErr == nil {
			if existing.RequestDigest != canonical.digest ||
				existing.Origin.RunID != grant.RunID ||
				existing.Origin.StageExecutionID != grant.StageExecutionID ||
				existing.Origin.LogicalAgentName != grant.LogicalAgentName {
				return ErrConflict
			}
			result, replayed = existing, true
			return nil
		}
		if !errors.Is(readErr, pgx.ErrNoRows) {
			return readErr
		}

		origin, ownerID, projectID, err := verifyTrustedExecution(ctx, tx, grant)
		if err != nil {
			return err
		}
		origin.InvocationID = canonical.request.InvocationID
		var count int
		if err := tx.QueryRow(ctx, `
SELECT count(*) FROM finding_proposal_receipts WHERE run_id = $1`, grant.RunID).Scan(&count); err != nil {
			return fmt.Errorf("count Run finding proposals: %w", err)
		}
		if count >= maxProposalsPerRun {
			return fmt.Errorf("%w: Run proposal quota exceeded", ErrInvalid)
		}

		artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		runArtifacts, err := artifactService.Run(grant.RunID)
		if err != nil {
			return err
		}
		evidence := make([]ExactArtifact, 0, len(canonical.request.EvidenceRefs))
		var evidenceBytes int64
		for _, ref := range canonical.request.EvidenceRefs {
			metadata, metadataErr := runArtifacts.Metadata(ctx, ref)
			if metadataErr != nil {
				if errors.Is(metadataErr, artifacts.ErrArtifactNotFound) {
					return ErrAccessDenied
				}
				return metadataErr
			}
			evidenceBytes += metadata.Size
			if evidenceBytes > maxProposalEvidenceBytes {
				return fmt.Errorf("%w: proposal evidence quota exceeded", ErrInvalid)
			}
			evidence = append(evidence, ExactArtifact{
				Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType,
				SizeBytes: metadata.Size,
			})
		}

		receiptID := deterministicID(
			"receipt", grant.AllocationID, input.InvocationID, input.SubmissionID,
		)
		proposalID := deterministicID(
			"proposal", grant.AllocationID, input.InvocationID, input.SubmissionID,
		)
		written, writeErr := artifactService.WriteFindingProposal(
			ctx, grant.RunID, proposalID,
			artifacts.Payload{MediaType: proposalMediaType, Data: canonical.proposalBytes},
		)
		if writeErr != nil {
			if errors.Is(writeErr, artifacts.ErrArtifactConflict) {
				existing, replayErr := readReceiptBySubmission(
					ctx, tx, grant.AllocationID, input.InvocationID, input.SubmissionID,
				)
				if replayErr == nil && existing.RequestDigest == canonical.digest {
					result, replayed = existing, true
					return nil
				}
				if replayErr == nil || errors.Is(replayErr, pgx.ErrNoRows) {
					return ErrConflict
				}
				return replayErr
			}
			return writeErr
		}
		proposal := ExactArtifact{
			Ref: written.Ref, Digest: digestBytes(canonical.proposalBytes),
			MediaType: written.MediaType, SizeBytes: written.Size,
		}
		if err := artifactService.PinExact(
			ctx, grant.RunID, mustRunScope(grant.RunID), proposal.Ref,
			artifacts.PinFindingProposal, receiptID+":proposal",
		); err != nil {
			return err
		}
		for index, item := range evidence {
			if err := artifactService.PinExact(
				ctx, grant.RunID, mustRunScope(grant.RunID), item.Ref,
				artifacts.PinFindingEvidence, fmt.Sprintf("%s:evidence:%d", receiptID, index+1),
			); err != nil {
				return err
			}
		}
		origin.RunDeleted = false
		if err := insertReceipt(
			ctx, tx, receiptID, proposalID, canonical, proposal, evidence,
			origin, ownerID, projectID, grant,
		); err != nil {
			return err
		}
		result, err = readReceiptBySubmission(
			ctx, tx, grant.AllocationID, input.InvocationID, input.SubmissionID,
		)
		return err
	})
	if err != nil {
		return Receipt{}, false, err
	}
	return result, replayed, nil
}

func verifyTrustedExecution(
	ctx context.Context,
	tx pgx.Tx,
	grant controlplane.AllocationGrant,
) (Origin, string, *string, error) {
	if strings.TrimSpace(grant.AllocationID) == "" || strings.TrimSpace(grant.RunID) == "" ||
		strings.TrimSpace(grant.StageExecutionID) == "" || strings.TrimSpace(grant.LogicalAgentName) == "" {
		return Origin{}, "", nil, ErrAccessDenied
	}
	var allocationMatches bool
	if err := tx.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1
      FROM stage_allocations AS allocation
      JOIN stage_executions AS execution
        ON execution.stage_execution_id = allocation.stage_execution_id
     WHERE allocation.allocation_id = $1
       AND allocation.stage_execution_id = $2
       AND execution.run_id = $3
       AND allocation.logical_agent_name = $4
       AND allocation.namespace = $5
       AND allocation.runtime_agent_id = $6
       AND allocation.runtime_agent_instance_id = $7
       AND allocation.release_completed_at IS NULL
)`, grant.AllocationID, grant.StageExecutionID, grant.RunID, grant.LogicalAgentName,
		grant.Namespace, grant.RuntimeAgentID, grant.RuntimeInstanceID).Scan(&allocationMatches); err != nil {
		return Origin{}, "", nil, fmt.Errorf("verify finding allocation provenance: %w", err)
	}
	if !allocationMatches {
		return Origin{}, "", nil, ErrAccessDenied
	}
	runs := runstore.NewPostgresStore(tx)
	run, err := runs.GetRun(ctx, grant.RunID)
	if err != nil {
		return Origin{}, "", nil, ErrAccessDenied
	}
	stage, err := runs.GetStageExecution(ctx, grant.StageExecutionID)
	if err != nil || stage.RunID != run.RunID {
		return Origin{}, "", nil, ErrAccessDenied
	}
	resolved, err := workflowconfig.DecodeResolvedStageSnapshot(stage.StageSpecSnapshot)
	if err != nil {
		return Origin{}, "", nil, fmt.Errorf("decode selected Stage snapshot: %w", err)
	}
	binding, ok := resolved.Agents[grant.LogicalAgentName]
	if !ok || binding.Namespace != grant.Namespace || !selectsFindingTool(binding.Template.Toolsets) {
		return Origin{}, "", nil, ErrToolNotSelected
	}
	origin := Origin{
		RunID: run.RunID, StageExecutionID: stage.StageExecutionID,
		AllocationID: grant.AllocationID, LogicalAgentName: grant.LogicalAgentName,
		Workflow: WorkflowOrigin{
			Name: run.WorkflowName, Version: run.WorkflowVersion,
			SchemaVersion: run.WorkflowSchemaVersion, ClosureDigest: digestBytes(run.WorkflowSnapshot),
		},
	}
	origin.Workflow.ConfigurationRef.Name = run.WorkflowName
	origin.Workflow.ConfigurationRef.Version = run.WorkflowVersion
	if run.AuditExecutionID != nil {
		var audit AuditOrigin
		var findingPolicy string
		err := tx.QueryRow(ctx, `
SELECT execution.audit_id, execution.execution_id, execution.role,
       audit.profile_snapshot #>> '{interaction,findingConfirmation}'
  FROM audit_executions AS execution
  JOIN audits AS audit USING (audit_id)
 WHERE execution.execution_id = $1 AND execution.run_id = $2`,
			*run.AuditExecutionID, run.RunID,
		).Scan(&audit.AuditID, &audit.ExecutionID, &audit.Role, &findingPolicy)
		if err != nil || findingPolicy == "disabled" {
			return Origin{}, "", nil, ErrToolNotSelected
		}
		origin.Audit = &audit
	}
	return origin, run.OwnerID, run.ProjectID, nil
}

func selectsFindingTool(selections []contracts.ToolsetSelection) bool {
	for _, selection := range selections {
		if selection.Ref.ToolsetID != "security-findings" || selection.Ref.Version != "1" {
			continue
		}
		for _, tool := range selection.Tools {
			if tool == "finding" {
				return true
			}
		}
	}
	return false
}

func mustRunScope(runID string) artifacts.Scope {
	scope, err := artifacts.RunScope(runID)
	if err != nil {
		panic("validated Run ID produced an invalid Artifact scope")
	}
	return scope
}

func insertReceipt(
	ctx context.Context,
	tx pgx.Tx,
	receiptID, proposalID string,
	canonical canonicalSubmission,
	proposal ExactArtifact,
	evidence []ExactArtifact,
	origin Origin,
	ownerID string,
	projectID *string,
	grant controlplane.AllocationGrant,
) error {
	configurationRef, _ := json.Marshal(origin.Workflow.ConfigurationRef)
	proposalRef, _ := json.Marshal(proposal.Ref)
	evidenceJSON, _ := json.Marshal(evidence)
	var auditExecutionID, auditID, auditRole *string
	if origin.Audit != nil {
		auditExecutionID, auditID, auditRole = &origin.Audit.ExecutionID, &origin.Audit.AuditID, &origin.Audit.Role
	}
	_, err := tx.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id, runtime_instance_id,
    stage_execution_id, logical_agent_name, invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id, audit_execution_id, audit_id, audit_role,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type, proposal_size_bytes, evidence
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22,
    $23, $24, $25, $26, $27
)`,
		receiptID, proposalID, grant.AllocationID, grant.RuntimeAgentID, grant.RuntimeInstanceID,
		grant.StageExecutionID, grant.LogicalAgentName, canonical.request.InvocationID,
		canonical.request.SubmissionID, canonical.request.Proposal.ClientKey, canonical.digest,
		grant.RunID, ownerID, projectID,
		auditExecutionID, auditID, auditRole, origin.Workflow.Name, origin.Workflow.Version,
		origin.Workflow.SchemaVersion, configurationRef, origin.Workflow.ClosureDigest,
		proposalRef, proposal.Digest, proposal.MediaType, proposal.SizeBytes,
		evidenceJSON,
	)
	if err != nil {
		if persistencepostgres.SQLState(err) == "23505" {
			return ErrConflict
		}
		return fmt.Errorf("insert finding proposal receipt: %w", err)
	}
	if _, err := tx.Exec(ctx, `
INSERT INTO finding_proposal_retention (receipt_id) VALUES ($1)`, receiptID); err != nil {
		return fmt.Errorf("insert finding proposal retention: %w", err)
	}
	return nil
}

func receiptResponse(receipt Receipt, replayed bool) SubmissionResponse {
	return SubmissionResponse{
		APIVersion: APIVersion, ProposalID: receipt.ProposalID, ReceiptID: receipt.ReceiptID,
		Proposal: receipt.Proposal, Replayed: replayed,
	}
}

func nowUTC() time.Time { return time.Now().UTC() }
