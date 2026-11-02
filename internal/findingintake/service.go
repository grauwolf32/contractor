package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
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
	storage *artifactpostgres.Storage

	pool         *pgxpool.Pool
	afterPrepare func(context.Context) // deterministic transfer-boundary integration test seam
}

func New(pool *pgxpool.Pool, storage *artifactpostgres.Storage) (*Service, error) {
	if storage == nil || pool == nil {
		return nil, errors.New("finding intake PostgreSQL pool is required")
	}
	return &Service{storage: storage, pool: pool}, nil
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
	// Retain the slot while both the candidate blob and receipt are published.
	ctx, releaseTransfer, err := s.storage.AcquireTransfer(ctx)
	if err != nil {
		return Receipt{}, false, err
	}
	defer releaseTransfer()
	// Replay precedes derived evidence reads: source deletion or a later
	// adapter version must not prevent reconciliation of a committed call.
	if previous, found, replayErr := s.FindReplay(ctx, grant, input); replayErr != nil {
		return Receipt{}, false, replayErr
	} else if found {
		return previous, true, nil
	}
	readStage := s.lazyPreparationStage(ctx, grant)
	anchors, err := s.prepareAnchors(ctx, grant, canonical.request, readStage)
	if err != nil {
		return Receipt{}, false, err
	}
	canonical.anchorProjection, err = auditdomain.EncodeFindingAnchors(anchors)
	if err != nil {
		return Receipt{}, false, err
	}
	fingerprint, err := s.prepareFingerprint(ctx, grant, canonical.request, readStage)
	if err != nil {
		return Receipt{}, false, err
	}
	fingerprintJSON, err := auditdomain.EncodeFindingFingerprint(fingerprint)
	if err != nil {
		return Receipt{}, false, err
	}
	receiptID := deterministicID("receipt", grant.AllocationID, input.InvocationID, input.SubmissionID)
	proposalID := deterministicID("proposal", grant.AllocationID, input.InvocationID, input.SubmissionID)
	var httpPayload *artifacts.Payload
	if canonical.request.HTTPEvidence != nil {
		data, encodeErr := auditdomain.EncodeHTTPExchangeArtifact(*canonical.request.HTTPEvidence, proposalID, receiptID)
		if encodeErr != nil || len(data) > artifacts.MaxPayloadSize {
			return Receipt{}, false, fmt.Errorf("%w: HTTP evidence payload", ErrInvalid)
		}
		prepared, prepareErr := s.storage.PreparePayload(ctx, s.pool, artifacts.Payload{MediaType: auditdomain.HTTPExchangeMediaType, Data: data})
		if prepareErr != nil {
			return Receipt{}, false, prepareErr
		}
		httpPayload = &prepared
	}
	preparedPayload, err := s.storage.PreparePayload(ctx, s.pool, artifacts.Payload{MediaType: proposalMediaType, Data: canonical.proposalBytes})
	if err != nil {
		return Receipt{}, false, err
	}
	if s.afterPrepare != nil {
		s.afterPrepare(ctx)
	}
	var result Receipt
	var replayed bool
	err = persistencepostgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		result, replayed = Receipt{}, false
		// Serialize the initially absent idempotency row as well as ordinary
		// replays. A transaction advisory lock avoids a race in which two
		// byte-identical requests both advance the proposal Artifact binding
		// before either receipt becomes visible.
		lockKey := persistencepostgres.NewAdvisoryKey(
			"finding-proposal-submission", grant.AllocationID, input.InvocationID, input.SubmissionID,
		)
		if err := persistencepostgres.LockAdvisory(ctx, tx, lockKey); err != nil {
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

		origin, ownerID, projectID, err := verifyTrustedExecution(ctx, tx, grant, s.storage)
		if err != nil {
			return err
		}
		origin.InvocationID = canonical.request.InvocationID
		// The quota counts every submission of the Run, so distinct
		// submissions must also serialize their count and insert.
		runLockKey := persistencepostgres.NewAdvisoryKey("run-proposal-quota", grant.RunID)
		if err := persistencepostgres.LockAdvisory(ctx, tx, runLockKey); err != nil {
			return fmt.Errorf("lock Run finding proposal quota: %w", err)
		}
		var count int
		if err := tx.QueryRow(ctx, `
SELECT count(*) FROM finding_proposal_receipts WHERE run_id = $1`, grant.RunID).Scan(&count); err != nil {
			return fmt.Errorf("count Run finding proposals: %w", err)
		}
		if count >= maxProposalsPerRun {
			return fmt.Errorf("%w: Run proposal quota exceeded", ErrInvalid)
		}

		artifactService := artifacts.NewService(s.storage.Repository(tx))
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
			if ref.Namespace == artifactpolicy.FindingProposalNamespace &&
				(metadata.MediaType != auditdomain.HTTPExchangeMediaType || !metadata.Frozen) {
				return ErrAccessDenied
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

		if httpPayload != nil {
			if evidenceBytes+int64(len(httpPayload.Data)) > maxProposalEvidenceBytes {
				return fmt.Errorf("%w: proposal evidence quota exceeded", ErrInvalid)
			}
			written, writeErr := artifactService.CreateFrozen(ctx, mustRunScope(grant.RunID),
				contracts.ArtifactRef{Namespace: artifactpolicy.FindingProposalNamespace, Name: "http.exchange." + proposalID}, *httpPayload)
			if writeErr != nil {
				if errors.Is(writeErr, artifacts.ErrAuditArtifactCollision) {
					return ErrConflict
				}
				return writeErr
			}
			evidence = append(evidence, ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(httpPayload.Data),
				MediaType: written.MediaType, SizeBytes: written.Size})
		}
		// finding-proposals is reserved for this path: generic Run writes can
		// neither create nor advance it, while the exact revision stays
		// readable in its RunScope (ART-R114).
		written, writeErr := artifactService.Create(
			ctx, mustRunScope(grant.RunID),
			contracts.ArtifactRef{Namespace: artifactpolicy.FindingProposalNamespace, Name: proposalID},
			preparedPayload,
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
			Ref: written.Ref, Digest: auditdomain.DigestBytes(canonical.proposalBytes),
			MediaType: written.MediaType, SizeBytes: written.Size,
		}
		for _, item := range evidence {
			if err := artifactService.RequireExact(ctx, mustRunScope(grant.RunID), item.Ref); err != nil {
				return err
			}
		}
		origin.RunDeleted = false
		var projectProposal *ExactArtifact
		var projectEvidence []ExactArtifact
		publicationError := ""
		var publicationProject bool
		if input.PublishProblem && projectID != nil {
			if err := tx.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM projects WHERE project_id=$1 AND owner_id=$2 AND kind='project' AND lifecycle_state='active')", *projectID, ownerID).Scan(&publicationProject); err != nil {
				return err
			}
		}
		if input.PublishProblem {
			switch {
			case projectID == nil || !publicationProject:
				publicationError = "problem_project_required"
			case len(evidence) == 0:
				publicationError = "problem_evidence_required"
			default:
				if err := hookstore.NewFindingEventsTx(tx, s.storage).LockFindingCorrelationFence(ctx, *projectID); err != nil {
					return err
				}
				retained, retainErr := retainFindingArtifact(ctx, artifactService, grant.RunID, *projectID,
					artifactpolicy.FindingMaterialsNamespace, deterministicID("proposal", receiptID), proposal)
				if retainErr != nil {
					return retainErr
				}
				projectProposal = &retained
				projectEvidence = make([]ExactArtifact, 0, len(evidence))
				for index, source := range evidence {
					retained, retainErr := retainFindingArtifact(ctx, artifactService, grant.RunID, *projectID,
						artifactpolicy.FindingMaterialsNamespace, deterministicID("evidence", receiptID, fmt.Sprint(index)), source)
					if retainErr != nil {
						return retainErr
					}
					projectEvidence = append(projectEvidence, retained)
				}
			}
		}
		if err := insertReceipt(
			ctx, tx, receiptID, proposalID, canonical, fingerprintJSON, proposal, evidence,
			origin, ownerID, projectID, grant,
			projectProposal, projectEvidence, publicationError,
		); err != nil {
			return err
		}
		if projectProposal != nil {
			if _, err := hookstore.NewFindingEventsTx(tx, s.storage).AdmitProjectFinding(ctx, *projectID, receiptID,
				deterministicID("finding", receiptID)); err != nil {
				return err
			}
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

// preparationStage reads the grant's StageExecution for the optional anchor
// and fingerprint derivations before the intake transaction.
type preparationStage func() (runstore.StageExecution, error)

// lazyPreparationStage returns a loader scoped to one Submit call: the first
// derivation that needs the Stage reads it, and the other reuses that
// snapshot. A derivation with no Stage input never reads it. The snapshot
// only selects optional evidence inputs; verifyTrustedExecution re-reads the
// Stage inside the intake transaction for provenance and authorization.
func (s *Service) lazyPreparationStage(ctx context.Context, grant controlplane.AllocationGrant) preparationStage {
	return sync.OnceValues(func() (runstore.StageExecution, error) {
		stage, err := runstore.NewPostgresStore(s.pool, s.storage).GetStageExecution(ctx, grant.StageExecutionID)
		if err != nil {
			return runstore.StageExecution{}, err
		}
		if stage.RunID != grant.RunID {
			return runstore.StageExecution{}, ErrAccessDenied
		}
		return stage, nil
	})
}

func verifyTrustedExecution(
	ctx context.Context,
	tx pgx.Tx,
	grant controlplane.AllocationGrant,
	storage *artifactpostgres.Storage,
) (Origin, string, *string, error) {
	if strings.TrimSpace(grant.AllocationID) == "" || strings.TrimSpace(grant.RunID) == "" ||
		strings.TrimSpace(grant.StageExecutionID) == "" || strings.TrimSpace(grant.LogicalAgentName) == "" {
		return Origin{}, "", nil, ErrAccessDenied
	}
	var allocationMatches bool
	if err := tx.QueryRow(ctx, verifyTrustedExecutionSQL, grant.AllocationID, grant.StageExecutionID, grant.RunID, grant.LogicalAgentName,
		grant.Namespace, grant.RuntimeAgentID, grant.RuntimeInstanceID).Scan(&allocationMatches); err != nil {
		return Origin{}, "", nil, fmt.Errorf("verify finding allocation provenance: %w", err)
	}
	if !allocationMatches {
		return Origin{}, "", nil, ErrAccessDenied
	}
	// Only a confirmed absence denies the grant. A storage failure keeps its
	// cause so the submission stays retryable instead of becoming a denial.
	runs := runstore.NewPostgresStore(tx, storage)
	run, err := runs.GetRun(ctx, grant.RunID)
	if errors.Is(err, runstore.ErrNotFound) {
		return Origin{}, "", nil, ErrAccessDenied
	}
	if err != nil {
		return Origin{}, "", nil, fmt.Errorf("read finding Run provenance: %w", err)
	}
	stage, err := runs.GetStageExecution(ctx, grant.StageExecutionID)
	if errors.Is(err, runstore.ErrNotFound) {
		return Origin{}, "", nil, ErrAccessDenied
	}
	if err != nil {
		return Origin{}, "", nil, fmt.Errorf("read finding StageExecution provenance: %w", err)
	}
	if stage.RunID != run.RunID {
		return Origin{}, "", nil, ErrAccessDenied
	}
	stageSpec, err := resolved.DecodeResolvedStageSnapshot(stage.StageSpecSnapshot)
	if err != nil {
		return Origin{}, "", nil, fmt.Errorf("decode selected Stage snapshot: %w", err)
	}
	binding, ok := stageSpec.Agents[grant.LogicalAgentName]
	if !ok || binding.Namespace != grant.Namespace || !selectsFindingTool(binding.Template.Toolsets) {
		return Origin{}, "", nil, ErrToolNotSelected
	}
	origin := Origin{
		RunID: run.RunID, StageExecutionID: stage.StageExecutionID,
		AllocationID: grant.AllocationID, LogicalAgentName: grant.LogicalAgentName,
		Workflow: WorkflowOrigin{
			Name: run.WorkflowName, Version: run.WorkflowVersion,
			SchemaVersion: run.WorkflowSchemaVersion, ClosureDigest: auditdomain.DigestBytes(run.WorkflowSnapshot),
		},
	}
	origin.Workflow.ConfigurationRef.Name = run.WorkflowName
	origin.Workflow.ConfigurationRef.Version = run.WorkflowVersion
	if run.AuditExecutionID != nil {
		var audit AuditOrigin
		err := tx.QueryRow(ctx, `
SELECT execution.audit_id, execution.execution_id, execution.role
  FROM audit_executions AS execution
  JOIN audits AS audit USING (audit_id)
 WHERE execution.execution_id = $1 AND execution.run_id = $2`,
			*run.AuditExecutionID, run.RunID,
		).Scan(&audit.AuditID, &audit.ExecutionID, &audit.Role)
		if errors.Is(err, pgx.ErrNoRows) {
			return Origin{}, "", nil, ErrToolNotSelected
		}
		if err != nil {
			return Origin{}, "", nil, fmt.Errorf("read finding Audit provenance: %w", err)
		}
		origin.Audit = &audit
	}
	return origin, run.OwnerID, run.ProjectID, nil
}

func selectsFindingTool(selections []contracts.ToolsetSelection) bool {
	return slices.ContainsFunc(selections, resolved.SelectsFindingProposalTool)
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
	fingerprintJSON []byte,
	proposal ExactArtifact,
	evidence []ExactArtifact,
	origin Origin,
	ownerID string,
	projectID *string,
	grant controlplane.AllocationGrant,
	projectProposal *ExactArtifact,
	projectEvidence []ExactArtifact,
	publicationError string,
) error {
	configurationRef, _ := json.Marshal(origin.Workflow.ConfigurationRef)
	proposalRef, _ := json.Marshal(proposal.Ref)
	evidenceJSON, _ := json.Marshal(evidence)
	var projectProposalJSON, projectEvidenceJSON []byte
	if projectProposal != nil {
		projectProposalJSON, _ = json.Marshal(projectProposal)
		projectEvidenceJSON, _ = json.Marshal(projectEvidence)
	}
	var auditExecutionID, auditID, auditRole *string
	if origin.Audit != nil {
		auditExecutionID, auditID, auditRole = &origin.Audit.ExecutionID, &origin.Audit.AuditID, &origin.Audit.Role
	}
	_, err := tx.Exec(ctx, insertReceiptSQL,
		receiptID, proposalID, grant.AllocationID, grant.RuntimeAgentID, grant.RuntimeInstanceID,
		grant.StageExecutionID, grant.LogicalAgentName, canonical.request.InvocationID,
		canonical.request.SubmissionID, canonical.request.Proposal.ClientKey, canonical.digest,
		grant.RunID, ownerID, projectID,
		auditExecutionID, auditID, auditRole, origin.Workflow.Name, origin.Workflow.Version,
		origin.Workflow.SchemaVersion, configurationRef, origin.Workflow.ClosureDigest,
		proposalRef, proposal.Digest, proposal.MediaType, proposal.SizeBytes,
		evidenceJSON, string(canonical.reportProjection), string(fingerprintJSON), string(canonical.anchorProjection),
		string(canonical.checksProjection),
		canonical.request.PublishProblem, publicationError, projectProposalJSON, projectEvidenceJSON,
	)
	if err != nil {
		switch persistencepostgres.Classify(err) {
		case persistencepostgres.ClassDuplicate:
			return ErrConflict
		case persistencepostgres.ClassInvalid:
			// A named CHECK, such as the comparison projection's size bound
			// on its stored jsonb text, refused the submitted proposal; the
			// same submission is refused again (PER-R13).
			return fmt.Errorf("insert finding proposal receipt: %w: %w", ErrInvalid, err)
		}
		// The receipt holds its proposal and evidence strictly: a revision
		// that history retention removed after this transaction proved it is
		// the same outcome as evidence that was never there (FAIL-DEL-7).
		if artifactpostgres.IsExactRevisionMissing(err) {
			return ErrAccessDenied
		}
		return fmt.Errorf("insert finding proposal receipt: %w", err)
	}
	return nil
}
