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
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
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
func (s *Service) Submit(ctx context.Context, grant controlplane.AllocationGrant, input Submission) (Receipt, bool, error) {
	return s.submit(ctx, intakeExecution{RunID: grant.RunID, StageID: grant.StageExecutionID, Worker: &grant}, input)
}

func (s *Service) submit(ctx context.Context, execution intakeExecution, input Submission) (Receipt, bool, error) {
	canonical, err := canonicalize(input)
	if err != nil {
		return Receipt{}, false, err
	}
	if execution.Worker == nil {
		canonical.digest = execution.RequestDigest
	}
	// Retain the slot while both the candidate blob and receipt are published.
	ctx, releaseTransfer, err := s.storage.AcquireTransfer(ctx)
	if err != nil {
		return Receipt{}, false, err
	}
	defer releaseTransfer()
	// Replay precedes derived evidence reads: source deletion or a later
	// adapter version must not prevent reconciliation of a committed call.
	if previous, found, replayErr := s.findExecutionReplay(ctx, execution, input); replayErr != nil {
		return Receipt{}, false, replayErr
	} else if found {
		return previous, true, nil
	}
	// Reject projectless calls before staging any payload. The transaction below
	// repeats this check under the execution and Project publication fences.
	var activeProject bool
	if err := s.pool.QueryRow(ctx, `SELECT coalesce(project.owner_id=run.owner_id AND project.kind='project' AND project.lifecycle_state='active',false) FROM workflow_runs AS run LEFT JOIN projects AS project ON project.project_id=run.project_id WHERE run.run_id=$1`, execution.RunID).Scan(&activeProject); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return Receipt{}, false, ErrAccessDenied
		}
		return Receipt{}, false, err
	}
	if !activeProject {
		return Receipt{}, false, ErrProjectRequired
	}
	anchors := auditdomain.UnavailableFindingAnchors("not_provided")
	fingerprint := auditdomain.UnavailableFindingFingerprint("not_provided")
	if execution.Worker != nil && canonical.request.Proposal.SeveritySuggestion != "informational" {
		readStage := s.lazyPreparationStage(ctx, *execution.Worker)
		anchors, err = s.prepareAnchors(ctx, *execution.Worker, canonical.request, readStage)
		if err != nil {
			return Receipt{}, false, err
		}
		fingerprint, err = s.prepareFingerprint(ctx, *execution.Worker, canonical.request, readStage)
		if err != nil {
			return Receipt{}, false, err
		}
	} else if execution.Fingerprint != nil {
		fingerprint = *execution.Fingerprint
	}
	canonical.anchorProjection, err = auditdomain.EncodeFindingAnchors(anchors)
	if err != nil {
		return Receipt{}, false, err
	}
	fingerprintJSON, err := auditdomain.EncodeFindingFingerprint(fingerprint)
	if err != nil {
		return Receipt{}, false, err
	}
	receiptID, proposalID := execution.receiptID(input), execution.proposalID(input)
	var httpPayload *artifacts.Payload
	if canonical.request.HTTPEvidence != nil {
		data, encodeErr := auditdomain.EncodeHTTPExchangeArtifact(*canonical.request.HTTPEvidence, execution.RunID, input.HTTPEvidenceInvocationID)
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
			"finding-proposal-submission", receiptID,
		)
		if err := persistencepostgres.LockAdvisory(ctx, tx, lockKey); err != nil {
			return fmt.Errorf("lock finding proposal submission: %w", err)
		}
		existing, readErr := s.readExecutionReceipt(ctx, tx, execution, input)
		if readErr == nil {
			if existing.RequestDigest != canonical.digest ||
				!execution.matchesOrigin(existing.Origin) {
				return ErrConflict
			}
			result, replayed = existing, true
			return nil
		}
		if !errors.Is(readErr, pgx.ErrNoRows) {
			return readErr
		}

		origin, ownerID, projectID, err := s.verifyIntakeExecution(ctx, tx, execution, canonical.request.ItemKey, canonical.request.Incidental)
		if canonical.request.Incidental && origin.Audit != nil {
			origin.Audit.Hypothesis = nil
		}
		if err != nil {
			return err
		}
		if projectID == nil {
			return ErrProjectRequired
		}
		var activeProject bool
		if err := tx.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM projects WHERE project_id=$1 AND owner_id=$2 AND kind='project' AND lifecycle_state='active')", *projectID, ownerID).Scan(&activeProject); err != nil {
			return err
		}
		if !activeProject {
			return ErrProjectRequired
		}
		origin.InvocationID = canonical.request.InvocationID
		// The quota counts every submission of the Run, so distinct
		// submissions must also serialize their count and insert.
		runLockKey := persistencepostgres.NewAdvisoryKey("run-proposal-quota", execution.RunID)
		if err := persistencepostgres.LockAdvisory(ctx, tx, runLockKey); err != nil {
			return fmt.Errorf("lock Run finding proposal quota: %w", err)
		}
		var count int
		if err := tx.QueryRow(ctx, `
SELECT count(*) FROM finding_proposal_receipts WHERE run_id = $1`, execution.RunID).Scan(&count); err != nil {
			return fmt.Errorf("count Run finding proposals: %w", err)
		}
		if count >= maxProposalsPerRun {
			return fmt.Errorf("%w: Run proposal quota exceeded", ErrInvalid)
		}

		artifactService := artifacts.NewService(s.storage.Repository(tx))
		runArtifacts, err := artifactService.Run(execution.RunID)
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
			if ref.Namespace == artifactpolicy.HTTPExchangeNamespace &&
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
			written, writeErr := artifactService.CreateFrozen(ctx, mustRunScope(execution.RunID),
				contracts.ArtifactRef{Namespace: artifactpolicy.HTTPExchangeNamespace, Name: auditdomain.HTTPExchangeArtifactName(input.HTTPEvidenceInvocationID, canonical.request.HTTPEvidence.Source, canonical.request.HTTPEvidence.RequestID)}, *httpPayload)
			if writeErr != nil {
				if errors.Is(writeErr, artifacts.ErrAuditArtifactCollision) {
					return ErrConflict
				}
				return writeErr
			}
			evidence = append(evidence, ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(httpPayload.Data),
				MediaType: written.MediaType, SizeBytes: written.Size})
		}
		if canonical.request.Proposal.SeveritySuggestion == "informational" {
			derived, deriveErr := auditdomain.FactEvidenceFingerprint(evidence)
			if deriveErr != nil {
				return deriveErr
			}
			fingerprintJSON, err = auditdomain.EncodeFindingFingerprint(derived)
			if err != nil {
				return err
			}
		}
		// finding-proposals is reserved for this path: generic Run writes can
		// neither create nor advance it, while the exact revision stays
		// readable in its RunScope (ART-R114).
		written, writeErr := artifactService.Create(
			ctx, mustRunScope(execution.RunID),
			contracts.ArtifactRef{Namespace: artifactpolicy.FindingProposalNamespace, Name: proposalID},
			preparedPayload,
		)
		if writeErr != nil {
			if errors.Is(writeErr, artifacts.ErrArtifactConflict) {
				existing, replayErr := s.readExecutionReceipt(ctx, tx, execution, input)
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
			if err := artifactService.RequireExact(ctx, mustRunScope(execution.RunID), item.Ref); err != nil {
				return err
			}
		}
		origin.RunDeleted = false
		if len(evidence) == 0 {
			return ErrEvidenceRequired
		}
		if err := hookstore.NewFindingEventsTx(tx, s.storage).LockFindingCorrelationFence(ctx, *projectID); err != nil {
			return err
		}
		retained, retainErr := retainFindingArtifact(ctx, artifactService, execution.RunID, *projectID,
			artifactpolicy.FindingMaterialsNamespace, deterministicID("proposal", receiptID), proposal)
		if retainErr != nil {
			return retainErr
		}
		projectProposal := &retained
		if origin.Audit != nil && origin.Audit.Hypothesis != nil {
			task, err := querystore.New(tx, s.storage).GetTask(ctx, ownerID, origin.Audit.AuditID, origin.Audit.TaskID)
			if err != nil {
				return err
			}
			if task.Hypothesis == nil || task.Hypothesis.ID != origin.Audit.Hypothesis.ID || task.Hypothesis.Revision != origin.Audit.Hypothesis.Revision {
				return ErrInvalid
			}
			data, err := strictjson.Canonical(task.Hypothesis)
			if err != nil {
				return err
			}
			scope, err := artifacts.ProjectScope(*projectID)
			if err != nil {
				return err
			}
			written, err := artifactService.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: artifactpolicy.FindingMaterialsNamespace, Name: deterministicID("hypothesis", receiptID)}, artifacts.Payload{MediaType: "application/json", Data: data})
			if err != nil {
				return err
			}
			origin.Audit.HypothesisSnapshot = &ExactArtifact{Ref: written.Ref, Digest: contentdigest.Bytes(data), MediaType: written.MediaType, SizeBytes: written.Size}
		}

		projectEvidence := make([]ExactArtifact, 0, len(evidence))
		for index, source := range evidence {
			retained, retainErr := retainFindingArtifact(ctx, artifactService, execution.RunID, *projectID,
				artifactpolicy.FindingMaterialsNamespace, deterministicID("evidence", receiptID, fmt.Sprint(index)), source)
			if retainErr != nil {
				return retainErr
			}
			projectEvidence = append(projectEvidence, retained)
		}
		if err := insertReceipt(
			ctx, tx, receiptID, proposalID, canonical, fingerprintJSON, proposal, evidence,
			origin, ownerID, projectID, execution.Worker,
			projectProposal, projectEvidence, "",
		); err != nil {
			return err
		}
		if _, err := hookstore.NewFindingEventsTx(tx, s.storage).AdmitProjectFinding(ctx, *projectID, receiptID,
			deterministicID("finding", receiptID)); err != nil {
			return err
		}
		result, err = s.readExecutionReceipt(ctx, tx, execution, input)
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
	selectedItem *string,
	incidental bool,
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
		key := ""
		if selectedItem != nil {
			key = *selectedItem
		}
		if incidental && selectedItem == nil {
			auditID, err := querystore.New(tx, storage).ExecutionAuditOrigin(ctx, *run.AuditExecutionID, run.RunID)
			if errors.Is(err, auditstore.ErrNotFound) || errors.Is(err, auditstore.ErrInvalid) {
				return Origin{}, "", nil, ErrAccessDenied
			}
			if err != nil {
				return Origin{}, "", nil, err
			}
			origin.Audit = &AuditOrigin{AuditID: auditID, ExecutionID: *run.AuditExecutionID}
		} else {
			task, err := querystore.New(tx, storage).ExecutionTaskOrigin(ctx, *run.AuditExecutionID, run.RunID, key)
			if errors.Is(err, auditstore.ErrNotFound) || errors.Is(err, auditstore.ErrInvalid) {
				return Origin{}, "", nil, ErrAccessDenied
			}
			if err != nil {
				return Origin{}, "", nil, fmt.Errorf("read finding Task provenance: %w", err)
			}
			origin.Audit = &AuditOrigin{AuditID: task.AuditID, ExecutionID: *run.AuditExecutionID, TaskID: task.TaskID, Hypothesis: task.Hypothesis}
		}
	} else if selectedItem != nil {
		return Origin{}, "", nil, ErrInvalid
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
	worker *controlplane.AllocationGrant,
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
	var auditExecutionID, auditID, auditTaskID *string
	var hypothesis, hypothesisSnapshot []byte
	if origin.Audit != nil {
		auditExecutionID, auditID = &origin.Audit.ExecutionID, &origin.Audit.AuditID
		if origin.Audit.TaskID != "" {
			auditTaskID = &origin.Audit.TaskID
		}
		if origin.Audit.Hypothesis != nil {
			hypothesis, _ = json.Marshal(origin.Audit.Hypothesis)
			if origin.Audit.HypothesisSnapshot != nil {
				hypothesisSnapshot, _ = json.Marshal(origin.Audit.HypothesisSnapshot)
			}
		}
	}
	stageID, runID := origin.StageExecutionID, origin.RunID
	var allocationID, runtimeAgentID, runtimeInstanceID, logicalAgentName *string
	if worker != nil {
		stageID, runID = worker.StageExecutionID, worker.RunID
		allocationID, runtimeAgentID = &worker.AllocationID, &worker.RuntimeAgentID
		runtimeInstanceID, logicalAgentName = &worker.RuntimeInstanceID, &worker.LogicalAgentName
	}
	var scanOrigin []byte
	if origin.Scan != nil {
		scanOrigin, _ = json.Marshal(origin.Scan)
	}
	_, err := tx.Exec(ctx, insertReceiptSQL,
		receiptID, proposalID, allocationID, runtimeAgentID, runtimeInstanceID,
		stageID, logicalAgentName, canonical.request.InvocationID,
		canonical.request.SubmissionID, canonical.request.Proposal.ClientKey, canonical.digest,
		runID, ownerID, projectID,
		auditExecutionID, auditID, auditTaskID, origin.Workflow.Name, origin.Workflow.Version,
		origin.Workflow.SchemaVersion, configurationRef, origin.Workflow.ClosureDigest,
		proposalRef, proposal.Digest, proposal.MediaType, proposal.SizeBytes,
		evidenceJSON, string(canonical.reportProjection), string(fingerprintJSON), string(canonical.anchorProjection),
		string(canonical.checksProjection),
		true, publicationError, projectProposalJSON, projectEvidenceJSON, hypothesis, hypothesisSnapshot, origin.Operation, scanOrigin,
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
