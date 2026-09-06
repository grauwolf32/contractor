package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type querier interface {
	QueryRow(context.Context, string, ...any) pgx.Row
	Query(context.Context, string, ...any) (pgx.Rows, error)
}

const receiptProjection = `
receipt.receipt_id, receipt.proposal_id, receipt.request_digest,
receipt.client_key,
receipt.run_id, receipt.stage_execution_id, receipt.allocation_id,
receipt.logical_agent_name, receipt.invocation_id,
receipt.workflow_name, receipt.workflow_version, receipt.workflow_schema_version,
receipt.workflow_configuration_ref, receipt.workflow_closure_digest,
receipt.audit_id, receipt.audit_execution_id, receipt.audit_role,
receipt.proposal_ref, receipt.proposal_digest, receipt.proposal_media_type,
receipt.proposal_size_bytes, receipt.evidence,
retention.state, retention.source_run_deleted_at, receipt.created_at`

func readReceiptBySubmission(
	ctx context.Context,
	db querier,
	allocationID, invocationID, submissionID string,
) (Receipt, error) {
	return scanReceipt(ctx, db, db.QueryRow(ctx, `
SELECT `+receiptProjection+`
  FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_retention AS retention USING (receipt_id)
 WHERE receipt.allocation_id = $1 AND receipt.invocation_id = $2
   AND receipt.submission_id = $3`, allocationID, invocationID, submissionID))
}

func readReceiptByID(ctx context.Context, db querier, receiptID string) (Receipt, error) {
	return scanReceipt(ctx, db, db.QueryRow(ctx, `
SELECT `+receiptProjection+`
  FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_retention AS retention USING (receipt_id)
 WHERE receipt.receipt_id = $1`, receiptID))
}

func scanReceipt(ctx context.Context, db querier, row pgx.Row) (Receipt, error) {
	var result Receipt
	var configurationRef, proposalRef, evidenceJSON []byte
	var auditID, auditExecutionID, auditRole *string
	var sourceDeletedAt *time.Time
	err := row.Scan(
		&result.ReceiptID, &result.ProposalID, &result.RequestDigest,
		&result.ClientKey,
		&result.Origin.RunID, &result.Origin.StageExecutionID, &result.Origin.AllocationID,
		&result.Origin.LogicalAgentName, &result.Origin.InvocationID,
		&result.Origin.Workflow.Name, &result.Origin.Workflow.Version,
		&result.Origin.Workflow.SchemaVersion, &configurationRef,
		&result.Origin.Workflow.ClosureDigest,
		&auditID, &auditExecutionID, &auditRole,
		&proposalRef, &result.Proposal.Digest, &result.Proposal.MediaType,
		&result.Proposal.SizeBytes, &evidenceJSON,
		&result.Retention, &sourceDeletedAt, &result.CreatedAt,
	)
	if err != nil {
		return Receipt{}, err
	}
	if err := json.Unmarshal(configurationRef, &result.Origin.Workflow.ConfigurationRef); err != nil ||
		json.Unmarshal(proposalRef, &result.Proposal.Ref) != nil ||
		json.Unmarshal(evidenceJSON, &result.Evidence) != nil {
		return Receipt{}, errors.New("decode stored finding proposal receipt")
	}
	if !validIdentity(result.ClientKey) || result.Proposal.Ref.ValidateExact() != nil ||
		result.Proposal.MediaType != proposalMediaType ||
		result.Proposal.Digest == "" || result.Proposal.SizeBytes < 0 {
		return Receipt{}, errors.New("validate stored finding proposal receipt")
	}
	result.Origin.RunDeleted = sourceDeletedAt != nil
	if auditID != nil {
		if auditExecutionID == nil || auditRole == nil {
			return Receipt{}, errors.New("validate stored finding Audit origin")
		}
		result.Origin.Audit = &AuditOrigin{
			AuditID: *auditID, ExecutionID: *auditExecutionID, Role: *auditRole,
		}
	}
	holds, err := readAuditHolds(ctx, db, result.ReceiptID)
	if err != nil {
		return Receipt{}, err
	}
	result.AuditHolds = holds
	return result, nil
}

func readAuditHolds(ctx context.Context, db querier, receiptID string) ([]AuditHold, error) {
	rows, err := db.Query(ctx, `
SELECT audit_id, project_id, proposal_ref, evidence, created_at
  FROM finding_proposal_audit_holds
 WHERE receipt_id = $1
 ORDER BY created_at, audit_id`, receiptID)
	if err != nil {
		return nil, fmt.Errorf("list finding proposal Audit holds: %w", err)
	}
	defer rows.Close()
	result := make([]AuditHold, 0)
	for rows.Next() {
		var hold AuditHold
		var proposal, evidence []byte
		if err := rows.Scan(&hold.AuditID, &hold.ProjectID, &proposal, &evidence, &hold.CreatedAt); err != nil {
			return nil, fmt.Errorf("scan finding proposal Audit hold: %w", err)
		}
		if json.Unmarshal(proposal, &hold.Proposal) != nil || json.Unmarshal(evidence, &hold.Evidence) != nil {
			return nil, errors.New("decode finding proposal Audit hold")
		}
		result = append(result, hold)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate finding proposal Audit holds: %w", err)
	}
	return result, nil
}

func (s *Service) ListRun(
	ctx context.Context,
	ownerID, runID string,
	query ListQuery,
) ([]Receipt, error) {
	if ownerID == "" || runID == "" || !validListQuery(query) {
		return nil, ErrInvalid
	}
	ids, err := listReceiptIDs(ctx, s.pool, `
SELECT receipt.receipt_id
  FROM finding_proposal_receipts AS receipt
  JOIN workflow_runs AS run ON run.run_id = receipt.run_id
 WHERE receipt.owner_id = $1 AND receipt.run_id = $2 AND run.owner_id = $1
   AND ($3::timestamptz IS NULL OR (receipt.created_at, receipt.receipt_id) > ($3, $4))
 ORDER BY receipt.created_at, receipt.receipt_id
 LIMIT $5`, ownerID, runID, query.AfterCreatedAt, query.AfterReceiptID, query.Limit)
	if err != nil {
		return nil, err
	}
	result, err := readReceipts(ctx, s.pool, ids)
	if err != nil {
		return nil, err
	}
	return s.hydrateReceiptDocuments(ctx, result)
}

func (s *Service) ListAuditInbox(
	ctx context.Context,
	ownerID, auditID string,
	query ListQuery,
) ([]Receipt, error) {
	if ownerID == "" || auditID == "" || !validListQuery(query) {
		return nil, ErrInvalid
	}
	ids, err := listReceiptIDs(ctx, s.pool, `
SELECT receipt.receipt_id
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.owner_id = $1
   AND (receipt.audit_id = $2 OR EXISTS (
       SELECT 1 FROM finding_proposal_audit_holds AS hold
        JOIN audits AS audit ON audit.audit_id = hold.audit_id
       WHERE hold.receipt_id = receipt.receipt_id
         AND hold.audit_id = $2 AND audit.owner_id = $1
   ))
   AND ($3::timestamptz IS NULL OR (receipt.created_at, receipt.receipt_id) > ($3, $4))
 ORDER BY receipt.created_at, receipt.receipt_id
 LIMIT $5`, ownerID, auditID, query.AfterCreatedAt, query.AfterReceiptID, query.Limit)
	if err != nil {
		return nil, err
	}
	result, err := readReceipts(ctx, s.pool, ids)
	if err != nil {
		return nil, err
	}
	return s.hydrateReceiptDocuments(ctx, result)
}

// GetAuditReceipt returns one exact proposal only when the authenticated owner
// can reach it through the requested Audit. It deliberately uses the same
// source/Audit-hold hydration path as inbox listing, so deleted source Runs do
// not weaken integrity checks or provenance.
func (s *Service) GetAuditReceipt(
	ctx context.Context,
	ownerID, auditID, receiptID string,
) (Receipt, error) {
	if ownerID == "" || auditID == "" || receiptID == "" {
		return Receipt{}, ErrInvalid
	}
	var admittedID string
	err := s.pool.QueryRow(ctx, `
SELECT receipt.receipt_id
  FROM finding_proposal_receipts AS receipt
  JOIN audits AS audit ON audit.audit_id = $2 AND audit.owner_id = $1
 WHERE receipt.receipt_id = $3 AND receipt.owner_id = $1
   AND (receipt.audit_id = $2 OR EXISTS (
       SELECT 1 FROM finding_proposal_audit_holds AS hold
        WHERE hold.receipt_id = receipt.receipt_id AND hold.audit_id = $2
   ))`, ownerID, auditID, receiptID).Scan(&admittedID)
	if errors.Is(err, pgx.ErrNoRows) {
		return Receipt{}, ErrNotFound
	}
	if err != nil {
		return Receipt{}, fmt.Errorf("authorize Audit finding receipt: %w", err)
	}
	receipt, err := readReceiptByID(ctx, s.pool, admittedID)
	if err != nil {
		return Receipt{}, err
	}
	values, err := s.hydrateReceiptDocuments(ctx, []Receipt{receipt})
	if err != nil {
		return Receipt{}, err
	}
	return values[0], nil
}

// ResolveAuditProposals converts Runtime-injected invocation-local keys into
// exact receipts already retained by the same Audit execution. It rejects
// ambiguous, foreign, missing, or repeated keys and never falls back to a
// Run-wide client-key guess.
func (s *Service) ResolveAuditProposals(
	ctx context.Context,
	ownerID, auditID, executionID, runID string,
	keys []ProposalKey,
) ([]ResolvedProposal, error) {
	if ownerID == "" || auditID == "" || executionID == "" || runID == "" ||
		len(keys) > auditdomain.MaximumProposalsPerItem {
		return nil, ErrInvalid
	}
	result := make([]ResolvedProposal, 0, len(keys))
	seen := make(map[string]struct{}, len(keys))
	for _, key := range keys {
		if !validIdentity(key.InvocationID) || !validIdentity(key.ClientKey) {
			return nil, ErrInvalid
		}
		identity := key.InvocationID + "\x00" + key.ClientKey
		if _, duplicate := seen[identity]; duplicate {
			return nil, ErrConflict
		}
		seen[identity] = struct{}{}
		var receiptID string
		var heldProposalJSON []byte
		err := s.pool.QueryRow(ctx, `
SELECT receipt.receipt_id, hold.proposal_ref
  FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_audit_holds AS hold
    ON hold.receipt_id = receipt.receipt_id AND hold.audit_id = $2
  JOIN audits AS audit ON audit.audit_id = hold.audit_id
 WHERE audit.owner_id = $1 AND receipt.owner_id = $1
   AND receipt.audit_id = $2 AND receipt.audit_execution_id = $3
   AND receipt.run_id = $4 AND receipt.invocation_id = $5
   AND receipt.client_key = $6`, ownerID, auditID, executionID, runID,
			key.InvocationID, key.ClientKey).Scan(&receiptID, &heldProposalJSON)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrNotFound
		}
		if err != nil {
			return nil, fmt.Errorf("resolve Audit proposal selection: %w", err)
		}
		var proposal ExactArtifact
		if json.Unmarshal(heldProposalJSON, &proposal) != nil || proposal.Ref.ValidateExact() != nil {
			return nil, artifacts.ErrArtifactIntegrity
		}
		receipt, err := readReceiptByID(ctx, s.pool, receiptID)
		if err != nil {
			return nil, err
		}
		result = append(result, ResolvedProposal{
			ReceiptID: receiptID, Proposal: proposal, Origin: receipt.Origin,
		})
	}
	return result, nil
}

func (s *Service) hydrateReceiptDocuments(
	ctx context.Context,
	receipts []Receipt,
) ([]Receipt, error) {
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(s.pool))
	for index := range receipts {
		receipt := &receipts[index]
		var store artifacts.ScopedStore
		var ref contracts.ArtifactRef
		var expected ExactArtifact
		var err error
		if !receipt.Origin.RunDeleted {
			store, err = artifactService.Run(receipt.Origin.RunID)
			ref, expected = receipt.Proposal.Ref, receipt.Proposal
		} else if len(receipt.AuditHolds) != 0 {
			hold := receipt.AuditHolds[0]
			store, err = artifactService.Project(hold.ProjectID)
			ref, expected = hold.Proposal.Ref, hold.Proposal
		} else {
			return nil, ErrNotFound
		}
		if err != nil {
			return nil, err
		}
		read, err := store.Read(ctx, ref)
		if err != nil {
			return nil, err
		}
		if read.Payload.MediaType != expected.MediaType ||
			int64(len(read.Payload.Data)) != expected.SizeBytes ||
			digestBytes(read.Payload.Data) != expected.Digest {
			return nil, artifacts.ErrArtifactIntegrity
		}
		document, err := auditdomain.DecodeFindingProposal(read.Payload.Data)
		if err != nil || document.ClientKey != receipt.ClientKey {
			return nil, artifacts.ErrArtifactIntegrity
		}
		receipt.Document = document
	}
	return receipts, nil
}

func validListQuery(query ListQuery) bool {
	if query.Limit < 1 || query.Limit > 200 {
		return false
	}
	return (query.AfterCreatedAt == nil) == (query.AfterReceiptID == "")
}

func listReceiptIDs(ctx context.Context, db querier, statement string, args ...any) ([]string, error) {
	rows, err := db.Query(ctx, statement, args...)
	if err != nil {
		return nil, fmt.Errorf("list finding proposal receipts: %w", err)
	}
	defer rows.Close()
	result := make([]string, 0)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("scan finding proposal receipt identity: %w", err)
		}
		result = append(result, id)
	}
	return result, rows.Err()
}

func readReceipts(ctx context.Context, db querier, ids []string) ([]Receipt, error) {
	result := make([]Receipt, 0, len(ids))
	for _, id := range ids {
		receipt, err := readReceiptByID(ctx, db, id)
		if err != nil {
			return nil, err
		}
		result = append(result, receipt)
	}
	return result, nil
}

func (s *Service) ImportIntoAudit(
	ctx context.Context,
	request ImportRequest,
) (AuditHold, bool, error) {
	if request.OwnerID == "" || request.AuditID == "" || request.RunID == "" ||
		request.Proposal.ValidateExact() != nil {
		return AuditHold{}, false, ErrInvalid
	}
	var result AuditHold
	var replayed bool
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		var receiptID, projectID string
		var proposalRefJSON, evidenceJSON []byte
		requestedProposal, _ := json.Marshal(request.Proposal)
		err := tx.QueryRow(ctx, `
SELECT receipt.receipt_id, audit.project_id, receipt.proposal_ref, receipt.evidence
  FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_retention AS retention USING (receipt_id)
  JOIN workflow_runs AS run ON run.run_id = receipt.run_id
  JOIN audits AS audit ON audit.audit_id = $2
 WHERE receipt.owner_id = $1 AND receipt.run_id = $3
   AND receipt.proposal_ref = $4::jsonb
   AND run.owner_id = $1 AND run.project_id = audit.project_id
   AND audit.owner_id = $1
   AND audit.state IN ('draft', 'active', 'waiting_review', 'paused')
   AND NOT EXISTS (
       SELECT 1
         FROM audit_report_candidates AS candidate
         JOIN audit_review_requests AS review
           ON review.request_id = candidate.request_id
          AND review.audit_id = candidate.audit_id
        WHERE candidate.audit_id = audit.audit_id AND review.state = 'pending'
   )
   AND audit.profile_snapshot #>> '{interaction,findingConfirmation}' = 'human-required'
 FOR UPDATE OF receipt, retention, audit`,
			request.OwnerID, request.AuditID, request.RunID, requestedProposal,
		).Scan(&receiptID, &projectID, &proposalRefJSON, &evidenceJSON)
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrNotFound
		}
		if err != nil {
			return fmt.Errorf("lock finding proposal for Audit import: %w", err)
		}
		var sourceProposal contracts.ArtifactRef
		var evidence []ExactArtifact
		if json.Unmarshal(proposalRefJSON, &sourceProposal) != nil ||
			json.Unmarshal(evidenceJSON, &evidence) != nil || !sameRef(sourceProposal, request.Proposal) {
			return ErrConflict
		}
		var heldProposalJSON, heldEvidenceJSON []byte
		var createdAt time.Time
		err = tx.QueryRow(ctx, `
SELECT proposal_ref, evidence, created_at
  FROM finding_proposal_audit_holds
 WHERE receipt_id = $1 AND audit_id = $2`, receiptID, request.AuditID).Scan(
			&heldProposalJSON, &heldEvidenceJSON, &createdAt,
		)
		if err == nil {
			if json.Unmarshal(heldProposalJSON, &result.Proposal) != nil ||
				json.Unmarshal(heldEvidenceJSON, &result.Evidence) != nil {
				return errors.New("decode replayed finding proposal Audit hold")
			}
			result.AuditID, result.ProjectID, result.CreatedAt = request.AuditID, projectID, createdAt
			replayed = true
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}

		artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		namespace := auditdomain.ArtifactNamespace(request.AuditID)
		proposalSource, err := exactArtifactFromRun(ctx, artifactService, request.RunID, sourceProposal)
		if err != nil {
			return err
		}
		result.Proposal, err = retainFindingArtifact(
			ctx, artifactService, request.RunID, projectID, namespace,
			deterministicID("finding-proposal", receiptID), proposalSource,
		)
		if err != nil {
			return err
		}
		result.Evidence = make([]ExactArtifact, 0, len(evidence))
		for index, source := range evidence {
			verified, err := exactArtifactFromRun(ctx, artifactService, request.RunID, source.Ref)
			if err != nil || verified.Digest != source.Digest ||
				verified.MediaType != source.MediaType || verified.SizeBytes != source.SizeBytes {
				if err != nil {
					return err
				}
				return artifacts.ErrArtifactIntegrity
			}
			retained, err := retainFindingArtifact(
				ctx, artifactService, request.RunID, projectID, namespace,
				deterministicID("finding-evidence", receiptID, strconv.Itoa(index+1)), verified,
			)
			if err != nil {
				return err
			}
			result.Evidence = append(result.Evidence, retained)
		}
		result.AuditID, result.ProjectID = request.AuditID, projectID
		proposalJSON, _ := json.Marshal(result.Proposal)
		retainedEvidenceJSON, _ := json.Marshal(result.Evidence)
		err = tx.QueryRow(ctx, `
INSERT INTO finding_proposal_audit_holds (
    receipt_id, audit_id, project_id, proposal_ref, evidence
) VALUES ($1, $2, $3, $4, $5)
RETURNING created_at`, receiptID, request.AuditID, projectID, proposalJSON, retainedEvidenceJSON).Scan(
			&result.CreatedAt,
		)
		return err
	})
	if err != nil {
		return AuditHold{}, false, err
	}
	return result, replayed, nil
}

func exactArtifactFromRun(
	ctx context.Context,
	service *artifacts.Service,
	runID string,
	ref contracts.ArtifactRef,
) (ExactArtifact, error) {
	store, err := service.Run(runID)
	if err != nil {
		return ExactArtifact{}, err
	}
	metadata, err := store.Metadata(ctx, ref)
	if err != nil {
		return ExactArtifact{}, err
	}
	return ExactArtifact{
		Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType,
		SizeBytes: metadata.Size,
	}, nil
}

func retainFindingArtifact(
	ctx context.Context,
	service *artifacts.Service,
	runID, projectID, namespace, name string,
	source ExactArtifact,
) (ExactArtifact, error) {
	result, err := service.ImportFindingArtifact(
		ctx, runID, source.Ref, projectID,
		contracts.ArtifactRef{Namespace: namespace, Name: name},
	)
	if err != nil {
		return ExactArtifact{}, err
	}
	return ExactArtifact{
		Ref: result.TargetRef, Digest: source.Digest,
		MediaType: result.MediaType, SizeBytes: result.Size,
	}, nil
}
