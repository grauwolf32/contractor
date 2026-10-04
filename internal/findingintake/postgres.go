package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
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
	result, err := scanReceiptRow(row)
	if err != nil {
		return Receipt{}, err
	}
	holds, err := readAuditHolds(ctx, db, result.ReceiptID)
	if err != nil {
		return Receipt{}, err
	}
	result.AuditHolds = holds
	return result, nil
}

func scanReceiptRow(row pgx.Row) (Receipt, error) {
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
	result, err := s.listRunReceiptRows(ctx, ownerID, runID, query)
	if err != nil || len(result) == 0 {
		return result, err
	}
	ids := make([]string, len(result))
	for index := range result {
		ids[index] = result[index].ReceiptID
	}
	holds, err := readRunAuditHoldsBatch(ctx, s.pool, ids)
	if err != nil {
		return nil, err
	}
	for index := range result {
		result[index].AuditHolds = holds[result[index].ReceiptID]
		if result[index].AuditHolds == nil {
			result[index].AuditHolds = []AuditHold{}
		}
	}
	return s.hydrateAuditReceiptBatch(ctx, result)
}

func (s *Service) listRunReceiptRows(
	ctx context.Context, ownerID, runID string, query ListQuery,
) ([]Receipt, error) {
	if ownerID == "" || runID == "" || !validListQuery(query) {
		return nil, ErrInvalid
	}
	rows, err := s.pool.Query(ctx, `
SELECT `+receiptProjection+`
  FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_retention AS retention USING (receipt_id)
  JOIN workflow_runs AS run ON run.run_id = receipt.run_id
 WHERE receipt.owner_id = $1 AND receipt.run_id = $2 AND run.owner_id = $1
   AND ($3::timestamptz IS NULL OR (receipt.created_at, receipt.receipt_id) > ($3, $4))
 ORDER BY receipt.created_at, receipt.receipt_id
 LIMIT $5`, ownerID, runID, query.AfterCreatedAt, query.AfterReceiptID, query.Limit)
	if err != nil {
		return nil, fmt.Errorf("list Run finding proposals: %w", err)
	}
	result, err := scanReceiptRows(rows)
	if err != nil {
		return nil, err
	}
	return result, nil
}

// ListAuditInbox lists the Audit's native child receipts and the receipts it
// holds, each with only this Audit's hold. After the source Run is deleted a
// proposal is readable only from this Audit's retained copy, so a native
// receipt the Audit never retained, such as one collection rejected, leaves
// the inbox instead of failing the page.
func (s *Service) ListAuditInbox(
	ctx context.Context,
	ownerID, auditID string,
	query ListQuery,
) ([]Receipt, error) {
	return s.listAuditInbox(ctx, ownerID, auditID, query, `
SELECT `+receiptProjection+`
  FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_retention AS retention USING (receipt_id)
 WHERE receipt.owner_id = $1
   AND ((receipt.audit_id = $2 AND retention.source_run_deleted_at IS NULL) OR EXISTS (
       SELECT 1 FROM finding_proposal_audit_holds AS hold
        JOIN audits AS audit
          ON audit.audit_id = hold.audit_id AND audit.project_id = hold.project_id
       WHERE hold.receipt_id = receipt.receipt_id
         AND hold.audit_id = $2 AND audit.owner_id = $1
   ))
   AND ($3::timestamptz IS NULL OR (receipt.created_at, receipt.receipt_id) > ($3, $4))
 ORDER BY receipt.created_at, receipt.receipt_id
 LIMIT $5`)
}

// ListAuditHeldInbox lists only receipts with an exact hold in the Audit. A
// native child receipt that collection did not retain, such as a rejected
// invalid proposal, stays in the owner inbox while its source Run exists but
// is never schedulable work.
func (s *Service) ListAuditHeldInbox(
	ctx context.Context,
	ownerID, auditID string,
	query ListQuery,
) ([]Receipt, error) {
	return s.listAuditInbox(ctx, ownerID, auditID, query, `
SELECT `+receiptProjection+`
  FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_retention AS retention USING (receipt_id)
  JOIN finding_proposal_audit_holds AS hold
    ON hold.receipt_id = receipt.receipt_id AND hold.audit_id = $2
  JOIN audits AS audit
    ON audit.audit_id = hold.audit_id AND audit.project_id = hold.project_id
 WHERE receipt.owner_id = $1 AND audit.owner_id = $1
   AND ($3::timestamptz IS NULL OR (receipt.created_at, receipt.receipt_id) > ($3, $4))
 ORDER BY receipt.created_at, receipt.receipt_id
 LIMIT $5`)
}

// listAuditInbox filters in the statement, so a page is never short because a
// row was dropped after LIMIT, and hydrates through the same Audit-scoped
// hold path as GetAuditReceipts.
func (s *Service) listAuditInbox(
	ctx context.Context,
	ownerID, auditID string,
	query ListQuery,
	statement string,
) ([]Receipt, error) {
	if ownerID == "" || auditID == "" || !validListQuery(query) {
		return nil, ErrInvalid
	}
	rows, err := s.pool.Query(ctx, statement,
		ownerID, auditID, query.AfterCreatedAt, query.AfterReceiptID, query.Limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit finding receipts: %w", err)
	}
	receipts, err := scanReceiptRows(rows)
	if err != nil {
		return nil, err
	}
	return s.hydrateAuditReceipts(ctx, ownerID, auditID, receipts)
}

// GetAuditReceipt reads one receipt admitted to the owner's Audit through
// GetAuditReceipts, so it exposes only that Audit's own hold and, after source
// Run deletion, reads the proposal from that hold's retained copy.
func (s *Service) GetAuditReceipt(
	ctx context.Context,
	ownerID, auditID, receiptID string,
) (Receipt, error) {
	receipts, err := s.GetAuditReceipts(ctx, ownerID, auditID, []string{receiptID})
	if err != nil {
		return Receipt{}, err
	}
	return receipts[0], nil
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
		hydrated, err := s.hydrateReceiptDocuments(ctx, []Receipt{receipt})
		if err != nil {
			return nil, err
		}
		if len(hydrated) != 1 {
			return nil, artifacts.ErrArtifactIntegrity
		}
		result = append(result, ResolvedProposal{
			ReceiptID: receiptID, Proposal: proposal, Origin: receipt.Origin,
			Document: hydrated[0].Document,
		})
	}
	return result, nil
}

func (s *Service) hydrateReceiptDocuments(ctx context.Context, receipts []Receipt) ([]Receipt, error) {
	repository := artifacts.NewPostgresRepository(s.pool)
	for index := range receipts {
		receipt := &receipts[index]
		request, expected, err := receiptArtifact(*receipt)
		if err != nil {
			return nil, err
		}
		read, err := repository.Read(ctx, request.Scope, request.Ref)
		if err != nil {
			return nil, err
		}
		document, err := decodeReceiptDocument(read, expected, receipt.ClientKey)
		if err != nil {
			return nil, err
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

// Owner imports target only an Audit that still accepts new work. Collection
// of an Audit child Run is also admitted while the Audit finalizes or cancels:
// a proposal found by a child Run is never dropped because its Audit began to
// close, and the settlement barrier waits for that collection before the
// report is built or the Audit becomes Cancelled.
var (
	ownerImportAuditStates         = []string{"draft", "active", "waiting_review", "paused"}
	collectionRetentionAuditStates = []string{
		"draft", "active", "waiting_review", "paused", "finalizing", "cancelling",
	}
)

func (s *Service) ImportIntoAudit(
	ctx context.Context,
	request ImportRequest,
) (AuditHold, bool, error) {
	if !validImportRequest(request) {
		return AuditHold{}, false, ErrInvalid
	}
	var hold AuditHold
	var replayed bool
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lockFindingImportAuthority(ctx, tx, request, false); err != nil {
			return err
		}
		var err error
		hold, replayed, err = s.importIntoAuditWithTx(ctx, tx, request, false)
		return err
	})
	if err != nil {
		return AuditHold{}, false, err
	}
	return hold, replayed, nil
}

// maxCollectionTransactionWork bounds the work of one collection transaction.
// A proposal counts once for its receipt, hold and direct-verification work,
// plus once for each proposal or evidence revision it imports. A transaction
// always retains at least one proposal, so even one with the maximum evidence
// commits alone, and a collection attempt that outlives one bounded
// transaction commits progress instead of rolling back a whole page.
const maxCollectionTransactionWork = 512

// RetainAuditCollectionBatch transfers a bounded page of an Audit child Run's
// proposals into its owning Audit during collection. Unlike ImportIntoAudit
// it admits a finalizing or cancelling Audit, and a terminal (or deleting)
// Audit returns ErrAuditClosed instead of ErrNotFound: its report is sealed or
// its data is being purged, so the proposals stay held by their source Run and
// the caller must not retry.
//
// The page commits in transactions bounded by maxCollectionTransactionWork.
// Each holds the source-Run and destination-Audit locks until it commits, and
// each receipt uses the same exact artifact and direct-verification checks as
// an owner import. A failed transaction leaves earlier ones committed; a retry
// skips them. A proposal whose own exact data can never be retained is
// rejected as RejectAuditCollection does, in the transaction that commits the
// proposals before it, so it cannot block the rest of the page.
func (s *Service) RetainAuditCollectionBatch(ctx context.Context, requests []ImportRequest) error {
	if len(requests) == 0 || len(requests) > MaxAuditReceiptBatchSize {
		return ErrInvalid
	}
	first := requests[0]
	for _, request := range requests {
		if !validImportRequest(request) || request.OwnerID != first.OwnerID ||
			request.AuditID != first.AuditID || request.RunID != first.RunID {
			return ErrInvalid
		}
	}
	var rejection *unretainableProposal
	for len(requests) != 0 {
		committed, err := s.retainAuditCollectionPrefix(ctx, requests, rejection)
		var unretainable *unretainableProposal
		if rejection == nil && errors.As(err, &unretainable) {
			// The transaction rolled back. Commit the proposals before the
			// unretainable one again, together with its rejection.
			rejection = unretainable
			continue
		}
		if err != nil {
			return err
		}
		requests, rejection = requests[committed:], nil
	}
	return nil
}

// retainAuditCollectionPrefix retains requests in order in one transaction
// until their work reaches maxCollectionTransactionWork, at least one, and
// returns how many it committed. With rejection set, it rejects the request at
// rejection.index once it reaches it and commits through that request.
func (s *Service) retainAuditCollectionPrefix(
	ctx context.Context, requests []ImportRequest, rejection *unretainableProposal,
) (int, error) {
	committed := 0
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		committed = 0
		if err := lockFindingImportAuthority(ctx, tx, requests[0], true); err != nil {
			return err
		}
		work := 0
		for index, request := range requests {
			if work >= maxCollectionTransactionWork {
				break
			}
			if rejection != nil && index == rejection.index {
				committed = index + 1
				return rejectAuditCollectionWithTx(ctx, tx, request, rejection.reason)
			}
			hold, replayed, err := s.importIntoAuditWithTx(ctx, tx, request, true)
			var unretainable *unretainableProposal
			if errors.As(err, &unretainable) {
				unretainable.index = index
			}
			if err != nil {
				return err
			}
			committed = index + 1
			work += collectionWork(hold, replayed)
		}
		return nil
	})
	return committed, err
}

// collectionWork weighs one retained receipt for maxCollectionTransactionWork.
func collectionWork(hold AuditHold, replayed bool) int {
	if replayed {
		return 1
	}
	return 2 + len(hold.Evidence)
}

// unretainableProposal is a deterministic failure of one proposal's own exact
// data: a pinned proposal or evidence revision is missing or no longer matches
// its receipt, or the receipt conflicts with the request. No retry can retain
// the proposal, so collection rejects it with reason and keeps the others.
type unretainableProposal struct {
	reason string
	err    error
	// index is the failed request's position in its collection batch.
	index int
}

func (e *unretainableProposal) Error() string { return e.reason + ": " + e.err.Error() }
func (e *unretainableProposal) Unwrap() error { return e.err }

// classifyProposalData marks a failure of one proposal's own data as
// unretainable and returns any other error unchanged. Callers apply it only to
// checks of that data, never to storage or shared Run output failures.
func classifyProposalData(err error) error {
	switch {
	case errors.Is(err, artifacts.ErrArtifactNotFound), errors.Is(err, artifacts.ErrExactRevisionRequired),
		errors.Is(err, artifacts.ErrInvalidName):
		return &unretainableProposal{reason: "finding-proposal-artifact-missing", err: err}
	case errors.Is(err, artifacts.ErrArtifactIntegrity):
		return &unretainableProposal{reason: "finding-proposal-artifact-invalid", err: err}
	case errors.Is(err, ErrConflict):
		return &unretainableProposal{reason: "finding-proposal-receipt-conflict", err: err}
	default:
		return err
	}
}

func validImportRequest(request ImportRequest) bool {
	return request.OwnerID != "" && request.AuditID != "" && request.RunID != "" &&
		request.Proposal.ValidateExact() == nil
}

// lockFindingImportAuthority takes the source-Run and destination-Audit locks
// that every hold creation in tx relies on, in the order Run deletion and
// Audit purge use.
func lockFindingImportAuthority(
	ctx context.Context, tx pgx.Tx, request ImportRequest, collection bool,
) error {
	var sourceProjectID string
	err := tx.QueryRow(ctx, `
SELECT project_id FROM workflow_runs
 WHERE run_id = $1 AND owner_id = $2 AND project_id IS NOT NULL
 FOR KEY SHARE`, request.RunID, request.OwnerID).Scan(&sourceProjectID)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	if err != nil {
		return fmt.Errorf("lock source Run for Audit finding import: %w", err)
	}
	var lockedAuditID, auditState string
	err = tx.QueryRow(ctx, `
SELECT audit_id, state FROM audits
 WHERE audit_id = $1 AND owner_id = $2 AND project_id = $3
 FOR UPDATE`, request.AuditID, request.OwnerID, sourceProjectID).Scan(&lockedAuditID, &auditState)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	if err != nil {
		return fmt.Errorf("lock destination Audit for finding import: %w", err)
	}
	if collection && !slices.Contains(collectionRetentionAuditStates, auditState) {
		return ErrAuditClosed
	}
	return nil
}

// importIntoAuditWithTx creates or replays one Audit hold in tx, which already
// holds lockFindingImportAuthority's locks for request.
func (s *Service) importIntoAuditWithTx(
	ctx context.Context, tx pgx.Tx, request ImportRequest, collection bool,
) (AuditHold, bool, error) {
	admittedStates := ownerImportAuditStates
	if collection {
		admittedStates = collectionRetentionAuditStates
	}
	var result AuditHold
	var receiptID, projectID, invocationID, clientKey, workflowClosureDigest string
	var proposalDigest, proposalMediaType string
	var proposalSizeBytes int64
	var proposalRefJSON, evidenceJSON []byte
	requestedProposal, _ := json.Marshal(request.Proposal)
	err := tx.QueryRow(ctx, importIntoAuditWithTxSQL,
		request.OwnerID, request.AuditID, request.RunID, requestedProposal, admittedStates,
	).Scan(
		&receiptID, &projectID, &proposalRefJSON, &evidenceJSON,
		&invocationID, &clientKey, &workflowClosureDigest,
		&proposalDigest, &proposalMediaType, &proposalSizeBytes,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return AuditHold{}, false, ErrNotFound
	}
	if err != nil {
		return AuditHold{}, false, fmt.Errorf("lock finding proposal for Audit import: %w", err)
	}
	var sourceProposal contracts.ArtifactRef
	var evidence []ExactArtifact
	if json.Unmarshal(proposalRefJSON, &sourceProposal) != nil ||
		json.Unmarshal(evidenceJSON, &evidence) != nil || !sourceProposal.SameExact(request.Proposal) {
		return AuditHold{}, false, classifyProposalData(ErrConflict)
	}
	proposalSource := ExactArtifact{
		Ref: sourceProposal, Digest: proposalDigest,
		MediaType: proposalMediaType, SizeBytes: proposalSizeBytes,
	}
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
	var heldProposalJSON, heldEvidenceJSON []byte
	var createdAt time.Time
	replayed := false
	err = tx.QueryRow(ctx, `
SELECT proposal_ref, evidence, created_at
  FROM finding_proposal_audit_holds
 WHERE receipt_id = $1 AND audit_id = $2`, receiptID, request.AuditID).Scan(
		&heldProposalJSON, &heldEvidenceJSON, &createdAt,
	)
	if err == nil {
		if json.Unmarshal(heldProposalJSON, &result.Proposal) != nil ||
			json.Unmarshal(heldEvidenceJSON, &result.Evidence) != nil {
			return AuditHold{}, false, errors.New("decode replayed finding proposal Audit hold")
		}
		result.AuditID, result.ProjectID, result.CreatedAt = request.AuditID, projectID, createdAt
		replayed = true
	} else if !errors.Is(err, pgx.ErrNoRows) {
		return AuditHold{}, false, err
	} else {
		namespace := auditdomain.ArtifactNamespace(request.AuditID)
		verifiedProposal, verifyErr := exactArtifactFromRun(
			ctx, artifactService, request.RunID, sourceProposal,
		)
		if verifyErr != nil {
			return AuditHold{}, false, classifyProposalData(verifyErr)
		}
		if verifiedProposal.Digest != proposalSource.Digest ||
			verifiedProposal.MediaType != proposalSource.MediaType ||
			verifiedProposal.SizeBytes != proposalSource.SizeBytes {
			return AuditHold{}, false, classifyProposalData(artifacts.ErrArtifactIntegrity)
		}
		result.Proposal, err = retainFindingArtifact(
			ctx, artifactService, request.RunID, projectID, namespace,
			deterministicID("finding-proposal", receiptID), proposalSource,
		)
		if err != nil {
			return AuditHold{}, false, err
		}
		result.Evidence = make([]ExactArtifact, 0, len(evidence))
		for index, source := range evidence {
			verified, err := exactArtifactFromRun(ctx, artifactService, request.RunID, source.Ref)
			if err != nil {
				return AuditHold{}, false, classifyProposalData(err)
			}
			if verified.Digest != source.Digest || verified.MediaType != source.MediaType ||
				verified.SizeBytes != source.SizeBytes {
				return AuditHold{}, false, classifyProposalData(artifacts.ErrArtifactIntegrity)
			}
			retained, err := retainFindingArtifact(
				ctx, artifactService, request.RunID, projectID, namespace,
				deterministicID("finding-evidence", receiptID, strconv.Itoa(index+1)), verified,
			)
			if err != nil {
				return AuditHold{}, false, err
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
		if err != nil {
			return AuditHold{}, false, err
		}
	}
	if err := tryAcceptDirectVerification(ctx, tx, artifactService, directVerificationInput{
		AuditID: request.AuditID, ProjectID: projectID,
		ReceiptID: receiptID, RunID: request.RunID,
		InvocationID: invocationID, ClientKey: clientKey,
		WorkflowClosureDigest: workflowClosureDigest,
		Proposal:              proposalSource,
	}); err != nil {
		return AuditHold{}, false, err
	}
	return result, replayed, nil
}

// RejectAuditCollection records that collection refused one invalid proposal
// of an Audit child Run, so only that proposal is left out of the Audit. The
// receipt stays held by its source Run. The finding.proposal_rejected event is
// written once per receipt; a replay is a no-op. A terminal or deleting Audit
// returns ErrAuditClosed, as RetainAuditCollectionBatch does.
func (s *Service) RejectAuditCollection(
	ctx context.Context,
	request ImportRequest,
	reason string,
) error {
	if !validImportRequest(request) || !validIdentity(reason) {
		return ErrInvalid
	}
	return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		var auditState string
		err := tx.QueryRow(ctx, `
SELECT state FROM audits
 WHERE audit_id = $1 AND owner_id = $2
 FOR UPDATE`, request.AuditID, request.OwnerID).Scan(&auditState)
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrNotFound
		}
		if err != nil {
			return fmt.Errorf("lock Audit for finding proposal rejection: %w", err)
		}
		if !slices.Contains(collectionRetentionAuditStates, auditState) {
			return ErrAuditClosed
		}
		return rejectAuditCollectionWithTx(ctx, tx, request, reason)
	})
}

// rejectAuditCollectionWithTx records one rejection in tx, which holds the
// destination Audit lock for a state that admits collection.
func rejectAuditCollectionWithTx(ctx context.Context, tx pgx.Tx, request ImportRequest, reason string) error {
	requestedProposal, _ := json.Marshal(request.Proposal)
	var receiptID string
	var recorded bool
	err := tx.QueryRow(ctx, `
SELECT receipt.receipt_id, EXISTS (
       SELECT 1 FROM audit_events AS event
        WHERE event.audit_id = receipt.audit_id
          AND event.kind = 'finding.proposal_rejected'
          AND event.entity_id = receipt.receipt_id
   )
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.owner_id = $1 AND receipt.audit_id = $2 AND receipt.run_id = $3
   AND receipt.proposal_ref = $4::jsonb`,
		request.OwnerID, request.AuditID, request.RunID, requestedProposal,
	).Scan(&receiptID, &recorded)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	if err != nil || recorded {
		return err
	}
	return auditstore.NewPostgresStore(tx).RecordRejectedFindingProposal(ctx,
		auditstore.RejectedFindingProposalParams{
			AuditID: request.AuditID, ReceiptID: receiptID,
			RunID: request.RunID, Reason: reason,
		})
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
