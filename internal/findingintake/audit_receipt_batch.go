package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/jackc/pgx/v5"
)

// MaxAuditReceiptBatchSize matches the bounded Audit finding page.
const MaxAuditReceiptBatchSize = 200

// GetAuditReceipts reads receipts admitted to the owner's Audit: its native
// child receipts and those it holds in its own Project. Returned receipts
// follow input order, including repeated identities, and expose only the
// requesting Audit's own hold, loaded for the whole page.
func (s *Service) GetAuditReceipts(ctx context.Context, ownerID, auditID string, ids []string) ([]Receipt, error) {
	if ownerID == "" || auditID == "" || len(ids) > MaxAuditReceiptBatchSize {
		return nil, ErrInvalid
	}
	for _, id := range ids {
		if id == "" {
			return nil, ErrInvalid
		}
	}
	if len(ids) == 0 {
		return []Receipt{}, nil
	}
	rows, err := s.pool.Query(ctx, `
SELECT `+receiptProjection+`
  FROM finding_proposal_receipts AS receipt
  JOIN audits AS audit ON audit.audit_id = $2 AND audit.owner_id = $1
 WHERE receipt.receipt_id = ANY($3::text[]) AND receipt.owner_id = $1
   AND (receipt.audit_id = $2 OR EXISTS (
       SELECT 1 FROM finding_proposal_audit_holds AS hold
        WHERE hold.receipt_id = receipt.receipt_id AND hold.audit_id = $2
          AND hold.project_id = audit.project_id
   ))`, ownerID, auditID, ids)
	if err != nil {
		return nil, fmt.Errorf("read Audit finding receipts: %w", err)
	}
	receipts, err := scanAuditReceiptBatch(rows, ids)
	if err != nil {
		return nil, err
	}
	return s.hydrateAuditReceipts(ctx, ownerID, auditID, receipts)
}

func scanAuditReceiptBatch(rows pgx.Rows, ids []string) ([]Receipt, error) {
	scanned, err := scanReceiptRows(rows)
	if err != nil {
		return nil, err
	}
	byID := make(map[string]Receipt, len(scanned))
	for _, receipt := range scanned {
		byID[receipt.ReceiptID] = receipt
	}
	result := make([]Receipt, len(ids))
	for index, id := range ids {
		receipt, ok := byID[id]
		if !ok {
			return nil, ErrNotFound
		}
		result[index] = receipt
	}
	return result, nil
}

func scanReceiptRows(rows pgx.Rows) ([]Receipt, error) {
	defer rows.Close()
	result := make([]Receipt, 0)
	for rows.Next() {
		receipt, err := scanReceiptRow(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, receipt)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit finding receipts: %w", err)
	}
	return result, nil
}

// hydrateAuditReceipts attaches only the requesting Audit's hold to each
// receipt and reads its exact proposal, from that hold's retained copy once
// the source Run is deleted.
func (s *Service) hydrateAuditReceipts(
	ctx context.Context, ownerID, auditID string, receipts []Receipt,
) ([]Receipt, error) {
	if len(receipts) == 0 {
		return receipts, nil
	}
	ids := make([]string, len(receipts))
	for index := range receipts {
		ids[index] = receipts[index].ReceiptID
	}
	holds, err := s.readAuditHoldsBatch(ctx, ownerID, auditID, ids)
	if err != nil {
		return nil, err
	}
	for index := range receipts {
		receipts[index].AuditHolds = holds[receipts[index].ReceiptID]
		if receipts[index].AuditHolds == nil {
			receipts[index].AuditHolds = []AuditHold{}
		}
	}
	return s.hydrateAuditReceiptBatch(ctx, receipts)
}

// readAuditHoldsBatch returns only the requesting Audit's holds, and only
// when that Audit belongs to the owner and the hold's Project. A receipt read
// through one Audit never names another Audit's copy or reads its bytes after
// source Run deletion, as collection publication also requires.
func (s *Service) readAuditHoldsBatch(
	ctx context.Context, ownerID, auditID string, ids []string,
) (map[string][]AuditHold, error) {
	rows, err := s.pool.Query(ctx, `
SELECT hold.receipt_id, hold.audit_id, hold.project_id, hold.proposal_ref,
       hold.evidence, hold.created_at
  FROM finding_proposal_audit_holds AS hold
  JOIN audits AS audit
    ON audit.audit_id = hold.audit_id AND audit.project_id = hold.project_id
 WHERE hold.receipt_id = ANY($1::text[]) AND hold.audit_id = $2
   AND audit.owner_id = $3
 ORDER BY hold.receipt_id, hold.created_at, hold.audit_id`, ids, auditID, ownerID)
	if err != nil {
		return nil, fmt.Errorf("list finding proposal Audit holds: %w", err)
	}
	return scanAuditHoldsBatch(rows, len(ids))
}

func readRunAuditHoldsBatch(ctx context.Context, db querier, ids []string) (map[string][]AuditHold, error) {
	rows, err := db.Query(ctx, `
SELECT receipt_id, audit_id, project_id, proposal_ref, evidence, created_at
  FROM finding_proposal_audit_holds
 WHERE receipt_id = ANY($1::text[])
 ORDER BY receipt_id, created_at, audit_id`, ids)
	if err != nil {
		return nil, fmt.Errorf("list Run finding proposal Audit holds: %w", err)
	}
	return scanAuditHoldsBatch(rows, len(ids))
}

func scanAuditHoldsBatch(rows pgx.Rows, size int) (map[string][]AuditHold, error) {
	defer rows.Close()
	result := make(map[string][]AuditHold, size)
	for rows.Next() {
		var id string
		var hold AuditHold
		var proposal, evidence []byte
		if err := rows.Scan(&id, &hold.AuditID, &hold.ProjectID, &proposal, &evidence, &hold.CreatedAt); err != nil {
			return nil, fmt.Errorf("scan finding proposal Audit hold: %w", err)
		}
		if json.Unmarshal(proposal, &hold.Proposal) != nil || json.Unmarshal(evidence, &hold.Evidence) != nil {
			return nil, errors.New("decode finding proposal Audit hold")
		}
		result[id] = append(result[id], hold)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate finding proposal Audit holds: %w", err)
	}
	return result, nil
}

func (s *Service) hydrateAuditReceiptBatch(ctx context.Context, receipts []Receipt) ([]Receipt, error) {
	repository := artifacts.NewPostgresRepository(s.pool)
	for start := 0; start < len(receipts); {
		requests := make([]artifacts.ExactReadRequest, 0, artifacts.MaxExactReadBatchSize)
		expected := make([]ExactArtifact, 0, artifacts.MaxExactReadBatchSize)
		var bytes int64
		end := start
		for end < len(receipts) && len(requests) < artifacts.MaxExactReadBatchSize {
			receipt := receipts[end]
			request, artifact, err := receiptArtifact(receipt)
			if err != nil {
				return nil, err
			}
			if artifact.SizeBytes < 0 {
				return nil, artifacts.ErrArtifactIntegrity
			}
			if artifact.SizeBytes > artifacts.MaxPayloadSize {
				return nil, artifacts.ErrPayloadTooLarge
			}
			if len(requests) > 0 && artifact.SizeBytes > artifacts.MaxPayloadSize-bytes {
				break
			}
			requests = append(requests, request)
			expected = append(expected, artifact)
			bytes += artifact.SizeBytes
			end++
		}
		batch := receipts[start:end]
		reads, err := repository.ReadExactBatch(ctx, requests)
		if err != nil {
			return nil, err
		}
		for index, read := range reads {
			document, err := decodeReceiptDocument(read, expected[index], batch[index].ClientKey)
			if err != nil {
				return nil, err
			}
			batch[index].Document = document
		}
		start = end
	}
	return receipts, nil
}

// Both single and batch receipt reads select the same retained source and
// enforce the same payload identity before exposing a proposal document.
func receiptArtifact(receipt Receipt) (artifacts.ExactReadRequest, ExactArtifact, error) {
	var scope artifacts.Scope
	var expected ExactArtifact
	var err error
	if !receipt.Origin.RunDeleted {
		scope, err = artifacts.RunScope(receipt.Origin.RunID)
		expected = receipt.Proposal
	} else if len(receipt.AuditHolds) != 0 {
		hold := receipt.AuditHolds[0]
		scope, err = artifacts.ProjectScope(hold.ProjectID)
		expected = hold.Proposal
	} else {
		return artifacts.ExactReadRequest{}, ExactArtifact{}, ErrNotFound
	}
	if err != nil {
		return artifacts.ExactReadRequest{}, ExactArtifact{}, err
	}
	return artifacts.ExactReadRequest{Scope: scope, Ref: expected.Ref}, expected, nil
}

func decodeReceiptDocument(read artifacts.ReadResult, expected ExactArtifact, clientKey string) (auditdomain.FindingProposal, error) {
	if read.Payload.MediaType != expected.MediaType || int64(len(read.Payload.Data)) != expected.SizeBytes || auditdomain.DigestBytes(read.Payload.Data) != expected.Digest {
		return auditdomain.FindingProposal{}, artifacts.ErrArtifactIntegrity
	}
	document, err := auditdomain.DecodeFindingProposal(read.Payload.Data)
	if err != nil || document.ClientKey != clientKey {
		return auditdomain.FindingProposal{}, artifacts.ErrArtifactIntegrity
	}
	return document, nil
}
