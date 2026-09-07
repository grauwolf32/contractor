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

// GetAuditReceipts preserves GetAuditReceipt's ownership, holds and exact
// artifact hydration, while loading related records for the whole page.
// Returned receipts follow input order, including repeated identities.
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
  JOIN finding_proposal_retention AS retention USING (receipt_id)
  JOIN audits AS audit ON audit.audit_id = $2 AND audit.owner_id = $1
 WHERE receipt.receipt_id = ANY($3::text[]) AND receipt.owner_id = $1
   AND (receipt.audit_id = $2 OR EXISTS (
       SELECT 1 FROM finding_proposal_audit_holds AS hold
        WHERE hold.receipt_id = receipt.receipt_id AND hold.audit_id = $2
   ))`, ownerID, auditID, ids)
	if err != nil {
		return nil, fmt.Errorf("read Audit finding receipts: %w", err)
	}
	receipts, err := scanAuditReceiptBatch(rows, ids)
	if err != nil {
		return nil, err
	}
	holds, err := s.readAuditHoldsBatch(ctx, ids)
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

func scanAuditReceiptBatch(rows pgx.Rows, ids []string) ([]Receipt, error) {
	defer rows.Close()
	byID := make(map[string]Receipt, len(ids))
	for rows.Next() {
		receipt, err := scanReceiptRow(rows)
		if err != nil {
			return nil, err
		}
		byID[receipt.ReceiptID] = receipt
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit finding receipts: %w", err)
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

func (s *Service) readAuditHoldsBatch(ctx context.Context, ids []string) (map[string][]AuditHold, error) {
	rows, err := s.pool.Query(ctx, `
SELECT receipt_id, audit_id, project_id, proposal_ref, evidence, created_at
  FROM finding_proposal_audit_holds
 WHERE receipt_id = ANY($1::text[])
 ORDER BY receipt_id, created_at, audit_id`, ids)
	if err != nil {
		return nil, fmt.Errorf("list finding proposal Audit holds: %w", err)
	}
	defer rows.Close()
	result := make(map[string][]AuditHold, len(ids))
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
	for start := 0; start < len(receipts); start += artifacts.MaxExactReadBatchSize {
		batch := receipts[start:min(start+artifacts.MaxExactReadBatchSize, len(receipts))]
		requests := make([]artifacts.ExactReadRequest, len(batch))
		expected := make([]ExactArtifact, len(batch))
		for index, receipt := range batch {
			request, artifact, err := receiptArtifact(receipt)
			if err != nil {
				return nil, err
			}
			requests[index], expected[index] = request, artifact
		}
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
	if read.Payload.MediaType != expected.MediaType || int64(len(read.Payload.Data)) != expected.SizeBytes || digestBytes(read.Payload.Data) != expected.Digest {
		return auditdomain.FindingProposal{}, artifacts.ErrArtifactIntegrity
	}
	document, err := auditdomain.DecodeFindingProposal(read.Payload.Data)
	if err != nil || document.ClientKey != clientKey {
		return auditdomain.FindingProposal{}, artifacts.ErrArtifactIntegrity
	}
	return document, nil
}
