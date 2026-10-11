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

// AuditReceiptKey names one receipt read through one of the owner's Audits.
type AuditReceiptKey struct {
	AuditID   string
	ReceiptID string
}

// GetAuditReceipts reads receipts admitted to the owner's Audit: its native
// child receipts and those it holds in its own Project. Returned receipts
// follow input order, including repeated identities, and expose only the
// requesting Audit's own hold, loaded for the whole page.
func (s *Service) GetAuditReceipts(ctx context.Context, ownerID, auditID string, ids []string) ([]Receipt, error) {
	if auditID == "" || len(ids) > MaxAuditReceiptBatchSize {
		return nil, ErrInvalid
	}
	keys := make([]AuditReceiptKey, len(ids))
	for index, id := range ids {
		keys[index] = AuditReceiptKey{AuditID: auditID, ReceiptID: id}
	}
	return s.GetOwnerAuditReceipts(ctx, ownerID, keys)
}

// GetOwnerAuditReceipts reads each keyed receipt through its own Audit of the
// owner exactly as GetAuditReceipts does, for a page whose findings belong to
// several Audits: one statement reads the receipts, one their holds and one
// exact Artifact read serves each bounded group of proposals, whatever the
// number of Audits. Returned receipts follow input order, including repeated
// keys, and each exposes only the hold of the Audit it was read through.
func (s *Service) GetOwnerAuditReceipts(ctx context.Context, ownerID string, keys []AuditReceiptKey) ([]Receipt, error) {
	if ownerID == "" || len(keys) > MaxAuditReceiptBatchSize {
		return nil, ErrInvalid
	}
	for _, key := range keys {
		if key.AuditID == "" || key.ReceiptID == "" {
			return nil, ErrInvalid
		}
	}
	if len(keys) == 0 {
		return []Receipt{}, nil
	}
	auditIDs, receiptIDs := distinctReceiptKeys(keys)
	rows, err := s.pool.Query(ctx, readOwnerAuditReceiptsSQL, ownerID, auditIDs, receiptIDs)
	if err != nil {
		return nil, fmt.Errorf("read Audit finding receipts: %w", err)
	}
	receipts, err := scanAuditReceiptBatch(rows, keys)
	if err != nil {
		return nil, err
	}
	return s.hydrateAuditReceipts(ctx, ownerID, keys, receipts)
}

// distinctReceiptKeys returns the parallel Audit and receipt identities of
// the distinct keys, in first-seen order.
func distinctReceiptKeys(keys []AuditReceiptKey) ([]string, []string) {
	seen := make(map[AuditReceiptKey]struct{}, len(keys))
	auditIDs := make([]string, 0, len(keys))
	receiptIDs := make([]string, 0, len(keys))
	for _, key := range keys {
		if _, duplicate := seen[key]; duplicate {
			continue
		}
		seen[key] = struct{}{}
		auditIDs = append(auditIDs, key.AuditID)
		receiptIDs = append(receiptIDs, key.ReceiptID)
	}
	return auditIDs, receiptIDs
}

// scanAuditReceiptBatch returns the receipts of keys in input order; a key
// whose receipt is not admitted to its Audit is ErrNotFound.
func scanAuditReceiptBatch(rows pgx.Rows, keys []AuditReceiptKey) ([]Receipt, error) {
	defer rows.Close()
	byKey := make(map[AuditReceiptKey]Receipt, len(keys))
	for rows.Next() {
		var auditID string
		receipt, err := scanReceiptRow(throughAudit{row: rows, auditID: &auditID})
		if err != nil {
			return nil, err
		}
		byKey[AuditReceiptKey{AuditID: auditID, ReceiptID: receipt.ReceiptID}] = receipt
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit finding receipts: %w", err)
	}
	result := make([]Receipt, len(keys))
	for index, key := range keys {
		receipt, ok := byKey[key]
		if !ok {
			return nil, ErrNotFound
		}
		result[index] = receipt
	}
	return result, nil
}

// throughAudit scans a receipt row followed by the Audit it was read through.
type throughAudit struct {
	row     pgx.Row
	auditID *string
}

func (r throughAudit) Scan(dest ...any) error {
	return r.row.Scan(append(dest, r.auditID)...)
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

// hydrateAuditReceipts attaches to each receipt only the hold of the Audit
// keys[index] reads it through and reads its exact proposal, from that hold's
// retained copy once the source Run is deleted.
func (s *Service) hydrateAuditReceipts(
	ctx context.Context, ownerID string, keys []AuditReceiptKey, receipts []Receipt,
) ([]Receipt, error) {
	if len(receipts) == 0 {
		return receipts, nil
	}
	holds, err := s.readAuditHoldsBatch(ctx, ownerID, keys)
	if err != nil {
		return nil, err
	}
	for index := range receipts {
		receipts[index].AuditHolds = make([]AuditHold, 0, 1)
		for _, hold := range holds[keys[index].ReceiptID] {
			if hold.AuditID == keys[index].AuditID {
				receipts[index].AuditHolds = append(receipts[index].AuditHolds, hold)
			}
		}
	}
	return s.hydrateAuditReceiptBatch(ctx, receipts)
}

// readAuditHoldsBatch returns, keyed by receipt, only the holds of the Audits
// the keys read their receipts through, and only when that Audit belongs to
// the owner and the hold's Project. A receipt read through one Audit never
// names another Audit's copy or reads its bytes after source Run deletion, as
// collection publication also requires.
func (s *Service) readAuditHoldsBatch(
	ctx context.Context, ownerID string, keys []AuditReceiptKey,
) (map[string][]AuditHold, error) {
	auditIDs, receiptIDs := distinctReceiptKeys(keys)
	rows, err := s.pool.Query(ctx, readOwnerAuditHoldsSQL, auditIDs, receiptIDs, ownerID)
	if err != nil {
		return nil, fmt.Errorf("list finding proposal Audit holds: %w", err)
	}
	return scanAuditHoldsBatch(rows, len(receiptIDs))
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
	if expected.Verify(read.Exact()) != nil {
		return auditdomain.FindingProposal{}, artifacts.ErrArtifactIntegrity
	}
	document, err := auditdomain.DecodeFindingProposal(read.Payload.Data)
	if err != nil || document.ClientKey != clientKey {
		return auditdomain.FindingProposal{}, artifacts.ErrArtifactIntegrity
	}
	return document, nil
}
