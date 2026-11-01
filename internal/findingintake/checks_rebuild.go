package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const maxChecksRebuildBatch = 128

// ChecksRebuildReport reports one bounded explicit data-migration batch.
// NULL column membership is durable progress; no second receipt is created.
type ChecksRebuildReport struct {
	Examined int  `json:"examined"`
	Rebuilt  int  `json:"rebuilt"`
	More     bool `json:"more"`
}

type pendingChecksProjection struct {
	receiptID, runID string
	original         ExactArtifact
	retention        RetentionState
	projectID        *string
	held             *ExactArtifact
}

// RebuildChecksProjectionBatch derives private check indexes for current
// receipts from their actual immutable source bytes. Callers install the same
// routed BlobRuntime as the Server, including its filesystem root. The normal
// admission reader never invokes this data migration as a fallback.
func (s *Service) RebuildChecksProjectionBatch(ctx context.Context, limit int) (ChecksRebuildReport, error) {
	var report ChecksRebuildReport
	if limit < 1 || limit > maxChecksRebuildBatch {
		return report, ErrInvalid
	}
	rows, err := s.pool.Query(ctx, pendingChecksProjectionSQL, limit)
	if err != nil {
		return report, fmt.Errorf("list pending finding check indexes: %w", err)
	}
	var pending []pendingChecksProjection
	var payloadBytes int64
	for rows.Next() {
		var value pendingChecksProjection
		var original, held []byte
		if err := rows.Scan(&value.receiptID, &value.runID, &original, &value.original.Digest,
			&value.original.MediaType, &value.original.SizeBytes, &value.retention, &value.projectID, &held); err != nil {
			rows.Close()
			return report, err
		}
		if strictjson.Decode(original, &value.original.Ref) != nil || value.original.Validate() != nil ||
			value.original.SizeBytes > auditdomain.MaximumDocumentBytes || value.retention == RetentionDiscarded {
			rows.Close()
			return report, fmt.Errorf("finding check index source %s: %w", value.receiptID, ErrInvalid)
		}
		if held != nil {
			value.held = &ExactArtifact{}
			if strictjson.Decode(held, value.held) != nil || value.held.Validate() != nil ||
				value.projectID == nil || value.held.Digest != value.original.Digest ||
				value.held.MediaType != value.original.MediaType || value.held.SizeBytes != value.original.SizeBytes {
				rows.Close()
				return report, fmt.Errorf("finding check index hold %s: %w", value.receiptID, ErrInvalid)
			}
		}
		if len(pending) > 0 && (len(pending) >= artifacts.MaxExactReadBatchSize || payloadBytes+value.original.SizeBytes > artifacts.MaxPayloadSize) {
			break
		}
		pending = append(pending, value)
		payloadBytes += value.original.SizeBytes
	}
	readErr := rows.Err()
	rows.Close()
	if readErr != nil {
		return report, readErr
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(s.pool))
	for _, value := range pending {
		report.Examined++
		var source artifacts.ScopedStore
		expected := value.original
		if value.held != nil {
			source, err = service.Project(*value.projectID)
			expected = *value.held
		} else if value.retention == RetentionSourceHeld {
			source, err = service.Run(value.runID)
		} else {
			err = artifacts.ErrArtifactNotFound
		}
		if err != nil {
			return report, fmt.Errorf("finding check index source %s: %w", value.receiptID, err)
		}
		content, err := source.ReadExact(ctx, expected)
		if err != nil {
			return report, fmt.Errorf("read finding check index source %s: %w", value.receiptID, err)
		}
		document, err := auditdomain.DecodeFindingProposal(content.Payload.Data)
		if err != nil {
			return report, fmt.Errorf("decode finding check index source %s: %w", value.receiptID, err)
		}
		projection, err := auditdomain.BuildFindingChecksProjection(content.Payload.Data, document)
		if err != nil {
			return report, fmt.Errorf("build finding check index %s: %w", value.receiptID, err)
		}
		encoded, err := auditdomain.EncodeFindingChecksProjection(projection)
		if err != nil {
			return report, err
		}
		ref, _ := json.Marshal(value.original.Ref)
		tag, err := s.pool.Exec(ctx, writeChecksProjectionSQL, value.receiptID, string(encoded), ref,
			value.original.Digest, value.original.MediaType, value.original.SizeBytes)
		if err != nil {
			return report, fmt.Errorf("commit finding check index %s: %w", value.receiptID, err)
		}
		if tag.RowsAffected() == 1 {
			report.Rebuilt++
		} else {
			var committed []byte
			if err := s.pool.QueryRow(ctx, `SELECT checks_projection FROM finding_proposal_receipts WHERE receipt_id=$1`, value.receiptID).Scan(&committed); err != nil {
				return report, err
			}
			stored, err := auditdomain.DecodeFindingChecksProjection(committed)
			if err != nil || !stored.Discarded && (stored.ProposalDigest != projection.ProposalDigest || stored.SizeBytes != projection.SizeBytes) {
				return report, errors.New("finding check index CAS lost to inconsistent content")
			}
		}
	}
	if err := s.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM finding_proposal_receipts WHERE checks_projection IS NULL LIMIT 1)`).Scan(&report.More); err != nil {
		return report, err
	}
	return report, nil
}
