package findingintake

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// ReceiptReferences is a bounded result projection; the complete source origin
// remains readable through Project Finding sources when a result reaches 128.
type ReceiptReferences struct {
	IDs       []string
	Truncated bool
}

func (s *Service) AuditExecutionReceiptReferences(ctx context.Context, ownerID, auditID, executionID, runID string) (map[string]ReceiptReferences, error) {
	if ownerID == "" || auditID == "" || executionID == "" || runID == "" {
		return nil, ErrInvalid
	}
	rows, err := s.pool.Query(ctx, auditExecutionReceiptReferencesSQL, ownerID, auditID, executionID, runID, maxProposalsPerRun+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := map[string]ReceiptReferences{}
	count := 0
	for rows.Next() {
		count++
		if count > maxProposalsPerRun {
			return nil, fmt.Errorf("%w: Run proposal quota", ErrInvalid)
		}
		var taskID, receiptID string
		if err := rows.Scan(&taskID, &receiptID); err != nil {
			return nil, err
		}
		value := result[taskID]
		if len(value.IDs) < auditcontract.MaxFindingReferences {
			value.IDs = append(value.IDs, receiptID)
		} else {
			value.Truncated = true
		}
		result[taskID] = value
	}
	return result, rows.Err()
}
