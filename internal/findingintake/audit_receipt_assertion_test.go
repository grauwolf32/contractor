package findingintake

import (
	"context"
	"testing"
)

// Inspect one receipt through the public inbox's actual bounded reader.
func readOneAuditReceipt(t testing.TB, service *Service, ctx context.Context, owner, audit, id string) (Receipt, error) {
	t.Helper()
	query := ListQuery{Limit: 200}
	for range 1024 {
		page, err := service.ListAuditInbox(ctx, owner, audit, query)
		if err != nil {
			return Receipt{}, err
		}
		for _, receipt := range page {
			if receipt.ReceiptID == id {
				return receipt, nil
			}
		}
		if len(page) < query.Limit {
			return Receipt{}, ErrNotFound
		}
		last := page[len(page)-1]
		query.AfterCreatedAt = &last.CreatedAt
		query.AfterReceiptID = last.ReceiptID
	}
	t.Fatal("fixture inbox exceeded its bounded page budget")
	return Receipt{}, ErrInvalid
}
