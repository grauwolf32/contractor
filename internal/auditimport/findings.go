package auditimport

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// One existing finding page per step. Holds/rejections record progress, so a
// lost response replays original retention and no private cursor ledger exists.
func (i *Importer) retainTaskFindings(ctx context.Context, snapshot auditstore.ReconcileSnapshot, execution auditstore.Execution) (bool, error) {
	if i.findings == nil || execution.RunID == nil {
		return false, nil
	}
	receipts, err := i.findings.ListAuditCollection(ctx, snapshot.Audit.OwnerID, execution.AuditID, *execution.RunID, findingintake.ListQuery{Limit: findingintake.MaxAuditReceiptBatchSize})
	if err != nil {
		return false, err
	}
	if len(receipts) == 0 {
		return false, nil
	}
	pending := make([]findingintake.ImportRequest, 0, len(receipts))
	for _, candidate := range receipts {
		receipt := candidate.Receipt
		if receipt.Origin.Audit == nil || receipt.Origin.Audit.AuditID != execution.AuditID || receipt.Origin.Audit.ExecutionID != execution.ExecutionID || receipt.Origin.Audit.TaskID == "" || receipt.Origin.RunID != *execution.RunID {
			return false, fmt.Errorf("%w: finding Task origin differs", ErrPermanent)
		}
		request := findingintake.ImportRequest{OwnerID: snapshot.Audit.OwnerID, AuditID: execution.AuditID, RunID: *execution.RunID, Proposal: receipt.Proposal.Ref}
		if candidate.ReadFailure != "" {
			if err := i.findings.RejectAuditCollection(ctx, request, candidate.ReadFailure); err != nil {
				if errors.Is(err, findingintake.ErrAuditClosed) {
					return false, nil
				}
				return false, err
			}
		} else {
			pending = append(pending, request)
		}
	}
	if len(pending) > 0 {
		if err := i.findings.RetainAuditCollectionBatch(ctx, pending); err != nil {
			if errors.Is(err, findingintake.ErrAuditClosed) {
				return false, nil
			}
			return false, err
		}
	}
	return true, nil
}

// Enrich only the trusted context with retained receipt identities. Workers
// author conclusions; they never author Finding IDs or trusted source origin.
func (i *Importer) taskFindingContext(ctx context.Context, ownerID string, execution auditstore.Execution, trusted, raw []byte) ([]byte, []byte, error) {
	var contextValue, conclusions map[string]any
	if strictjson.Decode(trusted, &contextValue) != nil || strictjson.Decode(raw, &conclusions) != nil || conclusions["schema"] != "contractor.audit.conclusions.v1" {
		return nil, nil, fmt.Errorf("%w: ordinary conclusions required", ErrPermanent)
	}
	if i.findings == nil {
		return trusted, raw, nil
	}
	references, err := i.findings.AuditExecutionReceiptReferences(ctx, ownerID, execution.AuditID, execution.ExecutionID, *execution.RunID)
	if err != nil {
		return nil, nil, err
	}
	truncated := map[string]bool{}
	for _, value := range contextValue["items"].([]any) {
		member := value.(map[string]any)
		refs := references[member["task_id"].(string)]
		if len(refs.IDs) > 0 {
			member["finding_receipts"] = refs.IDs
		}
		if refs.Truncated {
			truncated[member["item_key"].(string)] = true
		}
	}
	// The full source list remains in Project Findings. State the actual
	// bounded-result omission explicitly instead of silently losing its scope.
	if len(truncated) > 0 {
		items, ok := conclusions["items"].([]any)
		if !ok {
			return nil, nil, fmt.Errorf("%w: invalid conclusions", ErrPermanent)
		}
		for _, value := range items {
			item, ok := value.(map[string]any)
			if !ok {
				return nil, nil, fmt.Errorf("%w: invalid conclusion", ErrPermanent)
			}
			key, ok := item["item_key"].(string)
			if !ok {
				return nil, nil, fmt.Errorf("%w: invalid item key", ErrPermanent)
			}
			if truncated[key] {
				limitations, ok := item["limitations"].([]any)
				if !ok || len(limitations) >= 512 {
					return nil, nil, fmt.Errorf("%w: conclusion limitations bound", ErrPermanent)
				}
				item["limitations"] = append(limitations, "Additional finding sources exceed this result's 128-reference limit; consult Project Finding sources.")
			}
		}
	}
	trusted, err = strictjson.Canonical(contextValue)
	if err != nil {
		return nil, nil, err
	}
	raw, err = strictjson.Canonical(conclusions)
	return trusted, raw, err
}
