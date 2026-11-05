package auditstoretest

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// Items inspects historical Round fixtures. There is no public item projection;
// production common research reads only Tasks and exact materials.
func (s *Store) Items(ctx context.Context, ownerID, auditID string) ([]auditstore.Item, error) {
	if _, err := s.Get(ctx, ownerID, auditID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, testItemsSQL, auditID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []auditstore.Item{}
	for rows.Next() {
		var item auditstore.Item
		var task, origin, accepted, dependencies json.RawMessage
		var digest *string
		if err := rows.Scan(&item.ItemID, &item.AuditID, &item.RoundID, &item.ItemKey, &item.Ordinal, &item.Kind, &item.SubjectKey, &task, &item.Task.Digest, &origin, &item.WorkflowRole, &item.State, &item.FinalDisposition, &accepted, &digest, &item.LastExecutionItemID, &item.CreatedAt, &item.UpdatedAt, &dependencies); err != nil {
			return nil, err
		}
		if err := strictjson.Decode(task, &item.Task.Ref); err != nil {
			return nil, err
		}
		if err := strictjson.Decode(origin, &item.Origin); err != nil {
			return nil, err
		}
		if accepted != nil {
			item.AcceptedResult = &auditstore.ExactArtifact{Digest: *digest}
			if err := strictjson.Decode(accepted, &item.AcceptedResult.Ref); err != nil {
				return nil, err
			}
		}
		if dependencies != nil {
			value, err := auditdomain.DecodeRetainedDependencies(dependencies)
			if err != nil {
				return nil, err
			}
			item.Dependencies = &value
		}
		result = append(result, item)
	}
	return result, rows.Err()
}
