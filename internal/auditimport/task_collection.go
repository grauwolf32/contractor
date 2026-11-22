package auditimport

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// taskCollectionState carries one bounded terminal collection attempt.
// Store owns the transaction and natural receipt.
type taskCollectionState struct {
	execution      auditstore.Execution
	params         auditstore.CollectParams
	tasks          []auditstore.Task
	partials       partialTaskResults
	findingResults map[string]*auditstore.CanonicalFindingResult
}

func (c *taskCollectionState) commit(ctx context.Context, store Store, code string, retryable bool, result *auditcontract.Material) (bool, error) {
	params := &c.params
	params.Tasks = make([]auditstore.TaskCollection, 0, len(c.tasks))
	for _, task := range c.tasks {
		var gap *string
		if code != "" {
			value := code
			gap = &value
		}
		collection := auditstore.TaskCollection{
			ExecutionItemID: auditdomain.DeterministicID("attempt", c.execution.ExecutionID, task.Document.TaskID),
			TaskID:          task.Document.TaskID,
			Findings:        c.findingResults[task.Document.TaskID],
			Result:          result,
			GapCode:         gap,
			Retryable:       retryable,
		}
		if err := c.partials.selectTask(c.execution, task, &collection, &params.Retained); err != nil {
			return false, err
		}
		params.Tasks = append(params.Tasks, collection)
	}
	if code != "" {
		params.ErrorCode = &code
	}
	_, changed, err := store.Collect(ctx, *params)
	return changed, err
}
