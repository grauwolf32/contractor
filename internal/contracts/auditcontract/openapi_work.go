package auditcontract

import (
	"strconv"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type OpenAPIWorkArguments struct {
	Workflow  string
	Selection string
	MaxItems  int
}

// OpenAPIWorkOptions validates the concrete operation's bound domain choices.
// The same rule checks literal configuration and actual Stage arguments.
func OpenAPIWorkOptions(arguments map[string]any) (OpenAPIWorkArguments, error) {
	workflow, workflowOK := arguments["workflow"].(string)
	selection, selectionOK := arguments["selection"].(string)
	budget, budgetOK := arguments["max_items"].(string)
	count, err := strconv.Atoi(budget)
	if len(arguments) != 3 || !workflowOK || !contracts.ValidName(workflow) || !selectionOK || (selection != "operations" && selection != "scan-requests") || !budgetOK || err != nil || strconv.Itoa(count) != budget || count < 1 || count > MaxPreparedWork {
		return OpenAPIWorkArguments{}, contracts.ErrValidation
	}
	return OpenAPIWorkArguments{Workflow: workflow, Selection: selection, MaxItems: count}, nil
}
