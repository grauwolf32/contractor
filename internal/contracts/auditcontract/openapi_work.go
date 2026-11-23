package auditcontract

import (
	"strconv"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type OpenAPIWorkArguments struct {
	Workflow   string
	Selection  string
	MaxItems   int
	Scanner    string
	Parameters map[string]string
}

// OpenAPIWorkOptions validates the concrete operation's bound domain choices.
// The same rule checks literal configuration and actual Stage arguments.
func OpenAPIWorkOptions(arguments map[string]any) (OpenAPIWorkArguments, error) {
	workflow, workflowOK := arguments["workflow"].(string)
	selection, selectionOK := arguments["selection"].(string)
	budget, budgetOK := arguments["max_items"].(string)
	count, err := strconv.Atoi(budget)
	if len(arguments) < 3 || len(arguments) > 8 || !workflowOK || !contracts.ValidName(workflow) || !selectionOK || (selection != "operations" && selection != "scan-requests") || !budgetOK || err != nil || strconv.Itoa(count) != budget || count < 1 || count > MaxPreparedWork {
		return OpenAPIWorkArguments{}, contracts.ErrValidation
	}
	result := OpenAPIWorkArguments{Workflow: workflow, Selection: selection, MaxItems: count, Parameters: map[string]string{}}
	for name, raw := range arguments {
		switch name {
		case "workflow", "selection", "max_items":
			continue
		case "scanner", "server", "operations", "test_parameters", "credential_id":
			value, ok := raw.(string)
			if !ok {
				return OpenAPIWorkArguments{}, contracts.ErrValidation
			}
			if name == "scanner" {
				result.Scanner = value
			} else {
				result.Parameters[name] = value
			}
		default:
			return OpenAPIWorkArguments{}, contracts.ErrValidation
		}
	}
	if ValidateParameters(result.Parameters) != nil || selection == "scan-requests" && result.Scanner != "sqlmap" && result.Scanner != "nuclei" {
		return OpenAPIWorkArguments{}, contracts.ErrValidation
	}
	if selection == "operations" {
		if result.Scanner != "" {
			return OpenAPIWorkArguments{}, contracts.ErrValidation
		}
		for _, value := range result.Parameters {
			if value != "" {
				return OpenAPIWorkArguments{}, contracts.ErrValidation
			}
		}
	}
	return result, nil
}
