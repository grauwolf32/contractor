package scanplan

import (
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
)

func validNucleiJob(job ScanJob) bool {
	arguments := map[string]any{}
	parameters, artifacts := 0, 0
	for name, binding := range job.Execution.Arguments {
		switch binding.Source {
		case "literal":
			arguments[name] = binding.Value
		case "parameter":
			if _, exists := job.Parameters[binding.Name]; !exists && binding.Optional {
				continue
			}
			value, err := binding.ResolveParameter(job.Parameters)
			if err != nil {
				return false
			}
			arguments[name] = value
			parameters++
		case "artifact":
			ref, exists := job.Artifacts[binding.Name]
			if !exists && binding.Optional {
				continue
			}
			if !exists {
				return false
			}
			arguments[name] = ref
			artifacts++
		default:
			return false
		}
	}
	return parameters == len(job.Parameters) && artifacts == len(job.Artifacts) && scan.ValidateNucleiArguments(arguments) == nil
}

func validSQLMapParameters(job ScanJob) bool {
	binding, bound := job.Execution.Arguments["credential_id"]
	value, present := job.Parameters["credential_id"]
	if len(job.Parameters) > 1 || len(job.Parameters) == 1 && !present {
		return false
	}
	if !present {
		return !bound || binding.Source == "literal" || binding.Optional
	}
	return scan.IsScanCredentialBinding(job.Tool, "credential_id", binding) && contracts.ValidConfigID(value)
}
