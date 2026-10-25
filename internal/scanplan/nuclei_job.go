package scanplan

import "github.com/grauwolf32/contractor/internal/contracts/scan"

func validNucleiJob(job ScanJob) bool {
	arguments := map[string]any{}
	parameters, artifacts := 0, 0
	for name, binding := range job.Execution.Arguments {
		switch binding.Source {
		case "literal":
			arguments[name] = binding.Value
		case "parameter":
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
