package resolved

import (
	"fmt"
	"strings"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// ServerOperationConfig selects Server code invoked directly by Scheduler.
// Arguments use the shared literal/parameter binding contract.
type ServerOperationConfig struct {
	Ref       string                                   `json:"ref" yaml:"ref"`
	Arguments map[string]contracts.ToolArgumentBinding `json:"arguments" yaml:"arguments"`
}

func IsModelFreeStage(stage ResolvedStage) bool {
	return stage.Operation != nil || IsModelFreePlanner(stage.Planner)
}

// ValidateStageExecutor closes the two mutually exclusive execution paths.
// Operation identifiers are registered code contracts, never executable input.
func ValidateStageExecutor(stage ResolvedStage) error {
	if stage.Operation == nil {
		return ValidatePlannerAgents(stage.Planner, len(stage.Agents))
	}
	operation := stage.Operation
	if strings.TrimSpace(stage.Objective) == "" {
		return fmt.Errorf("Server operation objective must be resolved")
	}
	if stage.Planner != (PlannerRef{}) || (operation.Ref != contracts.SitemapOpenAPIJoinOperation && operation.Ref != contracts.SitemapApplyOpenAPIOperation) || operation.Arguments == nil {
		return fmt.Errorf("Stage must select exactly one registered operation or Planner")
	}
	if len(stage.Agents) != 0 || stage.Context.Workspace != nil || stage.Session != contracts.WorkerSessionIsolated || stage.ScanPlan != nil || stage.AuditScan != nil || stage.Instructions != (contracts.ResolvedInstructions{}) {
		return fmt.Errorf("Server operation forbids Workers, shared sessions, workspace and scan policies")
	}
	if err := contracts.ValidateArgumentBindings(operation.Arguments, false); err != nil {
		return err
	}
	if stage.On.Summarized != nil {
		return fmt.Errorf("Server operations cannot summarize")
	}
	for name, binding := range operation.Arguments {
		if name != "origin" && name != "contract" && name != "deployment" {
			return fmt.Errorf("unknown sitemap operation argument %q", name)
		}
		if binding.Source == "literal" {
			if _, ok := binding.Value.(string); !ok {
				return fmt.Errorf("sitemap operation arguments must be strings")
			}
		} else if binding.Parse != "" {
			return fmt.Errorf("sitemap operation arguments must be strings")
		}
	}
	for _, name := range []string{"origin", "contract"} {
		if _, ok := operation.Arguments[name]; !ok {
			return fmt.Errorf("sitemap operation requires argument %q", name)
		}
	}
	if len(stage.Context.Artifacts) != 2 || !stage.Context.Artifacts["sitemap"].Required || !stage.Context.Artifacts["openapi"].Required {
		return fmt.Errorf("sitemap operation requires exactly sitemap and openapi context inputs")
	}
	if stage.Result.Artifacts == nil || len(stage.Result.Artifacts) > 2 {
		return fmt.Errorf("sitemap operation results permit only explicit sitemap and report slots")
	}
	for name := range stage.Result.Artifacts {
		if name != "sitemap" && name != "report" {
			return fmt.Errorf("unknown sitemap operation output %q", name)
		}
	}
	for _, slot := range stage.Result.Artifacts {
		if slot.From == nil || !contracts.ValidName(slot.From.Namespace) || !contracts.ValidName(slot.From.Name) || artifactpolicy.IsPurposeReservedNamespace(slot.From.Namespace) || artifactpolicy.IsReservedMemoryBinding(slot.From.Namespace, slot.From.Name) {
			return fmt.Errorf("Server operation output requires an ordinary explicit artifact binding")
		}
	}
	return nil
}
