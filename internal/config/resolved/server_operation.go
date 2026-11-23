package resolved

import (
	"fmt"
	"strconv"
	"strings"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/contracts/memorycontract"
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
	if stage.Planner != (PlannerRef{}) || operation.Arguments == nil {
		return fmt.Errorf("Stage must select exactly one registered operation or Planner")
	}
	if len(stage.Agents) != 0 || stage.Context.Workspace != nil || stage.Session != contracts.WorkerSessionIsolated || stage.ScanPlan != nil || stage.ScanPreparation != nil || stage.Instructions != (contracts.ResolvedInstructions{}) {
		return fmt.Errorf("Server operation forbids Workers, shared sessions, workspace and scan policies")
	}
	if err := contracts.ValidateArgumentBindings(operation.Arguments, false); err != nil {
		return err
	}
	if stage.On.Summarized != nil {
		return fmt.Errorf("Server operations cannot summarize")
	}
	for _, slot := range stage.Result.Artifacts {
		if slot.From == nil || !contracts.ValidName(slot.From.Namespace) || !contracts.ValidName(slot.From.Name) || artifactpolicy.IsPurposeReservedNamespace(slot.From.Namespace) || artifactpolicy.IsReservedMemoryBinding(slot.From.Namespace, slot.From.Name) {
			return fmt.Errorf("Server operation output requires an ordinary explicit artifact binding")
		}
	}
	if operation.Ref == contracts.PublishFindingsOperation {
		pair := len(stage.Context.Artifacts) == 2 && stage.Context.Artifacts["candidates"].Required && stage.Context.Artifacts["evidence"].Required
		set := len(stage.Context.Artifacts) == 1 && stage.Context.Artifacts["materials"].Required
		if len(operation.Arguments) != 0 || (!pair && !set) || stage.Result.Artifacts == nil || len(stage.Result.Artifacts) != 0 {
			return fmt.Errorf("Finding publication requires exactly candidates and evidence inputs and no Artifact outputs")
		}
		return nil
	}
	if operation.Ref == contracts.ArtifactCopyOperation {
		if len(operation.Arguments) != 0 || len(stage.Context.Artifacts) != 1 || !stage.Context.Artifacts["source"].Required || len(stage.Result.Artifacts) != 1 || !stage.Result.Artifacts["copied"].Required {
			return fmt.Errorf("Artifact copy requires exactly source and copied slots and no arguments")
		}
		source, output := stage.Context.Artifacts["source"], stage.Result.Artifacts["copied"].From
		if artifactpolicy.IsModelHiddenBinding(source.Namespace, source.Name) || artifactpolicy.IsModelHiddenBinding(output.Namespace, output.Name) {
			return fmt.Errorf("Artifact copy requires ordinary visible source and output bindings")
		}
		return nil
	}
	if operation.Ref == contracts.MemorySnapshotOperation {
		return validateMemorySnapshotStage(stage)
	}
	if operation.Ref == contracts.OpenAPIPrepareWorkOperation {
		return validateOpenAPIWorkStage(stage)
	}
	if operation.Ref != contracts.SitemapOpenAPIJoinOperation && operation.Ref != contracts.SitemapApplyOpenAPIOperation {
		return fmt.Errorf("unknown Server operation")
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
	return nil
}

func validateOpenAPIWorkStage(stage ResolvedStage) error {
	values := map[string]any{"workflow": "check", "selection": "operations", "max_items": strconv.Itoa(auditcontract.MaxPreparedWork)}
	if len(stage.Operation.Arguments) < len(values) || len(stage.Operation.Arguments) > 8 {
		return fmt.Errorf("OpenAPI preparation requires workflow, selection and max_items arguments")
	}
	for name, binding := range stage.Operation.Arguments {
		if name == "scanner" || name == "server" || name == "operations" || name == "test_parameters" || name == "credential_id" {
			values[name] = ""
		}
		if _, exists := values[name]; !exists || binding.Parse != "" {
			return fmt.Errorf("invalid OpenAPI preparation argument")
		}
		if binding.Source == "literal" {
			values[name] = binding.Value
		}
	}
	_, err := auditcontract.OpenAPIWorkOptions(values)
	if err != nil {
		return fmt.Errorf("invalid OpenAPI preparation choices")
	}
	if len(stage.Context.Artifacts) < 1 || len(stage.Context.Artifacts) > 2 || !stage.Context.Artifacts["openapi"].Required {
		return fmt.Errorf("OpenAPI preparation requires an exact openapi context input")
	}
	for name := range stage.Context.Artifacts {
		if name != "openapi" && name != "request_values" {
			return fmt.Errorf("unknown OpenAPI preparation context input")
		}
	}
	output, exists := stage.Result.Artifacts["work"]
	if !exists || len(stage.Result.Artifacts) != 1 || !output.Required || len(output.MediaTypes) != 1 || output.MediaTypes[0] != "application/json" {
		return fmt.Errorf("OpenAPI preparation requires one JSON work output")
	}
	return nil
}

func validateMemorySnapshotStage(stage ResolvedStage) error {
	values := map[string]any{"namespace": "worker", "notes": "selected"}
	if len(stage.Operation.Arguments) != 2 || len(stage.Context.Artifacts) != 0 {
		return fmt.Errorf("Memory snapshot requires namespace and notes arguments and no context artifacts")
	}
	for name, binding := range stage.Operation.Arguments {
		if _, ok := values[name]; !ok || binding.Parse != "" {
			return fmt.Errorf("invalid Memory snapshot argument")
		}
		if binding.Source == "literal" {
			values[name] = binding.Value
		}
	}
	if _, err := memorycontract.SnapshotOptions(values); err != nil {
		return fmt.Errorf("invalid Memory snapshot selection")
	}
	output, exists := stage.Result.Artifacts["snapshot"]
	if !exists || len(stage.Result.Artifacts) != 1 || !output.Required || len(output.MediaTypes) != 1 || output.MediaTypes[0] != "application/json" || output.From == nil || artifactpolicy.IsModelHiddenBinding(output.From.Namespace, output.From.Name) {
		return fmt.Errorf("Memory snapshot requires one ordinary JSON snapshot output")
	}
	return nil
}
