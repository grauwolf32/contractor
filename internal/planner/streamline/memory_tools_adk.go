package streamline

import (
	"fmt"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"
)

func (p *streamlinePlanner) buildMemoryTools(
	state *executionState,
	allowed map[string]struct{},
) ([]tool.Tool, error) {
	result := make([]tool.Tool, 0, len(plannermemory.OperationNames()))
	for _, operation := range plannermemory.OperationNames() {
		bindings := p.memoryBindings(operation)
		if len(bindings) == 0 {
			continue
		}
		if _, collision := allowed[operation]; collision || operation == finishToolName {
			return nil, fmt.Errorf("Planner model-visible tool name %q collides", operation)
		}
		schema, err := memoryToolSchema(operation, bindings, p.profile.routesWorkers)
		if err != nil {
			return nil, fmt.Errorf("build %s schema: %w", operation, err)
		}
		built := &plannerMemoryTool{
			planner: p, state: state, operation: operation,
			description: memoryToolDescription(operation), schema: schema,
			bindings: append([]workerBinding(nil), bindings...), router: p.profile.routesWorkers,
		}
		result = append(result, built)
		allowed[operation] = struct{}{}
	}
	return result, nil
}

func (t *plannerMemoryTool) Name() string        { return t.operation }
func (t *plannerMemoryTool) Description() string { return t.description }
func (t *plannerMemoryTool) IsLongRunning() bool { return false }

func (t *plannerMemoryTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name: t.operation, Description: t.description, ParametersJsonSchema: t.schema,
	}
}

func (t *plannerMemoryTool) ProcessRequest(_ agent.Context, request *model.LLMRequest) error {
	return packLocalTool(request, t, t.Declaration())
}

func (t *plannerMemoryTool) Run(ctx agent.Context, raw any) (map[string]any, error) {
	return t.run(ctx, raw)
}
