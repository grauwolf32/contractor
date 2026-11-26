package streamline

import (
	"fmt"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"
)

func (p *streamlinePlanner) buildStateTools(
	state *executionState,
	allowed map[string]struct{},
) ([]tool.Tool, error) {
	result := make([]tool.Tool, 0, len(stateToolNames))
	for _, operation := range stateToolNames {
		bindings := p.stateBindings(operation)
		if len(bindings) == 0 {
			continue
		}
		if _, collision := allowed[operation]; collision || operation == finishToolName {
			return nil, fmt.Errorf("Planner model-visible tool name %q collides", operation)
		}
		schema, err := stateToolSchema(operation, bindings, p.profile.routesWorkers)
		if err != nil {
			return nil, fmt.Errorf("build %s schema: %w", operation, err)
		}
		result = append(result, &plannerStateTool{
			planner: p, state: state, operation: operation,
			description: stateToolDescription(operation), schema: schema,
			bindings: append([]workerBinding(nil), bindings...), router: p.profile.routesWorkers,
		})
		allowed[operation] = struct{}{}
	}
	return result, nil
}

func (t *plannerStateTool) Name() string        { return t.operation }
func (t *plannerStateTool) Description() string { return t.description }
func (t *plannerStateTool) IsLongRunning() bool { return false }

func (t *plannerStateTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name: t.operation, Description: t.description, ParametersJsonSchema: t.schema,
	}
}

func (t *plannerStateTool) ProcessRequest(_ agent.Context, request *model.LLMRequest) error {
	return packLocalTool(request, t, t.Declaration())
}

func (t *plannerStateTool) Run(ctx agent.Context, raw any) (map[string]any, error) {
	return t.run(ctx, raw)
}
