package streamline

import (
	"fmt"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"
)

// packLocalTool keeps the concrete tool as the invocation target. Using
// functiontool.New would validate its schema before Run and replace the
// domain's bounded model-visible errors with framework validation errors.
func packLocalTool(request *model.LLMRequest, current tool.Tool, declaration *genai.FunctionDeclaration) error {
	if request == nil {
		return fmt.Errorf("Planner model request is required")
	}
	if request.Tools == nil {
		request.Tools = make(map[string]any)
	}
	if _, duplicate := request.Tools[current.Name()]; duplicate {
		return fmt.Errorf("duplicate tool %q", current.Name())
	}
	request.Tools[current.Name()] = current
	if request.Config == nil {
		request.Config = &genai.GenerateContentConfig{}
	}
	for _, packed := range request.Config.Tools {
		if packed != nil && packed.FunctionDeclarations != nil {
			packed.FunctionDeclarations = append(packed.FunctionDeclarations, declaration)
			return nil
		}
	}
	request.Config.Tools = append(request.Config.Tools, &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{declaration},
	})
	return nil
}

func (p *streamlinePlanner) resolveWorkerBinding(
	bindings []workerBinding,
	workerName string,
) (workerBinding, bool) {
	if !p.profile.routesWorkers {
		if len(bindings) == 1 && workerName == "" {
			return bindings[0], true
		}
		return workerBinding{}, false
	}
	for _, binding := range bindings {
		if binding.logicalName == workerName {
			return binding, true
		}
	}
	return workerBinding{}, false
}
