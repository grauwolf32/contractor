package evalservice

import "github.com/grauwolf32/contractor/internal/config"

type workflowPins struct {
	instructions map[string]any
	tools        map[string]any
	models       map[string]any
	execution    map[string]any
}

func bindingPinValues(s BindingSnapshot, workflows map[string]config.ResolvedWorkflow) map[string]any {
	pins := workflowPins{
		instructions: map[string]any{},
		tools:        map[string]any{},
		models:       map[string]any{},
		execution:    map[string]any{},
	}
	for role, workflow := range workflows {
		for name, stage := range workflow.Stages {
			pins.addStage(role+"/"+name, stage)
		}
	}
	out := map[string]any{
		"instructions":   pins.instructions,
		"tools":          pins.tools,
		"models":         pins.models,
		"sampling":       pins.models,
		"skills":         s.Skills,
		"runtime-config": s.Runtime,
		"execution":      pins.execution,
		"standards":      s.Standards,
	}
	if s.Audit != nil {
		out["inventory"] = s.Audit.Inventory
		out["audit-execution"] = s.Audit.Execution
		out["audit-interaction"] = s.Audit.Interaction
	}
	return out
}

func (p workflowPins) addStage(key string, stage config.ResolvedStage) {
	p.instructions[key] = map[string]any{
		"instructions": stage.Instructions,
		"objective":    stage.Objective,
	}
	p.models[key] = stage.ExecutionConfig
	p.execution[key] = stage.On
	for agent, binding := range stage.Agents {
		agentKey := key + "/" + agent
		p.instructions[agentKey] = binding.Template.Instructions
		p.tools[agentKey] = map[string]any{
			"tools":     binding.Template.Toolsets,
			"execution": binding.Template.Execution,
			"sandbox":   binding.Template.SandboxProfile,
			"runtime":   binding.Template.Runtime,
		}
		p.models[agentKey] = binding.Template.Summarizer
	}
}
