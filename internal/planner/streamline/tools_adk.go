package streamline

// ADK bindings adapt the Server-owned operations to the live model loop.
// Only this boundary changes SDK control actions; the operations use ordinary
// contexts so their persistence and dispatch rules need no ADK session.

import (
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/grauwolf32/contractor/internal/planner"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

func (p *streamlinePlanner) buildTools(
	state *executionState,
	identity planner.SessionIdentity,
) ([]tool.Tool, map[string]struct{}, error) {
	result := make([]tool.Tool, 0, 8+len(p.workers)*2)
	allowed := make(map[string]struct{}, 8+len(p.workers)*2)
	addSubtask, err := functiontool.New(functiontool.Config{
		Name:        addSubtaskToolName,
		Description: "Append one bounded immutable subtask to the ordered Stage plan. Objective and instructions are stored once; later Worker dispatches identify this exact work only by subtask_id.",
	}, func(ctx agent.Context, args addSubtaskArgs) (plannerPlanOutput, error) {
		return p.addSubtask(ctx, state, identity, args), nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("build add_subtask tool: %w", err)
	}
	listSubtasks, err := functiontool.New(functiontool.Config{
		Name:        listSubtasksToolName,
		Description: "Return the current bounded ordered subtask plan, including adapter-controlled status and the exact current subtask_id.",
		InputSchema: &jsonschema.Schema{
			Type: "object", Properties: map[string]*jsonschema.Schema{},
			AdditionalProperties: &jsonschema.Schema{Not: &jsonschema.Schema{}},
		},
	}, func(ctx agent.Context, _ listSubtasksArgs) (plannerPlanOutput, error) {
		return p.listSubtasks(ctx, state), nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("build list_subtasks tool: %w", err)
	}
	result = append(result, addSubtask, listSubtasks)
	allowed[addSubtaskToolName] = struct{}{}
	allowed[listSubtasksToolName] = struct{}{}
	var executeCurrentSubtask tool.Tool
	if p.profile.routesWorkers {
		schema, schemaErr := routerExecuteSchema(p.workers)
		if schemaErr != nil {
			return nil, nil, fmt.Errorf("build Router execute schema: %w", schemaErr)
		}
		executeCurrentSubtask, err = functiontool.New(functiontool.Config{
			Name:        executeCurrentSubtaskToolName,
			Description: "Execute the exact current stored subtask with one selected immutable logical Worker. Server supplies the complete StageContext; subtask_id and worker_name are the only arguments.",
			InputSchema: schema,
		}, func(ctx agent.Context, args routerWorkerCallArgs) (workerCallOutput, error) {
			result, stop := p.routeWorker(ctx, state, identity, args)
			if stop {
				ctx.Actions().SkipSummarization = true
			}
			return result, nil
		})
	} else {
		binding := p.workers[0]
		executeCurrentSubtask, err = functiontool.New(functiontool.Config{
			Name: executeCurrentSubtaskToolName,
			Description: fmt.Sprintf(
				"Execute the exact current stored subtask with the sole prepared logical Worker %q (%s). Server supplies the complete immutable StageContext; subtask_id is the only argument.",
				binding.logicalName, binding.description,
			),
		}, func(ctx agent.Context, args workerCallArgs) (workerCallOutput, error) {
			result, stop := p.callWorker(ctx, state, identity, binding, args.SubtaskID)
			if stop {
				ctx.Actions().SkipSummarization = true
			}
			return result, nil
		})
	}
	if err != nil {
		return nil, nil, fmt.Errorf("build execute_current_subtask tool: %w", err)
	}
	result = append(result, executeCurrentSubtask)
	allowed[executeCurrentSubtaskToolName] = struct{}{}
	stateTools, err := p.buildStateTools(state, allowed)
	if err != nil {
		return nil, nil, err
	}
	result = append(result, stateTools...)
	memoryTools, err := p.buildMemoryTools(state, allowed)
	if err != nil {
		return nil, nil, err
	}
	result = append(result, memoryTools...)
	if p.compaction != nil {
		history, err := p.buildHistoryTool(state, identity.SessionID)
		if err != nil {
			return nil, nil, fmt.Errorf("build %s tool: %w", readContextHistoryToolName, err)
		}
		result = append(result, history)
		allowed[readContextHistoryToolName] = struct{}{}
	}
	finish, err := functiontool.New(functiontool.Config{
		Name:        finishToolName,
		Description: "Finish the Stage with a succeeded or failed candidate. A failed candidate requires a safe error; a succeeded candidate forbids one. Every artifact names a declared result slot and a namespace/name material from accepted Worker results or trusted Stage inputs. The adapter supplies its exact revision. Workflow Scheduler remains the acceptance and transition owner.",
	}, func(ctx agent.Context, args finishArgs) (completionToolOutput, error) {
		result := p.finish(ctx, state, identity, args)
		if result.Accepted {
			ctx.Actions().SkipSummarization = true
			ctx.Actions().Escalate = true
		}
		return result, nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("build finish tool: %w", err)
	}
	result = append(result, finish)
	allowed[finishToolName] = struct{}{}
	return result, allowed, nil
}
