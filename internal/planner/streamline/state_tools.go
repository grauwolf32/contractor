package streamline

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/stateview"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

const (
	getWorkspaceCoverageToolName = "get_workspace_coverage"
	listReadFilesToolName        = "list_read_files"
	listUnreadFilesToolName      = "list_unread_files"
	getWorkerToolUsageToolName   = "get_worker_tool_usage"
)

var stateToolNames = [...]string{
	getWorkspaceCoverageToolName,
	listReadFilesToolName,
	listUnreadFilesToolName,
	getWorkerToolUsageToolName,
}

type stateToolArgs struct {
	WorkerName string
	Cursor     string
	Limit      int
}

type plannerStateTool struct {
	planner     *streamlinePlanner
	state       *executionState
	operation   string
	description string
	schema      *jsonschema.Schema
	bindings    []workerBinding
	router      bool
}

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

func (t *plannerStateTool) ProcessRequest(_ agent.ToolContext, request *model.LLMRequest) error {
	if request == nil {
		return fmt.Errorf("Planner model request is required")
	}
	if request.Tools == nil {
		request.Tools = make(map[string]any)
	}
	if _, duplicate := request.Tools[t.operation]; duplicate {
		return fmt.Errorf("duplicate tool %q", t.operation)
	}
	request.Tools[t.operation] = t
	if request.Config == nil {
		request.Config = &genai.GenerateContentConfig{}
	}
	for _, packed := range request.Config.Tools {
		if packed != nil && packed.FunctionDeclarations != nil {
			packed.FunctionDeclarations = append(packed.FunctionDeclarations, t.Declaration())
			return nil
		}
	}
	request.Config.Tools = append(request.Config.Tools, &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{t.Declaration()},
	})
	return nil
}

func (t *plannerStateTool) Run(
	ctx agent.ToolContext,
	raw any,
) (map[string]any, error) {
	started := time.Now()
	args, code := decodeStateToolArgs(t.operation, t.router, raw)
	binding, bindingOK := t.planner.resolveStateBinding(t.bindings, args.WorkerName)
	if t.router && !bindingOK {
		code = stateview.CodeUnavailable
	}
	safeArguments := stateMetricArguments(t.operation, binding, args)
	if code != "" {
		failure := stateToolFailure(&stateview.Error{Code: code, Retryable: false})
		t.state.recordTool(
			t.operation, safeArguments, false, time.Since(started), 0, &failure,
		)
		return stateFailureResult(failure), nil
	}

	var value any
	var err error
	switch t.operation {
	case getWorkspaceCoverageToolName:
		value, err = t.planner.stateViews.GetWorkspaceCoverage(ctx, binding.logicalName)
	case listReadFilesToolName:
		value, err = t.planner.stateViews.ListReadFiles(
			ctx, binding.logicalName, args.Cursor, args.Limit,
		)
	case listUnreadFilesToolName:
		value, err = t.planner.stateViews.ListUnreadFiles(
			ctx, binding.logicalName, args.Cursor, args.Limit,
		)
	case getWorkerToolUsageToolName:
		value, err = t.planner.stateViews.GetWorkerToolUsage(ctx, binding.logicalName)
	default:
		err = &stateview.Error{Code: stateview.CodeRequestInvalid, Retryable: false}
	}
	if err != nil {
		failure := stateToolFailure(err)
		t.state.recordTool(
			t.operation, safeArguments, false, time.Since(started), 0, &failure,
		)
		return stateFailureResult(failure), nil
	}
	output, encodedSize := stateSuccessResult(t.operation, value)
	addStateResultMetrics(safeArguments, value)
	t.state.recordTool(
		t.operation, safeArguments, true, time.Since(started), encodedSize, nil,
	)
	return output, nil
}

func (p *streamlinePlanner) stateBindings(operation string) []workerBinding {
	result := make([]workerBinding, 0, len(p.workers))
	for _, binding := range p.workers {
		if operation == getWorkerToolUsageToolName || binding.workspaceState {
			result = append(result, binding)
		}
	}
	return result
}

func (p *streamlinePlanner) resolveStateBinding(
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

func decodeStateToolArgs(operation string, router bool, raw any) (stateToolArgs, string) {
	values, ok := raw.(map[string]any)
	if !ok {
		return stateToolArgs{}, stateview.CodeRequestInvalid
	}
	result := stateToolArgs{Limit: 100}
	allowed := map[string]struct{}{}
	if router {
		allowed["worker_name"] = struct{}{}
		worker, present := values["worker_name"]
		if !present {
			return result, stateview.CodeUnavailable
		}
		result.WorkerName, ok = worker.(string)
		if !ok || result.WorkerName == "" {
			return result, stateview.CodeUnavailable
		}
	} else if _, present := values["worker_name"]; present {
		return result, stateview.CodeRequestInvalid
	}
	switch operation {
	case listReadFilesToolName, listUnreadFilesToolName:
		allowed["cursor"] = struct{}{}
		allowed["limit"] = struct{}{}
		if rawCursor, present := values["cursor"]; present {
			result.Cursor, ok = rawCursor.(string)
			if !ok || len(result.Cursor) > 128 {
				return result, stateview.CodeRequestInvalid
			}
		}
		if rawLimit, present := values["limit"]; present {
			result.Limit, ok = stateToolInteger(rawLimit)
			if !ok || result.Limit < 1 || result.Limit > 100 {
				return result, stateview.CodeRequestInvalid
			}
		}
	case getWorkspaceCoverageToolName, getWorkerToolUsageToolName:
	default:
		return result, stateview.CodeRequestInvalid
	}
	for name := range values {
		if _, accepted := allowed[name]; !accepted {
			return result, stateview.CodeRequestInvalid
		}
	}
	return result, ""
}

func stateToolInteger(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return typed, true
	case int32:
		return int(typed), true
	case int64:
		if typed < math.MinInt || typed > math.MaxInt {
			return 0, false
		}
		return int(typed), true
	case float64:
		if math.IsNaN(typed) || math.IsInf(typed, 0) || typed != math.Trunc(typed) ||
			typed < float64(math.MinInt) || typed > float64(math.MaxInt) {
			return 0, false
		}
		return int(typed), true
	case json.Number:
		parsed, err := typed.Int64()
		if err != nil || parsed < math.MinInt || parsed > math.MaxInt {
			return 0, false
		}
		return int(parsed), true
	default:
		return 0, false
	}
}

func stateToolSchema(
	operation string,
	bindings []workerBinding,
	router bool,
) (*jsonschema.Schema, error) {
	properties := map[string]*jsonschema.Schema{}
	required := []string{}
	propertyOrder := []string{}
	if router {
		workers := make([]any, len(bindings))
		for index, binding := range bindings {
			workers[index] = binding.logicalName
		}
		properties["worker_name"] = &jsonschema.Schema{Type: "string", Enum: workers}
		required = append(required, "worker_name")
		propertyOrder = append(propertyOrder, "worker_name")
	}
	switch operation {
	case listReadFilesToolName, listUnreadFilesToolName:
		cursorMaximum := 128
		minimum := float64(1)
		maximum := float64(100)
		properties["cursor"] = &jsonschema.Schema{
			Type: "string", MaxLength: &cursorMaximum, Default: json.RawMessage(`""`),
		}
		properties["limit"] = &jsonschema.Schema{
			Type: "integer", Minimum: &minimum, Maximum: &maximum, Default: json.RawMessage(`100`),
		}
		propertyOrder = append(propertyOrder, "cursor", "limit")
	case getWorkspaceCoverageToolName, getWorkerToolUsageToolName:
	default:
		return nil, fmt.Errorf("unknown Worker State operation %q", operation)
	}
	return &jsonschema.Schema{
		Type: "object", Properties: properties, Required: required,
		AdditionalProperties: &jsonschema.Schema{Not: &jsonschema.Schema{}},
		PropertyOrder:        propertyOrder,
	}, nil
}

func stateMetricArguments(
	operation string,
	binding workerBinding,
	args stateToolArgs,
) map[string]any {
	workerName := binding.logicalName
	if workerName == "" {
		workerName = "invalid"
	}
	result := map[string]any{"workerName": workerName}
	if operation == listReadFilesToolName || operation == listUnreadFilesToolName {
		result["cursorPresent"] = args.Cursor != ""
		result["limit"] = args.Limit
	}
	return result
}

func addStateResultMetrics(arguments map[string]any, value any) {
	switch typed := value.(type) {
	case stateview.WorkspaceCoverage:
		arguments["scopedFiles"] = typed.ScopedFiles
		arguments["readFiles"] = typed.ReadFiles
		arguments["scopeComplete"] = typed.ScopeComplete
		arguments["detailComplete"] = typed.DetailComplete
	case stateview.FilePage:
		arguments["resultCount"] = len(typed.Files)
		arguments["hasNextPage"] = typed.NextCursor != ""
		arguments["complete"] = typed.Complete
	case stateview.WorkerToolUsage:
		arguments["modelCalls"] = typed.ModelCalls
		arguments["toolCalls"] = typed.ToolCalls
		arguments["toolKinds"] = len(typed.Tools)
	}
}

func stateSuccessResult(operation string, value any) (map[string]any, int) {
	field := "result"
	switch operation {
	case getWorkspaceCoverageToolName:
		field = "coverage"
	case listReadFilesToolName, listUnreadFilesToolName:
		field = "page"
	case getWorkerToolUsageToolName:
		field = "usage"
	}
	encoded, _ := json.Marshal(value)
	var projected any
	_ = json.Unmarshal(encoded, &projected)
	result := map[string]any{"ok": true, field: projected}
	resultBytes, _ := json.Marshal(result)
	return result, len(resultBytes)
}

func stateFailureResult(failure planner.Failure) map[string]any {
	return map[string]any{
		"ok": false,
		"error": map[string]any{
			"code": failure.Code, "message": failure.Message, "retryable": failure.Retryable,
		},
	}
}

func stateToolFailure(err error) planner.Failure {
	code := stateview.CodeUnavailable
	retryable := true
	var typed *stateview.Error
	if errors.As(err, &typed) {
		switch typed.Code {
		case stateview.CodeUnavailable:
			code, retryable = typed.Code, typed.Retryable
		case stateview.CodeChanged:
			code, retryable = typed.Code, true
		case stateview.CodeWorkspaceUnavailable,
			stateview.CodeWorkspaceIncomplete,
			stateview.CodeCursorInvalid,
			stateview.CodeRequestInvalid:
			code, retryable = typed.Code, false
		}
	}
	return planner.Failure{
		Code: code, Message: "Worker State projection failed (" + code + ")", Retryable: retryable,
	}
}

func stateToolDescription(operation string) string {
	switch operation {
	case getWorkspaceCoverageToolName:
		return "Inspect bounded content-free workspace coverage for the newest completed Worker subtask. Counts are exact only when the returned completeness flags permit it."
	case listReadFilesToolName:
		return "List model-observed read paths for the newest completed Worker subtask in first-observation order, using only the returned opaque cursor for pagination."
	case listUnreadFilesToolName:
		return "List checkpoint paths not read by the newest completed Worker subtask in lexical order. This fails closed when workspace coverage is incomplete."
	case getWorkerToolUsageToolName:
		return "Inspect content-free model, token and per-tool counts for the newest completed Worker subtask; arguments, results and errors are never returned."
	default:
		return "Inspect a bounded Worker State projection."
	}
}
