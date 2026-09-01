package streamline

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/planner"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

type memoryToolArgs struct {
	WorkerName  string   `json:"worker_name,omitempty"`
	Name        string   `json:"name,omitempty"`
	Content     string   `json:"content,omitempty"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
}

type plannerMemoryTool struct {
	planner     *streamlinePlanner
	state       *executionState
	operation   string
	description string
	schema      *jsonschema.Schema
	bindings    []workerBinding
	router      bool
}

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

// ProcessRequest deliberately mirrors ADK's function-tool packing without
// delegating invocation to functiontool.New. That helper validates the
// advertised JSON Schema before calling the handler and would expose the
// framework's open-ended validation errors instead of Memory's closed error
// vocabulary. The declaration remains exact; Run owns bounded validation.
func (t *plannerMemoryTool) ProcessRequest(_ agent.ToolContext, request *model.LLMRequest) error {
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

func (t *plannerMemoryTool) Run(
	ctx agent.ToolContext,
	raw any,
) (map[string]any, error) {
	args, code := decodeMemoryToolArgs(t.operation, t.router, raw)
	// Router authority is decided before the remaining semantic shape. This
	// keeps an absent, stale or operation-ineligible logical Worker in the
	// memory_forbidden class even when the same model call also contains a
	// malformed note argument.
	if t.router && args.WorkerName != "" {
		if _, allowed := t.planner.resolveMemoryBinding(t.bindings, args.WorkerName); !allowed {
			code = plannermemory.CodeForbidden
		}
	}
	if code != "" {
		started := time.Now()
		binding, _ := t.planner.resolveMemoryBinding(t.bindings, args.WorkerName)
		failure := planner.Failure{
			Code: code, Message: "Memory operation failed (" + code + ")", Retryable: false,
		}
		t.state.recordTool(
			t.operation, memoryMetricArguments(t.operation, binding, args), false,
			time.Since(started), 0, &failure,
		)
		return memoryFailureResult(failure), nil
	}
	return t.planner.callMemoryTool(ctx, t.state, t.operation, t.bindings, args), nil
}

func (p *streamlinePlanner) memoryBindings(operation string) []workerBinding {
	result := make([]workerBinding, 0, len(p.workers))
	for _, binding := range p.workers {
		if _, selected := binding.memoryTools[operation]; selected {
			result = append(result, binding)
		}
	}
	return result
}

func (p *streamlinePlanner) callMemoryTool(
	ctx agent.ToolContext,
	state *executionState,
	operation string,
	bindings []workerBinding,
	args memoryToolArgs,
) map[string]any {
	started := time.Now()
	binding, ok := p.resolveMemoryBinding(bindings, args.WorkerName)
	safeArguments := memoryMetricArguments(operation, binding, args)
	if !ok || binding.memory == nil {
		failure := planner.Failure{
			Code: plannermemory.CodeForbidden, Message: "Memory operation is not selected for that logical Worker",
			Retryable: false,
		}
		state.recordTool(operation, safeArguments, false, time.Since(started), 0, &failure)
		return memoryFailureResult(failure)
	}

	var output map[string]any
	var err error
	switch operation {
	case plannermemory.ToolListMemories:
		var memories []plannermemory.Preview
		memories, err = binding.memory.ListMemories(ctx)
		if err == nil {
			output = memoryListResult(memories)
		}
	case plannermemory.ToolReadMemory:
		var note plannermemory.Note
		note, err = binding.memory.ReadMemory(ctx, args.Name)
		if err == nil {
			output = memoryNoteResult(note)
		}
	case plannermemory.ToolWriteMemory:
		var note plannermemory.Note
		note, err = binding.memory.WriteMemory(
			ctx, args.Name, args.Content, args.Description, args.Tags,
		)
		if err == nil {
			output = memoryNoteResult(note)
		}
	case plannermemory.ToolAppendMemory:
		var note plannermemory.Note
		note, err = binding.memory.AppendMemory(ctx, args.Name, args.Content)
		if err == nil {
			output = memoryNoteResult(note)
		}
	case plannermemory.ToolSearchMemory:
		var memories []plannermemory.Preview
		memories, err = binding.memory.SearchMemory(ctx, args.Tags)
		if err == nil {
			output = memoryListResult(memories)
		}
	case plannermemory.ToolListMemoryTags:
		var tags []string
		tags, err = binding.memory.ListMemoryTags(ctx)
		if err == nil {
			output = map[string]any{"result": tags}
		}
	default:
		err = &plannermemory.ToolError{Code: plannermemory.CodeForbidden}
	}
	if err != nil {
		failure := memoryFailure(err)
		state.recordTool(operation, safeArguments, false, time.Since(started), 0, &failure)
		return memoryFailureResult(failure)
	}
	encoded, _ := json.Marshal(output)
	state.recordTool(operation, safeArguments, true, time.Since(started), len(encoded), nil)
	return output
}

func memoryNoteResult(note plannermemory.Note) map[string]any {
	encoded, _ := json.Marshal(note)
	var result map[string]any
	_ = json.Unmarshal(encoded, &result)
	return result
}

func memoryListResult(notes []plannermemory.Preview) map[string]any {
	encoded, _ := json.Marshal(notes)
	var result []any
	_ = json.Unmarshal(encoded, &result)
	return map[string]any{"result": result}
}

func memoryFailureResult(failure planner.Failure) map[string]any {
	return map[string]any{"error": map[string]any{
		"code": failure.Code, "message": failure.Message, "retryable": failure.Retryable,
	}}
}

func decodeMemoryToolArgs(operation string, router bool, raw any) (memoryToolArgs, string) {
	values, ok := raw.(map[string]any)
	if !ok {
		return memoryToolArgs{}, plannermemory.CodeInvalid
	}
	var result memoryToolArgs
	allowed := map[string]struct{}{}
	if router {
		allowed["worker_name"] = struct{}{}
		worker, present := values["worker_name"]
		if !present {
			return result, plannermemory.CodeForbidden
		}
		result.WorkerName, ok = worker.(string)
		if !ok || result.WorkerName == "" {
			return result, plannermemory.CodeForbidden
		}
	} else if _, present := values["worker_name"]; present {
		return result, plannermemory.CodeInvalid
	}
	requireString := func(name string) bool {
		allowed[name] = struct{}{}
		value, present := values[name]
		if !present {
			return false
		}
		parsed, valid := value.(string)
		if !valid {
			return false
		}
		switch name {
		case "name":
			result.Name = parsed
		case "content":
			result.Content = parsed
		}
		return true
	}
	parseTags := func(required bool) bool {
		allowed["tags"] = struct{}{}
		value, present := values["tags"]
		if !present {
			return !required
		}
		parsed, valid := stringArray(value)
		if !valid {
			return false
		}
		result.Tags = parsed
		return true
	}
	valid := true
	switch operation {
	case plannermemory.ToolListMemories, plannermemory.ToolListMemoryTags:
	case plannermemory.ToolReadMemory:
		valid = requireString("name")
	case plannermemory.ToolWriteMemory:
		valid = requireString("name") && requireString("content")
		allowed["description"] = struct{}{}
		if value, present := values["description"]; present {
			result.Description, ok = value.(string)
			valid = valid && ok
		}
		valid = valid && parseTags(false)
	case plannermemory.ToolAppendMemory:
		valid = requireString("name") && requireString("content")
	case plannermemory.ToolSearchMemory:
		valid = parseTags(true)
	default:
		return result, plannermemory.CodeForbidden
	}
	for name := range values {
		if _, accepted := allowed[name]; !accepted {
			valid = false
		}
	}
	if !valid {
		return result, plannermemory.CodeInvalid
	}
	return result, ""
}

func stringArray(value any) ([]string, bool) {
	switch typed := value.(type) {
	case []string:
		return append([]string(nil), typed...), true
	case []any:
		result := make([]string, len(typed))
		for index, item := range typed {
			var ok bool
			result[index], ok = item.(string)
			if !ok {
				return nil, false
			}
		}
		return result, true
	default:
		return nil, false
	}
}

func (p *streamlinePlanner) resolveMemoryBinding(
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

func memoryFailure(err error) planner.Failure {
	var bounded *plannermemory.ToolError
	if !errors.As(err, &bounded) {
		bounded = &plannermemory.ToolError{Code: plannermemory.CodeUnavailable, Retryable: true}
	}
	return planner.Failure{
		Code: bounded.Code, Message: "Memory operation failed (" + bounded.Code + ")",
		Retryable: bounded.Retryable,
	}
}

func memoryMetricArguments(
	operation string,
	binding workerBinding,
	args memoryToolArgs,
) map[string]any {
	result := map[string]any{}
	if binding.logicalName != "" {
		result["workerName"] = binding.logicalName
	} else {
		result["workerName"] = "invalid"
	}
	switch operation {
	case plannermemory.ToolReadMemory:
		result["noteName"] = safeMemoryName(args.Name)
	case plannermemory.ToolWriteMemory:
		result["noteName"] = safeMemoryName(args.Name)
		result["contentBytes"] = len([]byte(args.Content))
		result["descriptionBytes"] = len([]byte(args.Description))
		result["tagCount"] = len(args.Tags)
	case plannermemory.ToolAppendMemory:
		result["noteName"] = safeMemoryName(args.Name)
		result["contentBytes"] = len([]byte(args.Content))
	case plannermemory.ToolSearchMemory:
		result["tagCount"] = len(args.Tags)
	}
	return result
}

func safeMemoryName(value string) string {
	if _, err := plannermemory.ArtifactName(value); err != nil {
		return "invalid"
	}
	return value
}

func memoryToolSchema(
	operation string,
	bindings []workerBinding,
	router bool,
) (*jsonschema.Schema, error) {
	properties := map[string]*jsonschema.Schema{}
	required := make([]string, 0, 3)
	propertyOrder := make([]string, 0, 5)
	if router {
		workers := make([]any, len(bindings))
		for index, binding := range bindings {
			workers[index] = binding.logicalName
		}
		properties["worker_name"] = &jsonschema.Schema{Type: "string", Enum: workers}
		required = append(required, "worker_name")
		propertyOrder = append(propertyOrder, "worker_name")
	}
	addName := func() {
		maximum := plannermemory.MaximumNameBytes
		properties["name"] = &jsonschema.Schema{
			Type: "string", Pattern: `^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$`, MaxLength: &maximum,
		}
		required = append(required, "name")
		propertyOrder = append(propertyOrder, "name")
	}
	addContent := func() {
		minimum := 1
		properties["content"] = &jsonschema.Schema{Type: "string", MinLength: &minimum}
		required = append(required, "content")
		propertyOrder = append(propertyOrder, "content")
	}
	addTags := func(requiredTags bool) {
		itemMaximum := plannermemory.MaximumTagBytes
		properties["tags"] = &jsonschema.Schema{
			Type: "array", Items: &jsonschema.Schema{
				Type: "string", Pattern: `^[a-z][a-z0-9_-]*$`, MaxLength: &itemMaximum,
			},
		}
		if requiredTags {
			minimum := 1
			maximum := plannermemory.MaximumTags
			properties["tags"].MinItems = &minimum
			properties["tags"].MaxItems = &maximum
			properties["tags"].UniqueItems = true
			required = append(required, "tags")
		} else {
			properties["tags"].Default = json.RawMessage(`[]`)
		}
		propertyOrder = append(propertyOrder, "tags")
	}
	switch operation {
	case plannermemory.ToolListMemories, plannermemory.ToolListMemoryTags:
	case plannermemory.ToolReadMemory:
		addName()
	case plannermemory.ToolWriteMemory:
		addName()
		addContent()
		descriptionMaximum := plannermemory.MaximumDescription
		properties["description"] = &jsonschema.Schema{
			Type: "string", MaxLength: &descriptionMaximum, Default: json.RawMessage(`""`),
		}
		propertyOrder = append(propertyOrder, "description")
		addTags(false)
	case plannermemory.ToolAppendMemory:
		addName()
		addContent()
	case plannermemory.ToolSearchMemory:
		addTags(true)
	default:
		return nil, fmt.Errorf("unknown Memory operation %q", operation)
	}
	return &jsonschema.Schema{
		Type: "object", Properties: properties, Required: required,
		AdditionalProperties: &jsonschema.Schema{Not: &jsonschema.Schema{}},
		PropertyOrder:        propertyOrder,
	}, nil
}

func memoryToolDescription(operation string) string {
	switch operation {
	case plannermemory.ToolListMemories:
		return "List current shared Memory note previews without their content."
	case plannermemory.ToolReadMemory:
		return "Read one current shared Memory note by its exact logical name."
	case plannermemory.ToolWriteMemory:
		return "Create or replace one shared Memory note; replacement preserves creation order."
	case plannermemory.ToolAppendMemory:
		return "Append one newline and non-empty content to an existing shared Memory note."
	case plannermemory.ToolSearchMemory:
		return "List previews of shared Memory notes matching any supplied tag."
	case plannermemory.ToolListMemoryTags:
		return "List the current unique shared Memory tags in lexical order."
	default:
		return "Use selected shared Memory."
	}
}
