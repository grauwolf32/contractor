package contracts

import (
	"encoding/json"
	"math"
	"regexp"
	"strconv"
	"strings"

	"go.yaml.in/yaml/v4"
)

var toolArgumentName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]{0,63}$`)

// Preserve forbidden-field presence: explicit null is not model absence.
func (t *ResolvedAgentTemplate) UnmarshalJSON(data []byte) error {
	type wire ResolvedAgentTemplate
	var value wire
	fields, err := DecodeStrictObject(data, &value)
	if err != nil {
		return err
	}
	if ResolvedAgentTemplate(value).IsToolWorker() {
		for _, name := range []string{"instructions", "modelPolicy", "summarizer"} {
			if fields[name] != nil {
				return Invalidf("tool@1 forbids %s", name)
			}
		}
	} else if fields["execution"] != nil {
		return Invalidf("execution is only supported by tool@1")
	}
	*t = ResolvedAgentTemplate(value)
	return nil
}

type ToolArgumentBinding struct {
	Source  string `json:"source" yaml:"source" strictjson:"required"`
	Name    string `json:"name,omitempty" yaml:"name,omitempty"`
	Value   any    `json:"value,omitempty" yaml:"value,omitempty"`
	Parse   string `json:"parse,omitempty" yaml:"parse,omitempty"`
	Default any    `json:"default,omitempty" yaml:"default,omitempty"`
}

func (b *ToolArgumentBinding) UnmarshalJSON(data []byte) error {
	type plain ToolArgumentBinding
	var value plain
	fields, err := DecodeStrictObject(data, &value)
	if err != nil {
		return err
	}
	if value.Source == "parameter" {
		if fields["name"] == nil || fields["value"] != nil ||
			(fields["default"] != nil && value.Default == nil) ||
			(fields["parse"] != nil && value.Parse == "") {
			return Invalidf("parameter tool argument requires name and forbids null default or value")
		}
	} else if len(fields) != 2 || fields["source"] == nil ||
		(value.Source == "literal" && fields["value"] == nil) ||
		(value.Source != "literal" && fields["name"] == nil) {
		return Invalidf("tool argument must select exactly one source and its payload")
	}
	result := ToolArgumentBinding(value)
	if err := result.validateParameterOptions(); err != nil {
		return err
	}
	*b = result
	return nil
}

// ResolveParameter projects an explicitly bound string without changing the
// persisted parameter map. Parsing is opt-in; defaults apply only to absence.
func (b ToolArgumentBinding) ResolveParameter(parameters map[string]string) (any, error) {
	if b.Source != "parameter" || !validToolBindingName(b.Name) {
		return nil, Invalidf("invalid parameter tool argument")
	}
	if err := b.validateParameterOptions(); err != nil {
		return nil, err
	}
	value, present := parameters[b.Name]
	if !present {
		if b.Default == nil {
			return nil, Invalidf("required bound tool parameter is missing")
		}
		if b.Parse == "integer" {
			return toolDefaultInteger(b.Default)
		}
		return b.Default, nil
	}
	if b.Parse == "" {
		return value, nil
	}
	// Safe JSON integers have at most 16 decimal digits. Bound conversion work
	// before parsing; the scanner applies its narrower argument range later.
	if len(value) == 0 || len(value) > 16 || (len(value) > 1 && value[0] == '0') {
		return nil, Invalidf("tool parameter must be a canonical unsigned integer")
	}
	for _, digit := range value {
		if digit < '0' || digit > '9' {
			return nil, Invalidf("tool parameter must be a canonical unsigned integer")
		}
	}
	number, err := strconv.ParseUint(value, 10, 53)
	if err != nil || number > 9007199254740991 {
		return nil, Invalidf("tool parameter exceeds the exact integer bound")
	}
	return int64(number), nil
}

func (b ToolArgumentBinding) validateParameterOptions() error {
	if b.Source != "parameter" {
		if b.Parse != "" || b.Default != nil {
			return Invalidf("parse and default require a parameter binding")
		}
		return nil
	}
	if b.Parse != "" && b.Parse != "integer" {
		return Invalidf("unsupported tool parameter parser")
	}
	if b.Default == nil {
		return nil
	}
	if b.Parse == "integer" {
		_, err := toolDefaultInteger(b.Default)
		return err
	}
	value, ok := b.Default.(string)
	if !ok || len(value) > 8192 {
		return Invalidf("unparsed parameter default must be a bounded string")
	}
	return nil
}

func toolDefaultInteger(value any) (int64, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return 0, Invalidf("invalid integer tool parameter default")
	}
	var decoded any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		return 0, Invalidf("invalid integer tool parameter default")
	}
	number, ok := decoded.(float64)
	if !ok || math.IsNaN(number) || math.IsInf(number, 0) || number < 0 ||
		number > 9007199254740991 || math.Trunc(number) != number {
		return 0, Invalidf("integer tool parameter default must be an exact nonnegative integer")
	}
	return int64(number), nil
}

func (b *ToolArgumentBinding) UnmarshalYAML(node *yaml.Node) error {
	var fields map[string]any
	if err := node.Decode(&fields); err != nil {
		return err
	}
	data, err := json.Marshal(fields)
	if err != nil {
		return Invalidf("tool binding is not JSON-compatible")
	}
	return b.UnmarshalJSON(data)
}

type ToolExecutionConfig struct {
	Tool           string                         `json:"tool" yaml:"tool" strictjson:"required"`
	Arguments      map[string]ToolArgumentBinding `json:"arguments" yaml:"arguments" strictjson:"required"`
	ResultArtifact string                         `json:"resultArtifact" yaml:"resultArtifact" strictjson:"required"`
	TimeoutSeconds int                            `json:"timeoutSeconds" yaml:"timeoutSeconds" strictjson:"required"`
}

func (t ResolvedAgentTemplate) IsToolWorker() bool {
	return t.Runtime == (WorkerRuntimeRef{RuntimeID: "tool", Version: "1"})
}

func (p ResolvedModelPolicy) IsZero() bool { return p == (ResolvedModelPolicy{}) }

func (e ToolExecutionConfig) Validate() error {
	if !toolArgumentName.MatchString(e.Tool) || e.Arguments == nil || len(e.Arguments) > 32 ||
		!validToolBindingName(e.ResultArtifact) || e.TimeoutSeconds < 1 || e.TimeoutSeconds > 3600 {
		return Invalidf("invalid tool execution selector, bindings or deadline")
	}
	for key, binding := range e.Arguments {
		if err := binding.validateParameterOptions(); err != nil {
			return err
		}
		if !toolArgumentName.MatchString(key) {
			return Invalidf("invalid tool argument name")
		}
		switch binding.Source {
		case "parameter", "artifact":
			if !validToolBindingName(binding.Name) || binding.Value != nil {
				return Invalidf("invalid tool argument source binding")
			}
		case "literal":
			if binding.Name != "" {
				return Invalidf("literal tool argument must not have name")
			}
			encoded, err := json.Marshal(binding.Value)
			if err != nil {
				return Invalidf("invalid literal tool argument")
			}
			var value any
			if json.Unmarshal(encoded, &value) != nil {
				return Invalidf("invalid literal tool argument")
			}
			switch v := value.(type) {
			case string:
				if len(v) > 8192 {
					return Invalidf("literal tool string exceeds its bound")
				}
			case bool:
			case float64:
				if math.IsInf(v, 0) || math.IsNaN(v) || math.Abs(v) > 9007199254740991 {
					return Invalidf("literal tool number exceeds its bound")
				}
			default:
				return Invalidf("literal tool arguments must be non-null scalars")
			}
		default:
			return Invalidf("unknown tool argument source")
		}
	}
	return nil
}

func validToolBindingName(value string) bool {
	return len(value) > 0 && len(value) <= 128 && strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\r\n\t\x00")
}

func (e *ToolExecutionConfig) Clone() *ToolExecutionConfig {
	if e == nil {
		return nil
	}
	result := *e
	result.Arguments = make(map[string]ToolArgumentBinding, len(e.Arguments))
	for key, value := range e.Arguments {
		result.Arguments[key] = value
	}
	return &result
}

func (t ResolvedAgentTemplate) ValidateToolExecution() error {
	if !t.IsToolWorker() {
		if t.Execution != nil {
			return Invalidf("execution is only supported by tool@1")
		}
		return nil
	}
	if t.Execution == nil || !t.ModelPolicy.IsZero() || t.Instructions != (ResolvedInstructions{}) || t.Summarizer != nil || len(t.Skills) != 0 {
		return Invalidf("tool@1 requires execution and forbids model configuration, instructions and skills")
	}
	if t.SandboxProfile != (SandboxProfileRef{SandboxProfileID: "local-workdir", Version: "1"}) ||
		len(t.Toolsets) != 1 || len(t.Toolsets[0].Tools) != 1 || t.Toolsets[0].Tools[0] != t.Execution.Tool {
		return Invalidf("tool@1 requires one selected operation and local-workdir@1")
	}
	return t.Execution.Validate()
}
