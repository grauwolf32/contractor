package contracts

import (
	"bytes"
	"encoding/json"
	"math"
	"regexp"
	"strings"

	"go.yaml.in/yaml/v4"
)

var toolArgumentName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]{0,63}$`)

// Preserve forbidden-field presence: explicit null is not model absence.
func (t *ResolvedAgentTemplate) UnmarshalJSON(data []byte) error {
	type wire ResolvedAgentTemplate
	var value wire
	fields, err := decodeToolWorkerObject(data, &value)
	if err != nil {
		return err
	}
	if ResolvedAgentTemplate(value).IsToolWorker() {
		for _, name := range []string{"instructions", "modelPolicy", "summarizer"} {
			if fields[name] != nil {
				return invalidf("tool@1 forbids %s", name)
			}
		}
	} else if fields["execution"] != nil {
		return invalidf("execution is only supported by tool@1")
	}
	*t = ResolvedAgentTemplate(value)
	return nil
}

func (s *AllocationSpec) UnmarshalJSON(data []byte) error {
	type wire AllocationSpec
	var value wire
	fields, err := decodeToolWorkerObject(data, &value)
	if err != nil {
		return err
	}
	if value.AgentTemplate.IsToolWorker() {
		if fields["modelPolicy"] != nil || fields["completionContract"] != nil {
			return invalidf("tool@1 forbids modelPolicy and completionContract")
		}
		var settings map[string]json.RawMessage
		if err := json.Unmarshal(fields["runtimeSettings"], &settings); err != nil {
			return err
		}
		if settings["llmGatewayUrl"] != nil {
			return invalidf("tool@1 forbids llmGatewayUrl")
		}
	}
	*s = AllocationSpec(value)
	return nil
}

func decodeToolWorkerObject(data []byte, target any) (map[string]json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return nil, err
	}
	var fields map[string]json.RawMessage
	err := json.Unmarshal(data, &fields)
	return fields, err
}

type ToolArgumentBinding struct {
	Source string `json:"source" yaml:"source"`
	Name   string `json:"name,omitempty" yaml:"name,omitempty"`
	Value  any    `json:"value,omitempty" yaml:"value,omitempty"`
}

func (b *ToolArgumentBinding) UnmarshalJSON(data []byte) error {
	type plain ToolArgumentBinding
	var value plain
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	if len(fields) != 2 || fields["source"] == nil || (value.Source == "literal" && fields["value"] == nil) || (value.Source != "literal" && fields["name"] == nil) {
		return invalidf("tool argument must select exactly one source and its payload")
	}
	*b = ToolArgumentBinding(value)
	return nil
}

func (b *ToolArgumentBinding) UnmarshalYAML(node *yaml.Node) error {
	var fields map[string]any
	if err := node.Decode(&fields); err != nil {
		return err
	}
	data, err := json.Marshal(fields)
	if err != nil {
		return invalidf("tool binding is not JSON-compatible")
	}
	return b.UnmarshalJSON(data)
}

type ToolExecutionConfig struct {
	Tool           string                         `json:"tool" yaml:"tool"`
	Arguments      map[string]ToolArgumentBinding `json:"arguments" yaml:"arguments"`
	ResultArtifact string                         `json:"resultArtifact" yaml:"resultArtifact"`
	TimeoutSeconds int                            `json:"timeoutSeconds" yaml:"timeoutSeconds"`
}

func (t ResolvedAgentTemplate) IsToolWorker() bool {
	return t.Runtime == (WorkerRuntimeRef{RuntimeID: "tool", Version: "1"})
}

func (p ResolvedModelPolicy) IsZero() bool { return p == (ResolvedModelPolicy{}) }

func (e ToolExecutionConfig) Validate() error {
	if !toolArgumentName.MatchString(e.Tool) || e.Arguments == nil || len(e.Arguments) > 32 ||
		!validToolBindingName(e.ResultArtifact) || e.TimeoutSeconds < 1 || e.TimeoutSeconds > 3600 {
		return invalidf("invalid tool execution selector, bindings or deadline")
	}
	for key, binding := range e.Arguments {
		if !toolArgumentName.MatchString(key) {
			return invalidf("invalid tool argument name")
		}
		switch binding.Source {
		case "parameter", "artifact":
			if !validToolBindingName(binding.Name) || binding.Value != nil {
				return invalidf("invalid tool argument source binding")
			}
		case "literal":
			if binding.Name != "" {
				return invalidf("literal tool argument must not have name")
			}
			encoded, err := json.Marshal(binding.Value)
			if err != nil {
				return invalidf("invalid literal tool argument")
			}
			var value any
			if json.Unmarshal(encoded, &value) != nil {
				return invalidf("invalid literal tool argument")
			}
			switch v := value.(type) {
			case string:
				if len(v) > 8192 {
					return invalidf("literal tool string exceeds its bound")
				}
			case bool:
			case float64:
				if math.IsInf(v, 0) || math.IsNaN(v) || math.Abs(v) > 9007199254740991 {
					return invalidf("literal tool number exceeds its bound")
				}
			default:
				return invalidf("literal tool arguments must be non-null scalars")
			}
		default:
			return invalidf("unknown tool argument source")
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
			return invalidf("execution is only supported by tool@1")
		}
		return nil
	}
	if t.Execution == nil || !t.ModelPolicy.IsZero() || t.Instructions != (ResolvedInstructions{}) || t.Summarizer != nil || len(t.Skills) != 0 {
		return invalidf("tool@1 requires execution and forbids model configuration, instructions and skills")
	}
	if t.SandboxProfile != (SandboxProfileRef{SandboxProfileID: "local-workdir", Version: "1"}) ||
		len(t.Toolsets) != 1 || len(t.Toolsets[0].Tools) != 1 || t.Toolsets[0].Tools[0] != t.Execution.Tool {
		return invalidf("tool@1 requires one selected operation and local-workdir@1")
	}
	return t.Execution.Validate()
}
