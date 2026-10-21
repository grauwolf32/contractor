package resolved

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// ExecutionConfigPatch is the strict reference-only shape accepted by a Run.
// Its custom decoder preserves credential's omitted/string/null tri-state.
type ExecutionConfigPatch struct {
	Planner *ExecutionSelectionPatch
	Workers *ExecutionSelectionPatch
	Stages  map[string]StageExecutionConfigPatch
}

// StageExecutionConfigPatch is the planner and per-Agent part of one Stage's
// patch, and the body of a named ExecutionConfig profile.
type StageExecutionConfigPatch struct {
	Planner *ExecutionSelectionPatch
	Agents  map[string]ExecutionSelectionPatch
}

// ExecutionSelectionPatch selects, by reference, any of one consumer's
// ModelPolicy, LLMGatewayConfig and credential. The catalog loader builds it
// from YAML and resolves the references against its configuration set.
type ExecutionSelectionPatch struct {
	ModelPolicy           OptionalString
	SummarizerModelPolicy OptionalString
	LLMGateway            OptionalString
	Credential            OptionalString
}

// OptionalString is one member of a patch: absent, an explicit null (which
// only a credential accepts, to clear it) or a string value.
type OptionalString struct {
	Present bool
	Null    bool
	Value   string
}

func (p ExecutionConfigPatch) MarshalJSON() ([]byte, error) {
	return json.Marshal(p.canonicalValue())
}

func (p *ExecutionConfigPatch) UnmarshalJSON(data []byte) error {
	object, err := decodePatchObject(data, "executionConfig")
	if err != nil {
		return err
	}
	*p = ExecutionConfigPatch{}
	for key, raw := range object {
		switch key {
		case "planner":
			value := new(ExecutionSelectionPatch)
			if err := value.unmarshalJSON(raw, "executionConfig.planner"); err != nil {
				return err
			}
			p.Planner = value
		case "workers":
			value := new(ExecutionSelectionPatch)
			if err := value.unmarshalJSON(raw, "executionConfig.workers"); err != nil {
				return err
			}
			p.Workers = value
		case "stages":
			stages, err := decodePatchObject(raw, "executionConfig.stages")
			if err != nil {
				return err
			}
			p.Stages = make(map[string]StageExecutionConfigPatch, len(stages))
			for name, encoded := range stages {
				var value StageExecutionConfigPatch
				if err := value.unmarshalJSON(encoded, "executionConfig.stages."+name); err != nil {
					return err
				}
				p.Stages[name] = value
			}
		default:
			return fmt.Errorf("executionConfig contains unknown field %q", key)
		}
	}
	return nil
}

func (p *StageExecutionConfigPatch) unmarshalJSON(data []byte, field string) error {
	object, err := decodePatchObject(data, field)
	if err != nil {
		return err
	}
	*p = StageExecutionConfigPatch{}
	for key, raw := range object {
		switch key {
		case "planner":
			value := new(ExecutionSelectionPatch)
			if err := value.unmarshalJSON(raw, field+".planner"); err != nil {
				return err
			}
			p.Planner = value
		case "agents":
			agents, err := decodePatchObject(raw, field+".agents")
			if err != nil {
				return err
			}
			p.Agents = make(map[string]ExecutionSelectionPatch, len(agents))
			for name, encoded := range agents {
				var value ExecutionSelectionPatch
				if err := value.unmarshalJSON(encoded, field+".agents."+name); err != nil {
					return err
				}
				p.Agents[name] = value
			}
		default:
			return fmt.Errorf("%s contains unknown field %q", field, key)
		}
	}
	return nil
}

func (p *ExecutionSelectionPatch) unmarshalJSON(data []byte, field string) error {
	object, err := decodePatchObject(data, field)
	if err != nil {
		return err
	}
	*p = ExecutionSelectionPatch{}
	for key, raw := range object {
		switch key {
		case "modelPolicy":
			p.ModelPolicy, err = decodeOptionalString(raw, false, field+".modelPolicy")
		case "summarizerModelPolicy":
			p.SummarizerModelPolicy, err = decodeOptionalString(raw, false, field+".summarizerModelPolicy")
		case "llmGateway":
			p.LLMGateway, err = decodeOptionalString(raw, false, field+".llmGateway")
		case "credential":
			p.Credential, err = decodeOptionalString(raw, true, field+".credential")
		default:
			return fmt.Errorf("%s contains unknown field %q", field, key)
		}
		if err != nil {
			return err
		}
	}
	if !p.HasAny() {
		return fmt.Errorf("%s must select at least one field", field)
	}
	return nil
}

func decodePatchObject(data []byte, field string) (map[string]json.RawMessage, error) {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return nil, fmt.Errorf("%s must not be null", field)
	}
	var result map[string]json.RawMessage
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&result); err != nil || result == nil {
		return nil, fmt.Errorf("%s must be an object", field)
	}
	return result, nil
}

func decodeOptionalString(data []byte, allowNull bool, field string) (OptionalString, error) {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		if !allowNull {
			return OptionalString{}, fmt.Errorf("%s must not be null", field)
		}
		return OptionalString{Present: true, Null: true}, nil
	}
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return OptionalString{}, fmt.Errorf("%s must be a string", field)
	}
	return OptionalString{Present: true, Value: value}, nil
}

// HasAny reports whether the patch selects at least one member.
func (p ExecutionSelectionPatch) HasAny() bool {
	return p.ModelPolicy.Present || p.SummarizerModelPolicy.Present || p.LLMGateway.Present || p.Credential.Present
}

// canonicalValue is used by public request idempotency hashing. It preserves
// explicit credential null while normalizing omitted maps and key order through
// encoding/json.
func (p ExecutionConfigPatch) canonicalValue() map[string]any {
	result := make(map[string]any)
	if p.Planner != nil {
		result["planner"] = p.Planner.canonicalValue()
	}
	if p.Workers != nil {
		result["workers"] = p.Workers.canonicalValue()
	}
	if len(p.Stages) > 0 {
		stages := make(map[string]any, len(p.Stages))
		for name, value := range p.Stages {
			canonical := value.CanonicalValue()
			if len(canonical) > 0 {
				stages[name] = canonical
			}
		}
		if len(stages) > 0 {
			result["stages"] = stages
		}
	}
	return result
}

func (p ExecutionConfigPatch) CanonicalValue() map[string]any { return p.canonicalValue() }

// CanonicalValue is the JSON shape of the patch that an ExecutionConfig
// profile digest covers; it preserves an explicit credential null.
func (p StageExecutionConfigPatch) CanonicalValue() map[string]any {
	result := make(map[string]any)
	if p.Planner != nil {
		result["planner"] = p.Planner.canonicalValue()
	}
	if len(p.Agents) > 0 {
		agents := make(map[string]any, len(p.Agents))
		for name, value := range p.Agents {
			agents[name] = value.canonicalValue()
		}
		result["agents"] = agents
	}
	return result
}

func (p ExecutionSelectionPatch) canonicalValue() map[string]any {
	result := make(map[string]any)
	for name, value := range map[string]OptionalString{
		"modelPolicy": p.ModelPolicy, "summarizerModelPolicy": p.SummarizerModelPolicy, "llmGateway": p.LLMGateway, "credential": p.Credential,
	} {
		if !value.Present {
			continue
		}
		if value.Null {
			result[name] = nil
		} else {
			result[name] = value.Value
		}
	}
	return result
}
