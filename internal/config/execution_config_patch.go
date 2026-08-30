package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
)

// ExecutionConfigPatch is the strict reference-only shape accepted by a Run.
// Its custom decoder preserves credential's omitted/string/null tri-state.
type ExecutionConfigPatch struct {
	Planner *ExecutionSelectionPatch
	Workers *ExecutionSelectionPatch
	Stages  map[string]StageExecutionConfigPatch
}

type StageExecutionConfigPatch struct {
	Planner *ExecutionSelectionPatch
	Agents  map[string]ExecutionSelectionPatch
}

type ExecutionSelectionPatch struct {
	modelPolicy optionalString
	llmGateway  optionalString
	credential  optionalString
}

type optionalString struct {
	present bool
	null    bool
	value   string
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
			p.modelPolicy, err = decodeOptionalString(raw, false, field+".modelPolicy")
		case "llmGateway":
			p.llmGateway, err = decodeOptionalString(raw, false, field+".llmGateway")
		case "credential":
			p.credential, err = decodeOptionalString(raw, true, field+".credential")
		default:
			return fmt.Errorf("%s contains unknown field %q", field, key)
		}
		if err != nil {
			return err
		}
	}
	if !p.hasAny() {
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

func decodeOptionalString(data []byte, allowNull bool, field string) (optionalString, error) {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		if !allowNull {
			return optionalString{}, fmt.Errorf("%s must not be null", field)
		}
		return optionalString{present: true, null: true}, nil
	}
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return optionalString{}, fmt.Errorf("%s must be a string", field)
	}
	return optionalString{present: true, value: value}, nil
}

func (p ExecutionSelectionPatch) hasAny() bool {
	return p.modelPolicy.present || p.llmGateway.present || p.credential.present
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
			canonical := value.canonicalValue()
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

func (p StageExecutionConfigPatch) canonicalValue() map[string]any {
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
	for name, value := range map[string]optionalString{
		"modelPolicy": p.modelPolicy, "llmGateway": p.llmGateway, "credential": p.credential,
	} {
		if !value.present {
			continue
		}
		if value.null {
			result[name] = nil
		} else {
			result[name] = value.value
		}
	}
	return result
}

func sortedPatchKeys[T any](source map[string]T) []string {
	result := make([]string, 0, len(source))
	for key := range source {
		result = append(result, key)
	}
	sort.Strings(result)
	return result
}
