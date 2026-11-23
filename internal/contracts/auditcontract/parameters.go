package auditcontract

import (
	"maps"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// ValidateParameters retains the common Task parameter bounds for launch values.
func ValidateParameters(values map[string]string) error {
	if len(values) > 128 {
		return errInvalid
	}
	for name, value := range values {
		if !contracts.ValidName(name) || len(value) > 4096 || !utf8.ValidString(value) || strings.ContainsRune(value, 0) {
			return errInvalid
		}
	}
	return nil
}

// BindParameters derives an execution Program from the exact authored Program
// and its baseline's explicit launch values. Only declared bindings can change;
// the catalog snapshot and caller-owned maps remain untouched.
func (p Program) BindParameters(values map[string]string) (Program, error) {
	if ValidateParameters(values) != nil {
		return Program{}, errInvalid
	}
	for name := range values {
		if _, exists := p.Parameters[name]; !exists {
			return Program{}, errInvalid
		}
	}
	for name, slot := range p.Parameters {
		if _, exists := values[name]; slot.Required && !exists {
			return Program{}, errInvalid
		}
	}
	p.Workflows = maps.Clone(p.Workflows)
	for alias, binding := range p.Workflows {
		binding.Parameters = maps.Clone(binding.Parameters)
		if binding.Parameters == nil {
			binding.Parameters = map[string]string{}
		}
		for target, source := range binding.ParameterInputs {
			if value, exists := values[source]; exists {
				binding.Parameters[target] = value
			}
		}
		if ValidateParameters(binding.Parameters) != nil {
			return Program{}, errInvalid
		}
		p.Workflows[alias] = binding
	}
	return p, nil
}
