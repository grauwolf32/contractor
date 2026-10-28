package auditprofile

import (
	"fmt"
	"maps"
	"slices"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const MaxAuditCheckMethods = 128

type AuditCheckMethodMapping struct {
	PlanSchema           string                                 `json:"planSchema" yaml:"planSchema"`
	Method               string                                 `json:"method" yaml:"method"`
	Phase                string                                 `json:"phase" yaml:"phase"`
	WorkflowRole         string                                 `json:"workflowRole" yaml:"workflowRole"`
	RequiredCapabilities []contracts.CheckCapabilityRequirement `json:"requiredCapabilities" yaml:"-"`
}

type checkMethodSource struct {
	PlanSchema           string                   `yaml:"planSchema"`
	Method               string                   `yaml:"method"`
	WorkflowRole         string                   `yaml:"workflowRole"`
	RequiredCapabilities *[]checkCapabilitySource `yaml:"requiredCapabilities"`
}
type checkCapabilitySource struct {
	Stage string `yaml:"stage"`
	Agent string `yaml:"agent"`
	Ref   struct {
		ToolsetID string `yaml:"toolsetId"`
		Version   string `yaml:"version"`
	} `yaml:"ref"`
	Tools []string `yaml:"tools"`
}

type NamedCheckMethod struct {
	Name    string
	Mapping AuditCheckMethodMapping
}
type CheckMethodIndex struct {
	methods map[contracts.CheckPlanKey]NamedCheckMethod
}

func (index CheckMethodIndex) Select(key contracts.CheckPlanKey) (NamedCheckMethod, bool) {
	value, ok := index.methods[key]
	value.Mapping = cloneCheckMethod(value.Mapping)
	return value, ok
}

func CompileCheckMethods(profile ResolvedAuditProfile) (CheckMethodIndex, error) {
	result := CheckMethodIndex{methods: map[contracts.CheckPlanKey]NamedCheckMethod{}}
	if len(profile.Methods) > MaxAuditCheckMethods {
		return result, fmt.Errorf("methods exceeds its entry limit")
	}
	for _, name := range slices.Sorted(maps.Keys(profile.Methods)) {
		method := profile.Methods[name]
		if validateAuditMapKey("methods name", name) != nil || strings.ContainsRune(name, 0) {
			return result, fmt.Errorf("methods name is invalid")
		}
		key, known := contracts.LookupCheckPlanMethod(method.PlanSchema, method.Method)
		if !known || key.Phase != method.Phase {
			return result, fmt.Errorf("methods selects an unregistered schema, method or phase")
		}
		if _, duplicate := result.methods[key]; duplicate {
			return result, fmt.Errorf("methods contains an ambiguous semantic key")
		}
		binding, exists := profile.Workflows[method.WorkflowRole]
		if !exists || binding.Kind != AuditWorkflowCheck {
			return result, fmt.Errorf("methods must select an exact check binding")
		}
		if binding.Workflow.AuditTask != nil {
			return result, fmt.Errorf("finding check method requires an ordinary finding executor")
		}
		if err := ValidateCheckMethodCapabilities(binding, method.RequiredCapabilities); err != nil {
			return result, err
		}
		result.methods[key] = NamedCheckMethod{Name: name, Mapping: cloneCheckMethod(method)}
	}
	return result, nil
}

func ValidateCheckMethodCapabilities(binding ResolvedAuditWorkflowBinding, requirements []contracts.CheckCapabilityRequirement) error {
	if requirements == nil || len(requirements) > contracts.MaximumCheckCapabilities {
		return fmt.Errorf("requiredCapabilities must be an explicit bounded array")
	}
	previous := ""
	for _, required := range requirements {
		key := capabilityKey(required)
		if key <= previous || resolved.ValidateWorkflowName("stage", required.Stage) != nil || resolved.ValidateWorkflowName("agent", required.Agent) != nil {
			return fmt.Errorf("requiredCapabilities must be sorted unique exact Worker selections")
		}
		previous = key
		stage, exists := binding.Workflow.Stages[required.Stage]
		if !exists {
			return fmt.Errorf("required capability Stage is absent")
		}
		agent, exists := stage.Agents[required.Agent]
		if !exists {
			return fmt.Errorf("required capability Worker is absent")
		}
		selector := required.Ref.ToolsetID + "@" + required.Ref.Version
		known, exists := resolved.ToolsetTools(selector)
		if !exists || len(required.Tools) == 0 || len(required.Tools) > contracts.MaximumCheckCapabilityTools {
			return fmt.Errorf("required capability Toolset or tools are invalid")
		}
		selected := []string(nil)
		for _, selection := range agent.Template.Toolsets {
			if selection.Ref == required.Ref {
				selected = selection.Tools
				break
			}
		}
		last := ""
		for _, tool := range required.Tools {
			if tool <= last || !slices.Contains(known, tool) || !slices.Contains(selected, tool) {
				return fmt.Errorf("required capability is unavailable from its exact Worker")
			}
			last = tool
		}
	}
	return nil
}
func resolveCheckMethods(source map[string]checkMethodSource) (map[string]AuditCheckMethodMapping, error) {
	if len(source) == 0 {
		return nil, nil
	}
	result := make(map[string]AuditCheckMethodMapping, len(source))
	for name, value := range source {
		key, known := contracts.LookupCheckPlanMethod(value.PlanSchema, value.Method)
		if !known {
			return nil, fmt.Errorf("methods selects an unregistered schema or method")
		}
		if value.RequiredCapabilities == nil {
			return nil, fmt.Errorf("methods.requiredCapabilities is required")
		}
		mapping := AuditCheckMethodMapping{PlanSchema: key.Schema, Method: key.Method, Phase: key.Phase, WorkflowRole: value.WorkflowRole, RequiredCapabilities: []contracts.CheckCapabilityRequirement{}}
		for _, capability := range *value.RequiredCapabilities {
			tools := slices.Clone(capability.Tools)
			sort.Strings(tools)
			mapping.RequiredCapabilities = append(mapping.RequiredCapabilities, contracts.CheckCapabilityRequirement{Stage: capability.Stage, Agent: capability.Agent, Ref: contracts.ToolsetRef{ToolsetID: capability.Ref.ToolsetID, Version: capability.Ref.Version}, Tools: tools})
		}
		sort.Slice(mapping.RequiredCapabilities, func(i, j int) bool {
			return capabilityKey(mapping.RequiredCapabilities[i]) < capabilityKey(mapping.RequiredCapabilities[j])
		})
		result[name] = mapping
	}
	return result, nil
}
func capabilityKey(value contracts.CheckCapabilityRequirement) string {
	return value.Stage + "\x00" + value.Agent + "\x00" + value.Ref.ToolsetID + "@" + value.Ref.Version
}
func cloneCheckMethod(value AuditCheckMethodMapping) AuditCheckMethodMapping {
	value.RequiredCapabilities = append([]contracts.CheckCapabilityRequirement{}, value.RequiredCapabilities...)
	for i := range value.RequiredCapabilities {
		value.RequiredCapabilities[i].Tools = slices.Clone(value.RequiredCapabilities[i].Tools)
	}
	return value
}
