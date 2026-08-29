package planner

import (
	"fmt"
	"strings"
)

type Registry struct {
	factories map[string]Factory
}

func NewRegistry(factories ...Factory) (*Registry, error) {
	if len(factories) == 0 {
		return nil, fmt.Errorf("PlannerFactory registry must not be empty")
	}
	result := &Registry{factories: make(map[string]Factory, len(factories))}
	for _, factory := range factories {
		if factory == nil || strings.TrimSpace(factory.Ref()) == "" {
			return nil, fmt.Errorf("PlannerFactory and exact ref are required")
		}
		if _, exists := result.factories[factory.Ref()]; exists {
			return nil, fmt.Errorf("duplicate PlannerFactory %q", factory.Ref())
		}
		result.factories[factory.Ref()] = factory
	}
	return result, nil
}

func (r *Registry) Create(ref string, invocation Invocation) (Planner, error) {
	if r == nil {
		return nil, fmt.Errorf("PlannerFactory registry is nil")
	}
	factory, ok := r.factories[ref]
	if !ok {
		return nil, fmt.Errorf("unknown PlannerFactory %q", ref)
	}
	return factory.Create(invocation)
}
