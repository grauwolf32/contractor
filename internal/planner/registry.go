package planner

import (
	"context"
	"fmt"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"maps"
	"slices"
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

// Refs returns the sorted refs of the registered PlannerFactories.
func (r *Registry) Refs() []string {
	return slices.Sorted(maps.Keys(r.factories))
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

// ExecuteOperation routes an allocation-authenticated callback to the current
// Streamline invocation. Its Factory owns registration and removal during Run.
func (r *Registry) ExecuteOperation(ctx context.Context, stageID, allocationID string, request contracts.PlannerOperation) (contracts.PlannerOperationResult, error) {
	if r != nil {
		if factory, ok := r.factories[resolved.StreamlinePlanner].(interface {
			ExecuteOperation(context.Context, string, string, contracts.PlannerOperation) (contracts.PlannerOperationResult, error)
		}); ok {
			return factory.ExecuteOperation(ctx, stageID, allocationID, request)
		}
	}
	return contracts.PlannerOperationResult{}, fmt.Errorf("Runtime Planner is unavailable")
}
