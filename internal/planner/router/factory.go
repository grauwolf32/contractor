// Package router exposes the router@1 PlannerFactory. Its Runtime execution
// host is shared with Streamline, while the public factory preserves a
// distinct exact ref and the framework-neutral planner.Factory boundary.
package router

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
)

type Factory struct{ delegate *streamline.Factory }

func NewRuntimeFactory(delegate *streamline.Factory) (*Factory, error) {
	if delegate == nil {
		return nil, fmt.Errorf("Runtime Planner factory is required")
	}
	return &Factory{delegate: delegate}, nil
}
func (*Factory) Ref() string { return resolved.RouterPlanner }
func (f *Factory) Create(invocation planner.Invocation) (planner.Planner, error) {
	return f.delegate.CreateRouter(invocation)
}

var _ planner.Factory = (*Factory)(nil)
