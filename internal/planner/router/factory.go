// Package router exposes the router@1 PlannerFactory. Its Google ADK execution
// engine is shared with Streamline, while the public factory preserves a
// distinct exact ref and the framework-neutral planner.Factory boundary.
package router

import (
	"github.com/grauwolf32/contractor/internal/config/resolved"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
)

type Limits = streamline.Limits
type ADKSessionFactory = streamline.ADKSessionFactory
type InvocationModelFactory = streamline.InvocationModelFactory

type Factory struct {
	delegate *streamline.Factory
}

func NewConfiguredFactoryWithMemory(
	sessions planner.PlanSessionService,
	adkSessions ADKSessionFactory,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	stateReader planner.WorkerStateReader,
	memoryStore plannermemory.Store,
	modelFactory InvocationModelFactory,
	limits Limits,
) (*Factory, error) {
	delegate, err := streamline.NewConfiguredRouterDelegateWithMemory(
		sessions, adkSessions, invoker, inspector, stateReader, memoryStore, modelFactory, limits,
	)
	if err != nil {
		return nil, err
	}
	return &Factory{delegate: delegate}, nil
}

func (*Factory) Ref() string { return resolved.RouterPlanner }

func (f *Factory) Create(invocation planner.Invocation) (planner.Planner, error) {
	return f.delegate.Create(invocation)
}

var _ planner.Factory = (*Factory)(nil)
