// Package router exposes the router@1 PlannerFactory. Its Google ADK execution
// engine is shared with Streamline, while the public factory preserves a
// distinct exact ref and the framework-neutral planner.Factory boundary.
package router

import (
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
	"google.golang.org/adk/model"
)

const Ref = planner.RouterRef

type Limits = streamline.Limits
type ADKSessionFactory = streamline.ADKSessionFactory

func DefaultLimits() Limits { return streamline.DefaultLimits() }

type Factory struct {
	delegate *streamline.Factory
}

func NewFactory(
	sessions planner.SessionService,
	adkSessions ADKSessionFactory,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	llm model.LLM,
	limits Limits,
) (*Factory, error) {
	delegate, err := streamline.NewRouterDelegate(
		sessions, adkSessions, invoker, inspector, llm, limits,
	)
	if err != nil {
		return nil, err
	}
	return &Factory{delegate: delegate}, nil
}

func (*Factory) Ref() string { return Ref }

func (f *Factory) Create(invocation planner.Invocation) (planner.Planner, error) {
	return f.delegate.Create(invocation)
}

var _ planner.Factory = (*Factory)(nil)
