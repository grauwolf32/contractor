package streamline

import (
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/planner"
)

// NewConfiguredFactory supplies the production factory with no Memory store.
// Production composition always uses NewConfiguredFactoryWithMemory.
func NewConfiguredFactory(
	sessions planner.PlanSessionService,
	adkSessions ADKSessionFactory,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	stateReader planner.WorkerStateReader,
	modelFactory InvocationModelFactory,
	limits Limits,
) (*Factory, error) {
	return newFactory(
		streamlineProfile, sessions, adkSessions, invoker, inspector, stateReader, nil, modelFactory, limits,
	)
}

// NewConfiguredRouterDelegate supplies the Router test engine with no Memory store.
func NewConfiguredRouterDelegate(
	sessions planner.PlanSessionService,
	adkSessions ADKSessionFactory,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	stateReader planner.WorkerStateReader,
	modelFactory InvocationModelFactory,
	limits Limits,
) (*Factory, error) {
	return NewConfiguredRouterDelegateWithMemory(
		sessions, adkSessions, invoker, inspector, stateReader, nil, modelFactory, limits,
	)
}

func newLocalFactoryWithMemory(sessions planner.PlanSessionService, adkSessions ADKSessionFactory, invoker planner.WorkerInvoker, inspector planner.ArtifactInspector, stateReader planner.WorkerStateReader, memory plannermemory.Store, models InvocationModelFactory, limits Limits) (*Factory, error) {
	return newFactory(streamlineProfile, sessions, adkSessions, invoker, inspector, stateReader, memory, models, limits)
}
