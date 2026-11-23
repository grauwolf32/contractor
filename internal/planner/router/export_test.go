package router

import "github.com/grauwolf32/contractor/internal/planner"

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
	return NewConfiguredFactoryWithMemory(
		sessions, adkSessions, invoker, inspector, stateReader, nil, modelFactory, limits,
	)
}
