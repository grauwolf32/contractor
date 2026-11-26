package streamline

func (p *streamlinePlanner) resolveWorkerBinding(
	bindings []workerBinding,
	workerName string,
) (workerBinding, bool) {
	if !p.profile.routesWorkers {
		if len(bindings) == 1 && workerName == "" {
			return bindings[0], true
		}
		return workerBinding{}, false
	}
	for _, binding := range bindings {
		if binding.logicalName == workerName {
			return binding, true
		}
	}
	return workerBinding{}, false
}
