package config

import "testing"

func TestMVPDescriptorsDoNotExportFutureLiveTestingOperations(t *testing.T) {
	t.Parallel()

	futureOperations := map[string]bool{
		"get_vulnerability": true, "submit_verdict": true,
		"run_python": true, "execute_bash": true,
	}
	for ref, descriptor := range MVPDescriptors().Toolsets {
		for _, tool := range descriptor.Tools {
			if futureOperations[tool] {
				t.Errorf("current Toolset %s unexpectedly exports future live-testing operation %s", ref, tool)
			}
		}
	}
}
