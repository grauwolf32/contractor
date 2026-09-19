package public

import (
	"strings"
	"testing"
)

func TestHandlerRequiresExplicitRunServices(t *testing.T) {
	var dependencies Dependencies
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *Dependencies) { dependencies = *d })
	dependencies.RunCreator = newTestRunCreator(t, dependencies, fixture.runs, fixture.unit, true)
	for name, remove := range map[string]func(*Dependencies){
		"creation":  func(d *Dependencies) { d.RunCreator = nil },
		"queue":     func(d *Dependencies) { d.RunQueue = nil },
		"lifecycle": func(d *Dependencies) { d.RunLifecycle = nil },
		"reader":    func(d *Dependencies) { d.Runs = nil },
	} {
		t.Run(name, func(t *testing.T) {
			candidate := dependencies
			remove(&candidate)
			if _, err := NewHandler(candidate); err == nil || !strings.Contains(err.Error(), "dependencies are incomplete") {
				t.Fatalf("missing %s service: %v", name, err)
			}
		})
	}
	// Each resource can be supplied without the methods of the other services.
	dependencies.Runs = struct{ RunReader }{fixture.runs}
	dependencies.RunQueue = struct{ RunQueue }{fixture.runs}
	dependencies.RunLifecycle = struct{ RunLifecycle }{fixture.runs}
	if _, err := NewHandler(dependencies); err != nil {
		t.Fatalf("narrow services rejected: %v", err)
	}
}
