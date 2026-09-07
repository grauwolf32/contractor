package streamline

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"google.golang.org/adk/model"
)

func TestPlannerRequestHonorsPolicyTemperature(t *testing.T) {
	for _, profile := range []plannerProfile{streamlineProfile, routerProfile} {
		for _, test := range []struct {
			name       string
			configured bool
			value      *float64
		}{
			{"legacy", false, nil}, {"omitted", true, nil},
			{"zero", true, tempPointer(0)}, {"fraction", true, tempPointer(0.1)},
			{"thinking", true, tempPointer(1)},
		} {
			t.Run(profile.agentName+"/"+test.name, func(t *testing.T) {
				observed := false
				llm := &scriptedModel{steps: []modelStep{func(request *model.LLMRequest) (*model.LLMResponse, error) {
					observed = true
					wire, err := encodeChatRequest(request, "test-model", 1024)
					if err != nil {
						t.Fatal(err)
					}
					var payload map[string]any
					if err := json.Unmarshal(wire, &payload); err != nil {
						t.Fatal(err)
					}
					value, present := payload["temperature"]
					if test.configured && test.value == nil {
						if present {
							t.Fatalf("omitted temperature sent as %v", value)
						}
					} else {
						want := 0.0
						if test.value != nil {
							want = *test.value
						}
						if !present || value != want {
							t.Fatalf("temperature = %v (present %v), want %v", value, present, want)
						}
					}
					return functionStep(finishToolName, map[string]any{"outcome": "failed", "summary": "done", "artifacts": map[string]any{}})(request)
				}}}
				sessions := newFakeSessions()
				factory := mustFactory(t, sessions, &fakeWorkerInvoker{}, &fakeInspector{}, llm, Limits{})
				factory.profile = profile
				invocation := testInvocation("builder")
				invocation.Stage.Planner.PlannerID = strings.TrimSuffix(profile.ref, "@1")
				created, err := factory.Create(invocation)
				if err != nil {
					t.Fatal(err)
				}
				instance := created.(*streamlinePlanner)
				if test.configured {
					instance.invocation.ModelAccess = &planner.ModelAccess{ModelPolicy: contracts.ResolvedModelPolicy{Model: "test-model", Temperature: test.value}}
				}
				_, _ = instance.Run(t.Context())
				if !observed {
					t.Fatal("Planner made no model request")
				}
			})
		}
	}
}

func tempPointer(value float64) *float64 { return &value }
