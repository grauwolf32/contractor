package contracts

import (
	"encoding/json"
	"os"
	"testing"
)

func TestAuditCompletionSharedPythonFixtures(t *testing.T) {
	raw, err := os.ReadFile("../../api/testdata/v1alpha1/audit-completion-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name  string          `json:"name"`
		Model string          `json:"model"`
		Valid bool            `json:"valid"`
		Value json.RawMessage `json:"value"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) < 25 {
		t.Fatal("completion fixture coverage missing")
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			var err error
			switch c.Model {
			case "contract":
				err = privateReject[WorkerCompletionContract](c.Value)
			case "capabilities":
				err = privateReject[RuntimeCompletionCapabilities](c.Value)
			case "allocation":
				err = privateReject[AllocationSpec](c.Value)
			case "registration":
				err = privateReject[AgentRegistration](c.Value)
			default:
				t.Fatalf("unknown fixture model %q", c.Model)
			}
			if (err == nil) != c.Valid {
				t.Fatalf("valid=%v got %v", c.Valid, err)
			}
		})
	}
}

func TestCompletionCapabilityCloneDoesNotGrantOrShareSupport(t *testing.T) {
	empty := NormalizeAgentRegistration(AgentRegistration{})
	if empty.Capabilities != nil {
		t.Fatal("omission invented completion capability")
	}
	original := AgentRegistration{Capabilities: &RuntimeCompletionCapabilities{CompletionContracts: []string{AuditCheckResultsV1}}}
	cloned := NormalizeAgentRegistration(original)
	cloned.Capabilities.CompletionContracts[0] = "changed"
	if original.Capabilities.CompletionContracts[0] != AuditCheckResultsV1 {
		t.Fatal("capability clone aliases mutable source")
	}
}
