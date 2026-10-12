package contracts_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

func TestSharedAuditCompletionSharedPythonFixtures(t *testing.T) {
	raw := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "audit-completion-cases.json")
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
				err = privateReject[contracts.WorkerCompletionContract](c.Value)
			case "capabilities":
				err = privateReject[contracts.RuntimeCompletionCapabilities](c.Value)
			case "allocation":
				err = privateReject[control.AllocationSpec](c.Value)
			case "registration":
				err = privateReject[control.AgentRegistration](c.Value)
			default:
				t.Fatalf("unknown fixture model %q", c.Model)
			}
			if (err == nil) != c.Valid {
				t.Fatalf("valid=%v got %v", c.Valid, err)
			}
		})
	}
}
