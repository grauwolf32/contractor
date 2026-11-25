package contracts_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

func TestSharedRuntimeSlotsCases(t *testing.T) {
	var cases []struct {
		Name  string         `json:"name"`
		Type  string         `json:"type"`
		Patch map[string]any `json:"patch"`
		Valid bool           `json:"valid"`
	}
	if err := json.Unmarshal(contractstest.ReadFixture(t, "", "runtime-slots-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("empty slot fixture")
	}
	for _, item := range cases {
		t.Run(item.Name, func(t *testing.T) {
			names := map[string]string{"AgentRegistration": "agent-registration.json", "AgentHeartbeat": "agent-heartbeat.json", "HeartbeatResponse": "heartbeat-response.json"}
			value := wireCase(t, names[item.Type])
			for key, field := range item.Patch {
				value[key] = field
			}
			raw, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			switch item.Type {
			case "AgentRegistration":
				_, err = contracts.DecodePrivateStrict[control.AgentRegistration](raw)
			case "AgentHeartbeat":
				_, err = contracts.DecodePrivateStrict[control.AgentHeartbeat](raw)
			case "HeartbeatResponse":
				_, err = contracts.DecodePrivateStrict[control.HeartbeatResponse](raw)
			default:
				t.Fatal("unknown slot fixture type")
			}
			if (err == nil) != item.Valid {
				t.Fatalf("valid=%t: %v", item.Valid, err)
			}
		})
	}
}
