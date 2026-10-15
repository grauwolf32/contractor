package contracts

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func TestSharedPlannerSubtaskLimits(t *testing.T) {
	var fixture struct {
		Default int
		Maximum int
		Limits  []struct {
			Name      string
			Present   bool
			Value     json.RawMessage
			Valid     bool
			Effective int
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "planner-subtask-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Default != DefaultPlannerSubtasks || fixture.Maximum != MaxPlannerSubtasks {
		t.Fatal("shared Planner subtask bounds differ from the production contract")
	}
	for _, tc := range fixture.Limits {
		t.Run(tc.Name, func(t *testing.T) {
			data := []byte(`{}`)
			if tc.Present {
				data = []byte(fmt.Sprintf(`{"maxSubtasks":%s}`, tc.Value))
			}
			var holder struct {
				MaxSubtasks PlannerSubtaskLimit `json:"maxSubtasks,omitempty"`
			}
			err := strictjson.Decode(data, &holder)
			if (err == nil) != tc.Valid {
				t.Fatalf("valid=%v, decode error=%v", tc.Valid, err)
			}
			if err == nil && holder.MaxSubtasks.Effective() != tc.Effective {
				t.Fatalf("effective=%d, want %d", holder.MaxSubtasks.Effective(), tc.Effective)
			}
		})
	}
}
