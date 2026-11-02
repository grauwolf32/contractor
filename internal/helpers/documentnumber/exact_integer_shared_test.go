package documentnumber_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/documentnumber"
	"github.com/grauwolf32/contractor/internal/helpers/exactinteger"
)

func TestSharedExactIntegerBoundary(t *testing.T) {
	var fixture struct {
		Maximum int64 `json:"maximum"`
		Cases   []struct {
			Text    string `json:"text"`
			Integer bool   `json:"integer"`
			Float   bool   `json:"float"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "exact-integer-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("the boundary fixture is empty")
	}
	for name, maximum := range map[string]int64{"foundation": exactinteger.Max, "contracts": contracts.MaxExactInteger} {
		if fixture.Maximum != maximum {
			t.Errorf("%s boundary=%d, want %d", name, maximum, fixture.Maximum)
		}
	}
	for _, item := range fixture.Cases {
		t.Run(item.Text, func(t *testing.T) {
			_, integer := documentnumber.Integer(item.Text)
			_, floating := documentnumber.Float(item.Text)
			if integer != item.Integer || floating != item.Float {
				t.Fatalf("integer=%t float=%t, want %t %t", integer, floating, item.Integer, item.Float)
			}
		})
	}
}
