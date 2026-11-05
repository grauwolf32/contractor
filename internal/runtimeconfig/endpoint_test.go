package runtimeconfig

import (
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"testing"
)

func TestSharedRuntimeEndpointAdmission(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name  string
			URL   string
			Valid bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "runtime-endpoint-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			_, err := validateURL("endpoint", c.URL)
			if (err == nil) != c.Valid {
				t.Fatalf("admission valid=%v, want %v: %v", err == nil, c.Valid, err)
			}
		})
	}
}
