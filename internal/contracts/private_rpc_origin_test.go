package contracts_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPrivateRPCOrigins(t *testing.T) {
	var cases struct{ Valid, Invalid []string }
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "private-rpc-origin-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	for _, value := range cases.Valid {
		if _, err := contracts.ParsePrivateRPCOrigin("endpoint", value); err != nil {
			t.Fatalf("valid origin rejected: %v", err)
		}
	}
	for _, value := range cases.Invalid {
		if _, err := contracts.ParsePrivateRPCOrigin("endpoint", value); err == nil {
			t.Fatalf("invalid origin accepted: %q", value)
		}
	}
}
