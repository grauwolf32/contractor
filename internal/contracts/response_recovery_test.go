package contracts_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedDocumentResponsePolicy(t *testing.T) {
	var cases []struct {
		Mode, Runtime, Ref string
		Tools              []string
		Valid              bool
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "document-response-policy.json"), &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		ref := strings.Split(c.Ref, "@")
		template := contracts.ResolvedAgentTemplate{ResponseRecovery: c.Mode, Runtime: contracts.WorkerRuntimeRef{RuntimeID: c.Runtime, Version: "1"}, Toolsets: []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: ref[0], Version: ref[1]}, Tools: c.Tools}}}
		if err := template.ValidateResponseRecovery(); (err == nil) != c.Valid {
			t.Fatalf("%+v: %v", c, err)
		}
	}
}
