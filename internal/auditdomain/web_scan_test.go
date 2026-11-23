package auditdomain

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedWebScanParameters(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "audit-web-scan-parameters.json")
	var cases []struct {
		Name       string            `json:"name"`
		Parameters map[string]string `json:"parameters"`
		Valid      bool              `json:"valid"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("empty fixture gate")
	}
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			_, _, err := PrepareWebTarget([]byte(`{"schema":"contractor.audit.web-target.v1","url":"https://target.test"}`), test.Parameters)
			if (err == nil) != test.Valid {
				t.Fatalf("settings valid=%t: %v", test.Valid, err)
			}
		})
	}
}
