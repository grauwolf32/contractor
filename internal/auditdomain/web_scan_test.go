package auditdomain

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedWebScanSettings(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "audit-web-scan-settings.json")
	var cases []struct {
		Name     string          `json:"name"`
		Settings json.RawMessage `json:"settings"`
		Valid    bool            `json:"valid"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("empty fixture gate")
	}
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			_, err := DecodeWebScanSettings(test.Settings)
			if (err == nil) != test.Valid {
				t.Fatalf("settings valid=%t: %v", test.Valid, err)
			}
		})
	}
}
