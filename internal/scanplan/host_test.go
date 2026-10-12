package scanplan

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedHostSyntaxMatchesRuntimeCases(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "scan-host-syntax.json")
	var cases []struct {
		Host     string `json:"host"`
		Accepted bool   `json:"accepted"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) < 20 {
		t.Fatal("shared host cases missing")
	}
	for _, tc := range cases {
		t.Run(tc.Host, func(t *testing.T) {
			if got := scanHost(tc.Host); got != tc.Accepted {
				t.Errorf("scanHost(%q) = %v, want %v", tc.Host, got, tc.Accepted)
			}
		})
	}
}
