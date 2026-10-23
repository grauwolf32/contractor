package scanplan

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
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
			if got := scan.ValidHost(tc.Host); got != tc.Accepted {
				t.Errorf("scan.ValidHost(%q) = %v, want %v", tc.Host, got, tc.Accepted)
			}
		})
	}
}
