package scanplan

import (
	"encoding/json"
	"os"
	"testing"
)

func TestHostSyntaxMatchesRuntimeCases(t *testing.T) {
	data, err := os.ReadFile("../../api/testdata/scan-host-syntax.json")
	if err != nil {
		t.Fatal(err)
	}
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
