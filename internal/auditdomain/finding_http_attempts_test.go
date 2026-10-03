package auditdomain

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestFindingHTTPAttemptSharedContract(t *testing.T) {
	raw, err := os.ReadFile("testdata/finding-http-attempts.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name   string `json:"name"`
		URL    string `json:"url"`
		Repeat string `json:"repeat"`
		Count  int    `json:"count"`
		Status int    `json:"status"`
		Valid  bool   `json:"valid"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			attempt := FindingHTTPAttempt{
				Method: "GET", URL: test.URL + strings.Repeat(test.Repeat, test.Count),
				Headers: []FindingHTTPHeader{}, BodyBase64: "", Status: &test.Status,
			}
			if valid := validateFindingHTTPAttempt(attempt) == nil; valid != test.Valid {
				t.Fatalf("attempt validity = %v, want %v", valid, test.Valid)
			}
		})
	}
}
