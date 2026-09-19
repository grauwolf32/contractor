package auditservice

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// The same wire cases are checked against the OpenAPI request union in public.
func TestFindingDecisionPublicContractCases(t *testing.T) {
	data, err := os.ReadFile("../../api/testdata/public/finding-decision-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name    string `json:"name"`
		Valid   bool   `json:"valid"`
		Request struct {
			Verdict           AnalystVerdict   `json:"verdict"`
			Severity          *FindingSeverity `json:"severity"`
			Rationale         string           `json:"rationale"`
			DuplicateTargetID *string          `json:"duplicateTargetId"`
		} `json:"request"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			err := validateFindingDecision(DecideFindingParams{
				OwnerID: "user-1", AuditID: "audit-1", RequestID: "review-1", DecisionID: "decision-1",
				ExpectedRequestRevision: 1, IdempotencyKey: "decision-test", RequestDigest: "sha256:" + strings.Repeat("a", 64),
				Verdict: tc.Request.Verdict, Severity: tc.Request.Severity,
				Rationale: tc.Request.Rationale, DuplicateTargetID: tc.Request.DuplicateTargetID,
			})
			if (err == nil) != tc.Valid {
				t.Fatalf("decision validity = %v, want %v: %v", err == nil, tc.Valid, err)
			}
		})
	}
}
