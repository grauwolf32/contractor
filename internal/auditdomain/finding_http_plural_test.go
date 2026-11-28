package auditdomain

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedFindingMultipleHTTPProjections(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name     string
			Valid    bool
			Proposal json.RawMessage
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "finding-http-evidence-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			value, err := DecodeFindingProposal(test.Proposal)
			if (err == nil) != test.Valid {
				t.Fatalf("valid=%t: %v", test.Valid, err)
			}
			if err == nil && len(value.HTTPExchanges) != 2 {
				t.Fatal("lost projection")
			}
		})
	}
}

func TestFindingHTTPProjectionCountAndEveryEvidenceReference(t *testing.T) {
	proposal := findingFixture()
	status := 200
	for index := range MaximumEvidencePerItem {
		id := fmt.Sprintf("evidence-%d", index+1)
		proposal.EvidenceIDs = append(proposal.EvidenceIDs, id)
		proposal.HTTPExchanges = append(proposal.HTTPExchanges, FindingHTTPExchange{RequestID: int64(index + 1), RequestTag: "capture", ResponseBodyEvidenceID: id, Attempts: []FindingHTTPAttempt{{URL: "https://target.test/", Method: "GET", Headers: []FindingHTTPHeader{}, Status: &status}}})
	}
	if _, err := EncodeFindingProposal(proposal); err != nil {
		t.Fatal(err)
	}
	proposal.HTTPExchanges[len(proposal.HTTPExchanges)-1].ResponseBodyEvidenceID = "absent"
	if _, err := EncodeFindingProposal(proposal); err == nil {
		t.Fatal("unresolved final reference accepted")
	}
	proposal.HTTPExchanges[len(proposal.HTTPExchanges)-1].ResponseBodyEvidenceID = "evidence-256"
	proposal.HTTPExchanges = append(proposal.HTTPExchanges, proposal.HTTPExchanges[0])
	if _, err := EncodeFindingReportProjection(ProjectFindingReport(proposal)); err == nil {
		t.Fatal("257 projections accepted")
	}
}
