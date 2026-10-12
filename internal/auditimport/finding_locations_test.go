package auditimport

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestReportPreservesExplicitLocationsAndUnknownSubject(t *testing.T) {
	line, status := int64(42), 200
	document := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "call-test",
		Title: "Source and HTTP observation", Description: "An observed response with source context.",
		Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{}, EvidenceIDs: []string{},
		ProposedChecks: []auditdomain.ProposedCheck{}, Limitations: []string{},
		Locations: []auditdomain.FindingLocation{
			{File: "src/order.py", Line: &line},
			{URL: "https://target.test/order?x=%2f&x=2#detail", Method: "GET"},
		},
		HTTPExchange: &auditdomain.FindingHTTPExchange{RequestID: 1, RequestTag: "request-test", Attempts: []auditdomain.FindingHTTPAttempt{
			{URL: "https://target.test/order?x=%2f&x=2", Method: "GET", Headers: []auditdomain.FindingHTTPHeader{}, BodyBase64: "", Status: &status},
		}},
	}
	payload, err := auditdomain.EncodeFindingProposal(document)
	if err != nil {
		t.Fatal(err)
	}
	revision := "proposal-r1"
	artifact := auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "findings", Name: "proposal", Revision: &revision},
		Digest: auditdomain.DigestBytes(payload), SizeBytes: int64(len(payload)), MediaType: "application/json",
	}
	// The report reads the projection that intake stored with the receipt.
	stored, err := auditdomain.EncodeFindingReportProjection(auditdomain.ProjectFindingReport(document))
	if err != nil {
		t.Fatal(err)
	}
	projection, err := auditdomain.DecodeFindingReportProjection(stored)
	if err != nil {
		t.Fatal(err)
	}
	finding := newReportFinding(auditstore.ReportFinding{
		FindingID: "finding-1", State: "proposed", FirstProposal: artifact, Revision: 1, Projection: projection,
	})
	if finding.Subject != nil || !reflect.DeepEqual(finding.Locations, document.Locations) || !reflect.DeepEqual(finding.HTTPExchange, document.HTTPExchange) {
		t.Fatalf("report lost recorded context: %+v", finding)
	}
	encoded, err := json.Marshal(finding)
	if err != nil {
		t.Fatal(err)
	}
	var public map[string]any
	if err := json.Unmarshal(encoded, &public); err != nil {
		t.Fatal(err)
	}
	subject, exists := public["subject"]
	if !exists || subject != nil {
		t.Fatal("unknown subject was invented or omitted")
	}
}
