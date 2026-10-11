package auditdomain

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestFindingReportProjectionRoundTripsTheReportFields(t *testing.T) {
	line, status := int64(7), 302
	proposal := findingFixture()
	proposal.Subject = &FindingSubject{Kind: "endpoint", Key: "orders"}
	proposal.Hypothesis = "The redirect target is not validated."
	proposal.SeveritySuggestion = "high"
	proposal.StandardRefs = []StandardReference{{Scheme: "asvs", Version: "4.0.3", RequirementID: "V5.1.5"}}
	proposal.Limitations = []string{"Only the staging host was exercised."}
	proposal.EvidenceIDs = []string{"evidence-1"}
	proposal.Locations = []FindingLocation{{File: "src/redirect.py", Line: &line}, {URL: "https://target.test/r?to=x", Method: "GET"}}
	proposal.HTTPExchange = &FindingHTTPExchange{
		RequestID: 4, RequestTag: "redirect", ResponseBodyEvidenceID: "evidence-1",
		Attempts: []FindingHTTPAttempt{{
			URL: "https://target.test/r?to=x", Method: "GET", Headers: []FindingHTTPHeader{{Name: "Accept", Value: "*/*"}},
			Status: &status, ResponseHeaders: []FindingHTTPHeader{{Name: "Location", Value: "https://x.test"}},
		}},
	}
	if _, err := EncodeFindingProposal(proposal); err != nil {
		t.Fatal(err)
	}
	encoded, err := EncodeFindingReportProjection(ProjectFindingReport(proposal))
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeFindingReportProjection(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, ProjectFindingReport(proposal)) {
		t.Fatalf("projection round trip = %+v", decoded)
	}
	if strings.Contains(string(encoded), "client_key") || strings.Contains(string(encoded), "evidence_ids") {
		t.Fatalf("projection keeps proposal-only fields: %s", encoded)
	}
}

func TestFindingReportProjectionDecodesStrictly(t *testing.T) {
	valid := `{"description":"d","limitations":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"t"}`
	if _, err := DecodeFindingReportProjection([]byte(valid)); err != nil {
		t.Fatal(err)
	}
	for name, document := range map[string]string{
		"unknown field":       `{"description":"d","limitations":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"t","schema":"x"}`,
		"absent subject":      `{"description":"d","limitations":[],"severity_suggestion":"","standard_refs":[],"title":"t"}`,
		"null limitations":    `{"description":"d","limitations":null,"severity_suggestion":"","standard_refs":[],"subject":null,"title":"t"}`,
		"blank title":         `{"description":"d","limitations":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":" "}`,
		"unknown severity":    `{"description":"d","limitations":[],"severity_suggestion":"urgent","standard_refs":[],"subject":null,"title":"t"}`,
		"duplicate key":       `{"description":"d","description":"e","limitations":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"t"}`,
		"trailing data":       valid + `{}`,
		"subject extra field": `{"description":"d","limitations":[],"severity_suggestion":"","standard_refs":[],"subject":{"kind":"k","key":"v","x":1},"title":"t"}`,
		"exchange extra":      `{"description":"d","http_exchange":{"request_id":1,"request_tag":"r","attempts":[],"x":1},"limitations":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"t"}`,
		"exchange evidence":   `{"description":"d","http_exchange":{"request_id":1,"request_tag":"r","attempts":[{"method":"GET","url":"https://t.test/","headers":[],"body_base64":"","status":200}],"response_body_evidence_id":"bad id"},"limitations":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"t"}`,
	} {
		if _, err := DecodeFindingReportProjection([]byte(document)); err == nil {
			t.Errorf("%s: decoded", name)
		}
	}
}

func TestFindingProposalAndProjectionShareTheReportFieldRules(t *testing.T) {
	for name, mutate := range map[string]func(*FindingProposal){
		"blank description": func(p *FindingProposal) { p.Description = "" },
		"invalid subject":   func(p *FindingProposal) { p.Subject = &FindingSubject{Kind: "bad kind", Key: "k"} },
		"standard ref": func(p *FindingProposal) {
			p.StandardRefs = []StandardReference{{Scheme: "", Version: "1", RequirementID: "r"}}
		},
		"limitation": func(p *FindingProposal) { p.Limitations = []string{""} },
		"location":   func(p *FindingProposal) { p.Locations = []FindingLocation{{File: "../escape.py"}} },
	} {
		proposal := findingFixture()
		mutate(&proposal)
		_, proposalErr := EncodeFindingProposal(proposal)
		_, projectionErr := EncodeFindingReportProjection(ProjectFindingReport(proposal))
		var proposalCode, projectionCode *ValidationError
		if !errors.As(proposalErr, &proposalCode) || !errors.As(projectionErr, &projectionCode) ||
			proposalCode.Code != projectionCode.Code || proposalCode.Field != projectionCode.Field {
			t.Errorf("%s: proposal error %v, projection error %v", name, proposalErr, projectionErr)
		}
	}
	// The evidence reference is proposal-only: the projection keeps the
	// exchange shape but has no evidence IDs to resolve.
	proposal := findingFixture()
	status := 200
	proposal.HTTPExchange = &FindingHTTPExchange{
		RequestID: 1, RequestTag: "r", ResponseBodyEvidenceID: "evidence-9",
		Attempts: []FindingHTTPAttempt{{URL: "https://t.test/", Method: "GET", Headers: []FindingHTTPHeader{}, Status: &status}},
	}
	var reference *ValidationError
	if _, err := EncodeFindingProposal(proposal); !errors.As(err, &reference) || reference.Code != CodeReferenceInvalid {
		t.Fatalf("unresolved evidence reference = %v", err)
	}
	if _, err := EncodeFindingReportProjection(ProjectFindingReport(proposal)); err != nil {
		t.Fatalf("projection of a resolvable exchange = %v", err)
	}
}
