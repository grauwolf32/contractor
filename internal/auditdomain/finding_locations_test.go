package auditdomain

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestFindingLocationSharedContract(t *testing.T) {
	raw, err := os.ReadFile("testdata/finding-locations.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name     string          `json:"name"`
		Valid    bool            `json:"valid"`
		Location json.RawMessage `json:"location"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			var location FindingLocation
			err := json.Unmarshal(test.Location, &location)
			if (err == nil) != test.Valid {
				t.Fatalf("valid=%v, error=%v", test.Valid, err)
			}
			if !test.Valid {
				return
			}
			proposal := findingFixture()
			proposal.Locations = []FindingLocation{location}
			encoded, err := EncodeFindingProposal(proposal)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := DecodeFindingProposal(encoded)
			if err != nil || !reflect.DeepEqual(decoded, proposal) {
				t.Fatalf("finding did not round-trip: %v", err)
			}
			if !bytes.Contains(encoded, []byte(`"subject":null`)) {
				t.Fatal("unknown subject was invented")
			}
		})
	}
}

func TestFindingRejectsNullUnknownFieldsAndUnsupportedSchema(t *testing.T) {
	proposal := findingFixture()
	encoded, err := EncodeFindingProposal(proposal)
	if err != nil {
		t.Fatal(err)
	}
	for _, member := range []string{
		`"locations":null`, `"http_exchange":null`, `"Locations":[]`,
	} {
		candidate := append([]byte(nil), encoded[:len(encoded)-1]...)
		candidate = append(candidate, []byte(","+member+"}")...)
		if _, err := DecodeFindingProposal(candidate); err == nil {
			t.Fatalf("accepted %s", member)
		}
	}
	for _, replacement := range []string{`"subject":{}`, `"subject":{"kind":"code"}`} {
		candidate := bytes.Replace(encoded, []byte(`"subject":null`), []byte(replacement), 1)
		if _, err := DecodeFindingProposal(candidate); err == nil {
			t.Fatalf("accepted %s", replacement)
		}
	}
	proposal.Schema = "contractor.audit.finding-proposal.unsupported"
	if _, err := EncodeFindingProposal(proposal); err == nil {
		t.Fatal("accepted an unsupported schema")
	}

}

func TestFindingHTTPRetainsRequestAndExactResponseLink(t *testing.T) {
	proposal := findingFixture()
	status := 200
	proposal.EvidenceIDs = []string{"evidence-1"}
	proposal.HTTPExchange = &FindingHTTPExchange{
		RequestID: 1, RequestTag: "r-test-h000001", ResponseBodyEvidenceID: "evidence-1",
		Attempts: []FindingHTTPAttempt{{
			Method: "POST", URL: "https://example.test/?a=%2f&a=2", Status: &status,
			Headers:    []FindingHTTPHeader{{Name: "Authorization", Value: "Bearer test-token"}},
			BodyBase64: "AP8=",
		}},
	}
	encoded, err := EncodeFindingProposal(proposal)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeFindingProposal(encoded)
	if err != nil || !reflect.DeepEqual(decoded, proposal) {
		t.Fatalf("HTTP bytes changed: %v", err)
	}
	proposal.EvidenceIDs = []string{}
	if _, err := EncodeFindingProposal(proposal); err == nil {
		t.Fatal("dangling body evidence was accepted")
	}
}

func findingFixture() FindingProposal {
	return FindingProposal{
		Schema: FindingProposalSchema, ClientKey: "call-test", Title: "Finding", Description: "Observation",
		Preconditions: []string{}, StandardRefs: []StandardReference{}, EvidenceIDs: []string{},
		ProposedChecks: []ProposedCheck{}, Limitations: []string{},
	}
}
