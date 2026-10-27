package auditdomain

import (
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"testing"
)

func TestSharedAuditHTTPObservationMetadata(t *testing.T) {
	raw := contractstest.ReadFile(t, "api", "testdata", "audit-completion", "evidence-observations.json")
	var fixture struct {
		Capture json.RawMessage `json:"capture"`
		Cases   []struct {
			Name     string
			Valid    bool
			Evidence json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("empty observation cases")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			var metadata Evidence
			err := json.Unmarshal(tc.Evidence, &metadata)
			var evidence []byte
			if err == nil {
				evidence, err = EncodeEvidence(EvidenceEnvelope{Schema: EvidenceSchema, Evidence: []Evidence{metadata}})
			}
			if err == nil {
				results, _ := EncodeCheckResultSet(CheckResultSet{Schema: CheckResultsSchema, ExecutionManifestDigest: "sha256:" + repeatHex("a", 64), Results: []CheckResult{{ItemKey: "check", SubjectKey: "subject", Assessment: "inconclusive", Summary: "Observed", EvidenceIDs: []string{metadata.ID}, Coverage: ResultCoverage{Requested: []string{"observation"}, Completed: []string{}, Gaps: []string{"gap"}}, Proposals: []ProposalSelection{}}}})
				payload, _, buildErr := BuildPackage("http-result", PackageKindCheckResults, "", []PackageInput{{ID: CheckResultsMemberID, Path: "check-results.json", MediaType: JSONMediaType, Data: results}, {ID: EvidenceMemberID, Path: "evidence.json", MediaType: JSONMediaType, Data: evidence}, {ID: metadata.ContentMemberID, Path: "evidence/http.json", MediaType: JSONMediaType, Data: fixture.Capture}})
				if buildErr != nil {
					t.Fatal(buildErr)
				}
				_, err = DecodeCheckResultPackage(payload)
			}
			if (err == nil) != tc.Valid {
				t.Fatalf("valid=%v err=%v", tc.Valid, err)
			}
		})
	}
}
