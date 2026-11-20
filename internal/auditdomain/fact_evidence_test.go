package auditdomain

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedFactEvidenceFingerprint(t *testing.T) {
	var fixture struct {
		Algorithm   string
		MaxEvidence int
		Cases       []struct {
			Name, Value, GroupKey string
			Evidence              []contracts.ExactArtifact
			Projection            json.RawMessage
		}
		PublicDocuments []struct {
			Schema   string
			Valid    bool
			Document json.RawMessage
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "finding-fact-evidence-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Algorithm != factEvidenceAlgorithm || fixture.MaxEvidence != MaximumFactEvidence {
		t.Fatal("fact equality constants drifted")
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			projection, err := FactEvidenceFingerprint(c.Evidence)
			if err != nil || projection.Value != c.Value {
				t.Fatalf("%+v %v", projection, err)
			}
			stored, err := DecodeFindingFingerprint(c.Projection)
			if err != nil || stored != projection {
				t.Fatal(stored, err)
			}
			encoded, err := EncodeFindingFingerprint(projection)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := DecodeFindingFingerprint(encoded); err != nil {
				t.Fatal(err)
			}
			key, err := FindingGroupKey("project-a", projection.Algorithm, projection.Value)
			if err != nil || key != c.GroupKey {
				t.Fatal(key, err)
			}
		})
	}
	for _, c := range fixture.PublicDocuments {
		if c.Schema != "FindingFingerprintProjection" {
			continue
		}
		_, err := DecodeFindingFingerprint(c.Document)
		if (err == nil) != c.Valid {
			t.Fatalf("strict fact projection valid=%v err=%v", c.Valid, err)
		}
	}
	good := contracts.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: "proof", Name: "selected", Revision: pointerForFact("r1")}, Digest: "sha256:" + strings.Repeat("a", 64), MediaType: "text/plain", SizeBytes: 1}
	badDigest := good
	badDigest.Digest = "model-supplied-invalid"
	noMetadata := good
	noMetadata.MediaType = ""
	for _, evidence := range [][]contracts.ExactArtifact{nil, {}, {badDigest}, {noMetadata}, {good, good, good, good, good}} {
		if _, err := FactEvidenceFingerprint(evidence); err == nil {
			t.Fatal("invalid evidence accepted")
		}
	}
}
func pointerForFact(s string) *string { return &s }
