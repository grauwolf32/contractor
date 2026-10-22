package auditdomain

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func TestSharedFindingFingerprintCases(t *testing.T) {
	var fixture struct {
		Vectors []struct {
			Name      string
			Tuple     any
			Canonical string
			Digest    string
		}
		AdapterCases []struct {
			Name         string
			Report       string
			Availability string
			Reason       string
			Value        string
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "finding-fingerprint-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, c := range fixture.Vectors {
		t.Run(c.Name, func(t *testing.T) {
			canonical, err := strictjson.Canonical(c.Tuple)
			if err != nil || string(canonical) != c.Canonical {
				t.Fatalf("canonical=%s: %v", canonical, err)
			}
			digest, err := contentdigest.CanonicalJSON(c.Tuple)
			if err != nil || digest != c.Digest {
				t.Fatalf("digest=%s: %v", digest, err)
			}
		})
	}
	for _, c := range fixture.AdapterCases {
		t.Run(c.Name, func(t *testing.T) {
			source, report := fingerprintDescriptors([]byte(c.Report))
			line := int64(3)
			got := FingerprintPinnedSARIF([]byte(c.Report), "https://example.test:443/repo.git", strings.Repeat("a", 40), source, report, []FindingLocation{{File: "src/a.py", Line: &line}})
			if got.Availability != c.Availability || got.Reason != c.Reason || got.Value != c.Value {
				t.Fatalf("projection=%+v", got)
			}
			encoded, err := EncodeFindingFingerprint(got)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := DecodeFindingFingerprint(encoded); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func fingerprintDescriptors(data []byte) (contracts.ExactArtifact, contracts.ExactArtifact) {
	revision := "1"
	return contracts.ExactContent(contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &revision}, "application/zip", []byte("source bytes")), contracts.ExactContent(contracts.ArtifactRef{Namespace: "inputs", Name: "report", Revision: &revision}, "application/sarif+json", data)
}

func TestFindingFingerprintStrictProjection(t *testing.T) {
	valid, err := EncodeFindingFingerprint(UnavailableFindingFingerprint("not_provided"))
	if err != nil {
		t.Fatal(err)
	}
	for _, data := range []string{
		`{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable"}`,
		`{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable","reason":"invented"}`,
		strings.TrimSuffix(string(valid), "}") + `,"value":null}`,
		strings.TrimSuffix(string(valid), "}") + `,"Value":"forged"}`,
		strings.TrimSuffix(string(valid), "}") + `,"reason":"not_recorded"}`,
		strings.TrimSuffix(string(valid), "}") + `,"report":{}}`,
		"null",
	} {
		if _, err := DecodeFindingFingerprint([]byte(data)); err == nil {
			t.Errorf("accepted %s", data)
		}
	}
	large := make([]byte, MaximumFingerprintReportBytes+1)
	source, report := fingerprintDescriptors(large)
	if got := FingerprintPinnedSARIF(large, "", "", source, report, nil); got.Reason != "input_limit" {
		t.Fatal(got)
	}
}
