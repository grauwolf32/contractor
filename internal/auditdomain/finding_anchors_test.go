package auditdomain

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func TestSharedFindingAnchorCorpus(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name         string
			File         string
			Line         int64
			QualifyLocal bool
			ZipBase64    string
			Derived      json.RawMessage
			Keys         []FindingAnchorKeys
		}
		Unavailable []json.RawMessage
		Invalid     []struct {
			Name    string
			Derived json.RawMessage
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "finding-anchor-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			basis, err := DecodeFindingAnchorDerivation(c.Derived)
			if err != nil {
				t.Fatal(err)
			}
			payload, err := base64.StdEncoding.DecodeString(c.ZipBase64)
			if err != nil {
				t.Fatal(err)
			}
			source := contracts.ExactContent(*basis.SourceRef, "application/zip", payload)
			proposal := FindingProposal{Locations: []FindingLocation{{File: c.File, Line: &c.Line}}}
			if c.QualifyLocal {
				proposal.StandardRefs = []StandardReference{{Scheme: "CWE", Version: "4.20", RequirementID: "CWE-321"}}
			}
			resolved, err := ResolveFindingAnchors(basis, "https://example.test:443/repo.git", strings.Repeat("a", 40), source, proposal)
			if err != nil {
				t.Fatal(err)
			}
			got, _ := strictjson.Canonical(resolved.Keys)
			want, _ := strictjson.Canonical(c.Keys)
			if string(got) != string(want) {
				t.Fatalf("keys=%s want=%s", got, want)
			}
			encoded, err := EncodeFindingAnchors(resolved)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := DecodeFindingAnchors(encoded); err != nil {
				t.Fatal(err)
			}
			shifted, err := ResolveFindingAnchors(basis, resolved.Repository, strings.Repeat("b", 40), source, proposal)
			if err != nil || shifted.Keys[0].SiteKey != resolved.Keys[0].SiteKey {
				t.Fatalf("site changed across snapshots: %v", err)
			}
			if c.QualifyLocal && shifted.Keys[0].IssueKey == resolved.Keys[0].IssueKey {
				t.Fatal("snapshot issue matched another commit")
			}
			if c.QualifyLocal {
				proposal.StandardRefs = nil
				if _, err := ResolveFindingAnchors(basis, resolved.Repository, resolved.Commit, source, proposal); err == nil {
					t.Fatal("local context attached to unrelated classification")
				}
			}
		})
	}
	for _, data := range fixture.Unavailable {
		if _, err := DecodeFindingAnchorDerivation(data); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range fixture.Invalid {
		t.Run(c.Name, func(t *testing.T) {
			if _, err := DecodeFindingAnchorDerivation(c.Derived); err == nil {
				t.Fatal("invalid private record accepted")
			}
		})
	}
}
