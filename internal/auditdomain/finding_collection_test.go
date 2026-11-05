package auditdomain

import (
	"bytes"
	"encoding/json"
	"reflect"
	"sort"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// The process tests (tests/e2e) feed the Server and the Runtime this
// collection; their copy must stay the shared fixture byte for byte.
func TestSharedFindingCollectionProcessCopyMatches(t *testing.T) {
	shared := contractstest.ReadFile(t, "api", "testdata", "findings", "finding-collection-v2.fixture.json")
	process := contractstest.ReadFile(t, "tests", "e2e", "testdata", "findings", "finding-collection-v2.fixture.json")
	if !bytes.Equal(process, shared) {
		t.Fatal("tests/e2e/testdata/findings/finding-collection-v2.fixture.json differs from the shared fixture")
	}
}

// Frozen values were produced independently with Python hashlib/json/zipfile.
// The ASCII metadata keys and integral numbers need no extra JCS normalization.
func collectionFixture(t *testing.T) (FindingCollection, map[string][]byte, string, string) {
	t.Helper()
	raw := contractstest.ReadFile(t, "api", "testdata", "findings", "finding-collection-v2.fixture.json")
	var fixture struct {
		Collection    FindingCollection `json:"collection"`
		Contents      map[string]string `json:"contents"`
		PackageID     string            `json:"package_id"`
		PackageDigest string            `json:"package_digest"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	contents := make(map[string][]byte, len(fixture.Contents))
	for id, body := range fixture.Contents {
		contents[id] = []byte(body)
	}
	return fixture.Collection, contents, fixture.PackageID, fixture.PackageDigest
}

func TestSharedFindingCollectionConformanceAndConsumerTargets(t *testing.T) {
	value, contents, packageID, packageDigest := collectionFixture(t)
	payload, err := BuildFindingCollectionPackage(value, contents)
	if err != nil {
		t.Fatal(err)
	}
	decoded, pkg, err := DecodeFindingCollectionPackage(payload)
	if err != nil || !reflect.DeepEqual(decoded, value) {
		t.Fatalf("collection provenance changed: %+v, %v", decoded, err)
	}
	if pkg.Manifest.PackageID != packageID || pkg.Digest != packageDigest {
		t.Fatalf("conformance mismatch: package=%s digest=%s", pkg.Manifest.PackageID, pkg.Digest)
	}
	again, err := BuildFindingCollectionPackage(decoded, contents)
	if err != nil || !bytes.Equal(again, payload) {
		t.Fatalf("rebuilding the snapshot changed bytes: %v", err)
	}
	var fixture struct {
		Targets map[string]contracts.ArtifactRef `json:"consumer_targets"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "findings", "finding-collection-v2.fixture.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	targets := fixture.Targets
	if len(targets) != len(decoded.Documents) {
		t.Fatal("fixture must name every consumer target")
	}
	seen := make(map[string]bool)
	for _, document := range decoded.Documents {
		ref, ok := targets[document.ID]
		if !ok || ref.Validate() != nil || ref.Revision != nil || ref.Name != document.ID {
			t.Fatalf("invalid consumer write target: %+v", ref)
		}
		key := ref.Namespace + "/" + ref.Name
		if seen[key] {
			t.Fatal("colliding source bindings aliased in the consumer")
		}
		seen[key] = true
		member, _ := pkg.MemberByID(document.ID)
		if !bytes.Equal(member.Data(), contents[document.ID]) {
			t.Fatal("original bytes were changed")
		}
	}
	first, second := value.Entries[0], value.Entries[1]
	if targets[first.Evidence[0].DocumentID] == targets[second.Evidence[0].DocumentID] {
		t.Fatal("identical evidence from different Runs lost its scoped identity")
	}
	if first.AuditOrigin != nil || first.Finding != nil || second.Finding == nil || second.Finding.Verdict != "needs_evidence" {
		t.Fatal("invented an Audit origin or decision")
	}
	value.SnapshotAt = "2026-09-06T09:00:01Z"
	changed, err := BuildFindingCollectionPackage(value, contents)
	if err != nil {
		t.Fatal(err)
	}
	_, changedPackage, err := DecodeFindingCollectionPackage(changed)
	if err != nil || changedPackage.Digest == pkg.Digest {
		t.Fatalf("different snapshot reused package digest: %v", err)
	}
}

func TestSharedFindingCollectionEmptySelectionAndOptionalReviews(t *testing.T) {
	value, _, _, _ := collectionFixture(t)
	value.Entries = []FindingCollectionEntry{}
	value.Documents = []FindingCollectionDocument{}
	payload, err := BuildFindingCollectionPackage(value, nil)
	if err != nil {
		t.Fatal(err)
	}
	decoded, _, err := DecodeFindingCollectionPackage(payload)
	if err != nil || decoded.Entries == nil || len(decoded.Entries) != 0 {
		t.Fatalf("empty selection failed: %+v %v", decoded, err)
	}
	for _, tc := range []struct{ state, verdict, severity, target string }{
		{"proposed", "reopen", "", ""}, {"confirmed", "true_positive", "high", ""},
		{"rejected", "false_positive", "", ""}, {"duplicate", "duplicate", "", "target"}, {"needs-evidence", "needs_evidence", "", ""},
	} {
		t.Run(tc.state, func(t *testing.T) {
			value, contents, _, _ := collectionFixture(t)
			value.Entries[1].Finding = &FindingCollectionFinding{ProjectID: "project", FindingID: "finding", Revision: 1, State: &tc.state, Verdict: tc.verdict, Severity: tc.severity, DuplicateTargetID: tc.target}
			if _, err := BuildFindingCollectionPackage(value, contents); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestSharedFindingCollectionSharesEvidenceWithoutMergingReceipts(t *testing.T) {
	value, contents, _, _ := collectionFixture(t)
	sharedID := value.Entries[0].Evidence[0].DocumentID
	replacedID := value.Entries[1].Evidence[0].DocumentID
	value.Entries[1].Evidence[0].DocumentID = sharedID
	for i, document := range value.Documents {
		if document.ID == replacedID {
			value.Documents = append(value.Documents[:i], value.Documents[i+1:]...)
			break
		}
	}
	delete(contents, replacedID)
	payload, err := BuildFindingCollectionPackage(value, contents)
	if err != nil {
		t.Fatal(err)
	}
	decoded, _, err := DecodeFindingCollectionPackage(payload)
	if err != nil || len(decoded.Entries) != 2 || len(decoded.Documents) != 3 || decoded.Entries[0].ReceiptID == decoded.Entries[1].ReceiptID {
		t.Fatalf("sharing evidence merged proposals: %+v %v", decoded, err)
	}
}

func TestSharedFindingCollectionAuditHoldPreservesMembershipWithoutReview(t *testing.T) {
	value, contents, _, _ := collectionFixture(t)
	value.Sources = []FindingCollectionSource{{Kind: "audit", ID: "audit-a"}}
	for i := range value.Entries {
		value.Entries[i].AuditOrigin = nil
		value.Entries[i].SelectedSources = []FindingCollectionSource{{Kind: "audit", ID: "audit-a"}}
		value.Entries[i].AuditHolds = []string{"audit-a"}
	}
	payload, err := BuildFindingCollectionPackage(value, contents)
	if err != nil {
		t.Fatal(err)
	}
	decoded, _, err := DecodeFindingCollectionPackage(payload)
	if err != nil || !reflect.DeepEqual(decoded, value) {
		t.Fatalf("hold provenance: %+v %v", decoded, err)
	}
	for _, holds := range [][]string{{"audit-a", "audit-a"}, {"audit-z", "audit-a"}, {"../audit"}} {
		value.Entries[0].AuditHolds = holds
		if _, err := EncodeFindingCollection(value); err == nil {
			t.Fatalf("invalid hold membership accepted: %v", holds)
		}
	}
}

func TestSharedFindingCollectionRejectsInvalidMetadata(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*FindingCollection)
		code   string
	}{
		{"schema", func(v *FindingCollection) { v.Schema = "contractor.findings.collection.v1" }, CodeSchemaUnsupported},
		{"timestamp", func(v *FindingCollection) { v.SnapshotAt = "2026-09-06T12:00:00+03:00" }, CodeInvalid},
		{"no sources", func(v *FindingCollection) { v.Sources = nil }, CodeLimitExceeded},
		{"missing entries", func(v *FindingCollection) { v.Entries = nil }, CodeLimitExceeded},
		{"missing documents", func(v *FindingCollection) { v.Documents = nil }, CodeLimitExceeded},
		{"source order", func(v *FindingCollection) { v.Sources[0], v.Sources[1] = v.Sources[1], v.Sources[0] }, CodeInvalid},
		{"duplicate source", func(v *FindingCollection) { v.Sources[1] = v.Sources[0] }, CodeInvalid},
		{"source membership", func(v *FindingCollection) { v.Sources[1].ID = "run-foreign" }, CodeReferenceInvalid},
		{"versionless ref", func(v *FindingCollection) { v.Documents[0].Ref.Revision = nil }, CodeReferenceInvalid},
		{"invalid namespace", func(v *FindingCollection) { v.Documents[0].Ref.Namespace = "../outside" }, CodeReferenceInvalid},
		{"invalid scope", func(v *FindingCollection) { v.Documents[0].Scope.Kind = "audit" }, CodeReferenceInvalid},
		{"scope identity", func(v *FindingCollection) { v.Documents[0].Scope.ID = "run-foreign" }, CodeReferenceInvalid},
		{"digest", func(v *FindingCollection) { v.Documents[0].Digest = "sha256:wrong" }, CodeReferenceInvalid},
		{"negative bytes", func(v *FindingCollection) { v.Documents[0].SizeBytes = -1 }, CodeReferenceInvalid},
		{"MIME parameters", func(v *FindingCollection) { v.Documents[0].MediaType = "text/plain; charset=utf-8" }, CodeReferenceInvalid},
		{"duplicate document", func(v *FindingCollection) { v.Documents[1] = v.Documents[0] }, CodeReferenceInvalid},
		{"unreferenced document", func(v *FindingCollection) { v.Entries[0].Evidence[0].DocumentID = v.Entries[1].Evidence[0].DocumentID }, CodeReferenceInvalid},
		{"duplicate receipt", func(v *FindingCollection) { v.Entries[1].ReceiptID = v.Entries[0].ReceiptID }, CodeInvalid},
		{"duplicate proposal", func(v *FindingCollection) { v.Entries[1].ProposalID = v.Entries[0].ProposalID }, CodeInvalid},
		{"entry order", func(v *FindingCollection) { v.Entries[0], v.Entries[1] = v.Entries[1], v.Entries[0] }, CodeInvalid},
		{"proposal document", func(v *FindingCollection) { v.Entries[0].ProposalDocumentID = "doc-missing" }, CodeReferenceInvalid},
		{"missing evidence", func(v *FindingCollection) { v.Entries[0].Evidence = nil }, CodeLimitExceeded},
		{"missing evidence document", func(v *FindingCollection) { v.Entries[0].Evidence[0].DocumentID = "doc-missing" }, CodeReferenceInvalid},
		{"duplicate evidence", func(v *FindingCollection) {
			v.Entries[0].Evidence = append(v.Entries[0].Evidence, v.Entries[0].Evidence[0])
		}, CodeInvalid},
		{"unknown retention", func(v *FindingCollection) { v.Entries[0].Retention = "confirmed" }, CodeInvalid},
		{"invalid origin", func(v *FindingCollection) { v.Entries[1].AuditOrigin.ExecutionID = "" }, CodeInvalid},
		{"missing selected sources", func(v *FindingCollection) { v.Entries[0].SelectedSources = nil }, CodeLimitExceeded},
		{"foreign selected source", func(v *FindingCollection) {
			v.Entries[0].SelectedSources = []FindingCollectionSource{{Kind: "audit", ID: "foreign"}}
		}, CodeReferenceInvalid},
		{"zero revision", func(v *FindingCollection) {
			v.Entries[1].Finding = &FindingCollectionFinding{ProjectID: "project", FindingID: "finding"}
		}, CodeInvalid},
		{"unsafe integer", func(v *FindingCollection) {
			v.Entries[1].Finding = &FindingCollectionFinding{ProjectID: "project", FindingID: "finding", Revision: 1 << 53}
		}, CodeInvalid},
		{"entry limit", func(v *FindingCollection) { v.Entries = make([]FindingCollectionEntry, MaximumCollectionEntries+1) }, CodeLimitExceeded},
		{"document limit", func(v *FindingCollection) {
			v.Documents = make([]FindingCollectionDocument, MaximumCollectionDocuments+1)
		}, CodeLimitExceeded},
		{"source limit", func(v *FindingCollection) { v.Sources = make([]FindingCollectionSource, MaximumCollectionSources+1) }, CodeLimitExceeded},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			value, _, _, _ := collectionFixture(t)
			tc.mutate(&value)
			_, err := EncodeFindingCollection(value)
			if testDomainErrorCode(err) != tc.code {
				t.Fatalf("error = %v, want %s", err, tc.code)
			}
		})
	}
}

func TestSharedFindingCollectionRejectsAmbiguousJSON(t *testing.T) {
	value, _, _, _ := collectionFixture(t)
	valid, err := EncodeFindingCollection(value)
	if err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string][]byte{
		"unknown field":    append([]byte(`{"unknown":true,`), valid[1:]...),
		"duplicate field":  append([]byte(`{"schema":"contractor.findings.collection.v2",`), valid[1:]...),
		"case alias":       bytes.Replace(valid, []byte(`"schema"`), []byte(`"Schema"`), 1),
		"noncanonical":     append([]byte("\n"), valid...),
		"null origin":      bytes.Replace(valid, []byte(`"entries":[{`), []byte(`"entries":[{"audit_origin":null,`), 1),
		"trailing payload": append(append([]byte{}, valid...), []byte(` {}`)...),
		"invalid UTF8":     append([]byte{0xff}, valid...),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeFindingCollection(input); testDomainErrorCode(err) != CodeInvalid {
				t.Fatalf("error = %v", err)
			}
		})
	}
	if _, err := DecodeFindingCollection(bytes.Repeat([]byte(" "), MaximumCollectionJSONBytes+1)); testDomainErrorCode(err) != CodeLimitExceeded {
		t.Fatalf("metadata byte limit = %v", err)
	}
}

func TestSharedFindingCollectionRejectsMissingAndForgedContents(t *testing.T) {
	for _, mode := range []string{"missing", "extra", "changed", "wrong evidence ID", "invalid proposal"} {
		t.Run(mode, func(t *testing.T) {
			value, contents, _, _ := collectionFixture(t)
			id := value.Entries[0].ProposalDocumentID
			switch mode {
			case "missing":
				delete(contents, id)
			case "extra":
				contents["doc-extra"] = []byte("extra")
			case "changed":
				contents[id] = []byte("changed")
			case "wrong evidence ID":
				value.Entries[0].Evidence[0].EvidenceID = "evidence-foreign"
			case "invalid proposal":
				for i, doc := range value.Documents {
					if doc.ID != id {
						continue
					}
					body := []byte(`{"schema":"not-a-proposal"}`)
					doc.Digest, doc.SizeBytes = DigestBytes(body), int64(len(body))
					var err error
					doc.ID, err = FindingCollectionDocumentID(doc)
					if err != nil {
						t.Fatal(err)
					}
					value.Documents[i] = doc
					value.Entries[0].ProposalDocumentID = doc.ID
					delete(contents, id)
					contents[doc.ID] = body
				}
				sort.Slice(value.Documents, func(i, j int) bool { return value.Documents[i].ID < value.Documents[j].ID })
			}
			if _, err := BuildFindingCollectionPackage(value, contents); err == nil {
				t.Fatal("incomplete or contradictory collection was accepted")
			}
		})
	}
}

func TestSharedFindingCollectionDecoderChecksArchiveAgainstMetadata(t *testing.T) {
	for _, mode := range []string{"package ID", "kind", "collection path", "missing document", "extra document", "wrong bytes", "document path", "document MIME"} {
		t.Run(mode, func(t *testing.T) {
			value, contents, packageID, _ := collectionFixture(t)
			metadata, err := EncodeFindingCollection(value)
			if err != nil {
				t.Fatal(err)
			}
			inputs := []PackageInput{{ID: FindingCollectionMemberID, Path: "collection.json", MediaType: JSONMediaType, Data: metadata}}
			for _, doc := range value.Documents {
				inputs = append(inputs, PackageInput{ID: doc.ID, Path: "documents/" + doc.ID, MediaType: doc.MediaType, Data: contents[doc.ID]})
			}
			kind := PackageKindFindingCollection
			switch mode {
			case "package ID":
				packageID = "collection-forged"
			case "kind":
				kind = PackageKindEvidence
			case "collection path":
				inputs[0].Path = "other.json"
			case "missing document":
				inputs = inputs[:len(inputs)-1]
			case "extra document":
				inputs = append(inputs, PackageInput{ID: "extra", Path: "extra.txt", MediaType: "text/plain", Data: []byte("extra")})
			case "wrong bytes":
				inputs[1].Data = []byte("changed")
			case "document path":
				inputs[1].Path = "unexpected.txt"
			case "document MIME":
				inputs[1].MediaType = "application/octet-stream"
			}
			payload, _, err := BuildPackage(packageID, kind, "", inputs)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := DecodeFindingCollectionPackage(payload); err == nil {
				t.Fatal("valid ZIP with inconsistent collection was accepted")
			}
		})
	}
}

func TestSharedFindingCollectionTaskOriginBounds(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name   string `json:"name"`
			Origin string `json:"origin"`
			Valid  bool   `json:"valid"`
		} `json:"audit_origin_cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "findings", "finding-collection-v2.fixture.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) < 10 {
		t.Fatal("missing shared origin boundary cases")
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			value, _, _, _ := collectionFixture(t)
			valid, err := EncodeFindingCollection(value)
			if err != nil {
				t.Fatal(err)
			}
			original, _ := json.Marshal(value.Entries[1].AuditOrigin)
			// Use the same canonical key ordering as the emitted collection, retaining
			// the case's authored numeric spelling for strict read rejection.
			var origin map[string]any
			if err := json.Unmarshal(original, &origin); err != nil {
				t.Fatal(err)
			}
			canonical, _ := json.Marshal(origin)
			input := bytes.Replace(valid, canonical, []byte(test.Origin), 1)
			if bytes.Equal(input, valid) && test.Origin != string(canonical) {
				t.Fatal("origin replacement did not exercise decoder")
			}
			if _, err := DecodeFindingCollection(input); (err == nil) != test.Valid {
				t.Fatalf("valid=%v want %v: %v", err == nil, test.Valid, err)
			}
		})
	}
}

func TestSharedFindingCollectionProjectSourceKinds(t *testing.T) {
	var fixture struct {
		SourceKinds []struct {
			Kind  string `json:"kind"`
			Valid bool   `json:"valid"`
		} `json:"source_kinds"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "findings", "finding-collection-v2.fixture.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.SourceKinds) == 0 {
		t.Fatal("missing source-kind fixtures")
	}
	for _, test := range fixture.SourceKinds {
		t.Run(test.Kind, func(t *testing.T) {
			value, _, _, _ := collectionFixture(t)
			value.Sources = []FindingCollectionSource{{Kind: test.Kind, ID: "selected-source"}}
			for i := range value.Entries {
				value.Entries[i].SelectedSources = value.Sources
			}
			_, err := EncodeFindingCollection(value)
			if (err == nil) != test.Valid {
				t.Fatalf("source %q: %v", test.Kind, err)
			}
		})
	}
}

func TestSharedFindingCollectionExactHypothesisContext(t *testing.T) {
	var fixture struct {
		Body string `json:"hypothesis_context_body"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "findings", "finding-collection-v2.fixture.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	value, contents, _, _ := collectionFixture(t)
	revision := "hypothesis-v7"
	document := FindingCollectionDocument{Scope: FindingCollectionSource{Kind: "project", ID: "project-context"}, Ref: contracts.ArtifactRef{Namespace: "finding-materials", Name: "hypothesis-context", Revision: &revision}, Digest: DigestBytes([]byte(fixture.Body)), MediaType: JSONMediaType, SizeBytes: int64(len(fixture.Body))}
	var err error
	document.ID, err = FindingCollectionDocumentID(document)
	if err != nil {
		t.Fatal(err)
	}
	value.Documents = append(value.Documents, document)
	sort.Slice(value.Documents, func(i, j int) bool { return value.Documents[i].ID < value.Documents[j].ID })
	contents[document.ID] = []byte(fixture.Body)
	value.Entries[1].AuditOrigin.Hypothesis = &auditcontract.HypothesisRef{ID: "selected-hypothesis", Revision: 7}
	value.Entries[1].AuditOrigin.HypothesisSnapshotDocumentID = document.ID
	if _, err := BuildFindingCollectionPackage(value, contents); err != nil {
		t.Fatal(err)
	}
	value.Entries[1].AuditOrigin.Hypothesis.Revision = 8
	if _, err := BuildFindingCollectionPackage(value, contents); err == nil {
		t.Fatal("context for a different hypothesis revision was accepted")
	}
}
