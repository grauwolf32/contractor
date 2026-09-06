package auditdomain

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// Frozen values were produced independently with Python hashlib/json/zipfile.
// The ASCII metadata keys and integral numbers need no extra JCS normalization.
func collectionFixture(t *testing.T) (FindingCollection, map[string][]byte, string, string) {
	t.Helper()
	raw, err := os.ReadFile("testdata/finding-collection-v1.fixture.json")
	if err != nil {
		t.Fatal(err)
	}
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

func TestFindingCollectionConformanceAndConsumerTargets(t *testing.T) {
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
	targets, err := FindingCollectionTargets(payload)
	if err != nil {
		t.Fatal(err)
	}
	seen := make(map[string]bool)
	for _, document := range decoded.Documents {
		ref, ok := targets[document.ID]
		if !ok || ref.Validate() != nil || ref.Revision != nil || ref.Namespace != "findings-"+strings.TrimPrefix(packageDigest, "sha256:") || ref.Name != document.ID {
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
	if first.AuditOrigin != nil || len(first.Reviews) != 0 || second.Reviews[0].DecisionID != "" {
		t.Fatal("invented an Audit origin or decision")
	}
	value.SnapshotAt = "2026-09-06T09:00:01Z"
	changed, err := BuildFindingCollectionPackage(value, contents)
	if err != nil {
		t.Fatal(err)
	}
	changedTargets, err := FindingCollectionTargets(changed)
	if err != nil || changedTargets[first.ProposalDocumentID].Namespace == targets[first.ProposalDocumentID].Namespace {
		t.Fatalf("different snapshot reused consumer namespace: %v", err)
	}
	if _, err := DecodeCheckResultPackage(payload); ErrorCode(err) != CodePackageInvalid {
		t.Fatalf("collection was accepted as check results: %v", err)
	}
}

func TestFindingCollectionEmptySelectionAndOptionalReviews(t *testing.T) {
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
	for _, state := range []string{"proposed", "confirmed", "rejected", "duplicate", "needs-evidence"} {
		t.Run(state, func(t *testing.T) {
			value, contents, _, _ := collectionFixture(t)
			value.Entries[1].Reviews[0].State = state
			if _, err := BuildFindingCollectionPackage(value, contents); err != nil {
				t.Fatalf("captured state requires an invented decision: %v", err)
			}
		})
	}
}

func TestFindingCollectionSharesEvidenceWithoutMergingReceipts(t *testing.T) {
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

func TestFindingCollectionAuditHoldPreservesMembershipWithoutReview(t *testing.T) {
	value, contents, _, _ := collectionFixture(t)
	value.Sources = []FindingCollectionSource{{Kind: "audit", ID: "audit-a"}}
	for i := range value.Entries {
		value.Entries[i].AuditOrigin = nil
		value.Entries[i].Reviews = []FindingCollectionReview{}
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
	for _, holds := range [][]string{{"audit-a", "audit-a"}, {"audit-z", "audit-a"}, {"../audit"}, {"audit-foreign"}} {
		value.Entries[0].AuditHolds = holds
		if _, err := EncodeFindingCollection(value); err == nil {
			t.Fatalf("invalid hold membership accepted: %v", holds)
		}
	}
}

func TestFindingCollectionRejectsInvalidMetadata(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*FindingCollection)
		code   string
	}{
		{"schema", func(v *FindingCollection) { v.Schema = "contractor.findings.collection.v2" }, CodeSchemaUnsupported},
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
		{"missing reviews", func(v *FindingCollection) { v.Entries[0].Reviews = nil }, CodeLimitExceeded},
		{"duplicate review", func(v *FindingCollection) {
			v.Entries[1].Reviews = append(v.Entries[1].Reviews, v.Entries[1].Reviews[0])
		}, CodeInvalid},
		{"review state", func(v *FindingCollection) { v.Entries[1].Reviews[0].State = "approved" }, CodeInvalid},
		{"zero revision", func(v *FindingCollection) { v.Entries[1].Reviews[0].Revision = 0 }, CodeInvalid},
		{"unsafe integer", func(v *FindingCollection) { v.Entries[1].Reviews[0].Revision = 1 << 53 }, CodeInvalid},
		{"invalid decision", func(v *FindingCollection) { v.Entries[1].Reviews[0].DecisionID = "../decision" }, CodeInvalid},
		{"entry limit", func(v *FindingCollection) { v.Entries = make([]FindingCollectionEntry, MaximumCollectionEntries+1) }, CodeLimitExceeded},
		{"document limit", func(v *FindingCollection) {
			v.Documents = make([]FindingCollectionDocument, MaximumCollectionDocuments+1)
		}, CodeLimitExceeded},
		{"source limit", func(v *FindingCollection) { v.Sources = make([]FindingCollectionSource, MaximumCollectionSources+1) }, CodeLimitExceeded},
		{"review limit", func(v *FindingCollection) {
			v.Entries[0].Reviews = make([]FindingCollectionReview, MaximumCollectionReviews+1)
		}, CodeLimitExceeded},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			value, _, _, _ := collectionFixture(t)
			tc.mutate(&value)
			_, err := EncodeFindingCollection(value)
			if ErrorCode(err) != tc.code {
				t.Fatalf("error = %v, want %s", err, tc.code)
			}
		})
	}
}

func TestFindingCollectionRejectsAmbiguousJSON(t *testing.T) {
	value, _, _, _ := collectionFixture(t)
	valid, err := EncodeFindingCollection(value)
	if err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string][]byte{
		"unknown field":    append([]byte(`{"unknown":true,`), valid[1:]...),
		"duplicate field":  append([]byte(`{"schema":"contractor.findings.collection.v1",`), valid[1:]...),
		"case alias":       bytes.Replace(valid, []byte(`"schema"`), []byte(`"Schema"`), 1),
		"noncanonical":     append([]byte("\n"), valid...),
		"null origin":      bytes.Replace(valid, []byte(`"entries":[{`), []byte(`"entries":[{"audit_origin":null,`), 1),
		"trailing payload": append(append([]byte{}, valid...), []byte(` {}`)...),
		"invalid UTF8":     append([]byte{0xff}, valid...),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeFindingCollection(input); ErrorCode(err) != CodeInvalid {
				t.Fatalf("error = %v", err)
			}
		})
	}
	if _, err := DecodeFindingCollection(bytes.Repeat([]byte(" "), MaximumCollectionJSONBytes+1)); ErrorCode(err) != CodeLimitExceeded {
		t.Fatalf("metadata byte limit = %v", err)
	}
}

func TestFindingCollectionRejectsMissingAndForgedContents(t *testing.T) {
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
					doc.Digest, doc.SizeBytes = digestBytes(body), int64(len(body))
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

func TestFindingCollectionDecoderChecksArchiveAgainstMetadata(t *testing.T) {
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
			if targets, err := FindingCollectionTargets(payload); err == nil || targets != nil {
				t.Fatal("invalid collection exposed consumer write targets")
			}
		})
	}
}
