package findinganalysis

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func testAnalysis() Document {
	return Document{Schema: Schema, ClientKey: "candidate", Sources: []Source{}, Locations: []Location{}, Flows: []Flow{}, HTTPExchanges: []HTTPExchange{}}
}
func testEvidence(t *testing.T, name, media string, data []byte) auditdomain.FindingCollectionDocument {
	t.Helper()
	revision := "1"
	doc := auditdomain.FindingCollectionDocument{Scope: auditdomain.FindingCollectionSource{Kind: "run", ID: "run"}, Ref: contracts.ArtifactRef{Namespace: "evidence", Name: name, Revision: &revision}, Digest: contentdigest.Bytes(data), MediaType: media, SizeBytes: int64(len(data))}
	id, err := auditdomain.FindingCollectionDocumentID(doc)
	if err != nil {
		t.Fatal(err)
	}
	doc.ID = id
	return doc
}
func resolvedFixture(t *testing.T, value Document, extras map[string][]byte, docs ...auditdomain.FindingCollectionDocument) (*Resolved, error) {
	t.Helper()
	data, err := strictjson.Canonical(value)
	if err != nil {
		t.Fatal(err)
	}
	doc := testEvidence(t, "analysis", MediaType, data)
	contents := map[string][]byte{doc.ID: data}
	for k, v := range extras {
		contents[k] = v
	}
	return new(Resolver).Resolve(t.Context(), "candidate", append(docs, doc), contents, nil)
}
func TestStrictCanonicalAnalysis(t *testing.T) {
	data, err := strictjson.Canonical(testAnalysis())
	if err != nil {
		t.Fatal(err)
	}
	if _, err = decode(data); err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{
		string(data) + "\n", strings.Replace(string(data), `"schema":`, `"Schema":`, 1),
		strings.Replace(string(data), `"sources":[]`, `"sources":null`, 1),
		strings.Replace(string(data), `,"sources":[]`, "", 1),
		strings.Replace(string(data), `"sources":[]`, `"sources":[],"sources":[]`, 1),
		strings.Replace(string(data), `"sources":[]`, `"sources":[],"rule":null`, 1),
		strings.Replace(string(data), `"client_key":"candidate"`, `"client_key":"\ud800"`, 1),
	} {
		if _, err := decode([]byte(input)); err == nil {
			t.Errorf("accepted malformed document %s", input)
		}
	}
}
func TestResolveExactFileAndUnicodeRegion(t *testing.T) {
	source := []byte("\tα😀x\r\nend\n")
	doc := testEvidence(t, "source", "text/plain", source)
	value := testAnalysis()
	value.Sources = []Source{{ID: "source", Kind: "file", ArtifactRef: doc.Ref}}
	end := int64(5)
	start := int64(2)
	value.Locations = []Location{{ID: "sink", SourceID: "source", Path: "src/café #%.py", FileDigest: doc.Digest, Region: &Region{StartLine: 1, StartColumn: &start, EndColumn: &end}}}
	value.PrimaryLocationID = "sink"
	result, err := resolvedFixture(t, value, map[string][]byte{doc.ID: source}, doc)
	if err != nil || len(result.Sources) != 1 || len(result.Sources[0].RootKey) != 64 {
		t.Fatalf("resolve: %+v, %v", result, err)
	}
	end = 6
	if _, err := resolvedFixture(t, value, map[string][]byte{doc.ID: source}, doc); err == nil {
		t.Fatal("accepted out-of-file Unicode column")
	}
	end = 5
	value.Locations[0].FileDigest = contentdigest.Bytes([]byte("different"))
	if _, err := resolvedFixture(t, value, map[string][]byte{doc.ID: source}, doc); err == nil {
		t.Fatal("accepted wrong exact file digest")
	}
}
func TestRegionLineEndingsAndEmptyFiles(t *testing.T) {
	one, two, three := int64(1), int64(2), int64(3)
	for _, tc := range []struct {
		data   string
		region *Region
		valid  bool
	}{
		{"", nil, true}, {"", &Region{StartLine: 1}, false},
		{"α\r\n\tx", &Region{StartLine: 2, StartColumn: &two, EndColumn: &three}, true},
		{"α\r\n\tx", &Region{StartLine: 3}, false},
		{"x", &Region{StartLine: 1, StartColumn: &two, EndColumn: &two}, true},
		{"x", &Region{StartLine: 1, StartColumn: &three}, false},
		{"x\n", &Region{StartLine: 2, StartColumn: &one}, true},
	} {
		if got := validRegion([]byte(tc.data), tc.region); got != tc.valid {
			t.Errorf("region %+v on %q: %v", tc.region, tc.data, got)
		}
	}
}
func TestExplicitDependencyClosureAndNoAnalysis(t *testing.T) {
	if v, err := new(Resolver).Resolve(t.Context(), "candidate", nil, nil, nil); err != nil || v != nil {
		t.Fatalf("no analysis: %v %v", v, err)
	}
	value := testAnalysis()
	revision := "1"
	value.Sources = []Source{{ID: "missing", Kind: "file", ArtifactRef: contracts.ArtifactRef{Namespace: "evidence", Name: "missing", Revision: &revision}}}
	if _, err := resolvedFixture(t, value, nil); err == nil {
		t.Fatal("fetched unselected evidence")
	}
	value = testAnalysis()
	value.ClientKey = "other"
	if _, err := resolvedFixture(t, value, nil); err == nil {
		t.Fatal("accepted another proposal client key")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := new(Resolver).Resolve(ctx, "candidate", nil, nil, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
}
func TestArchiveRejectsTraversalDuplicateAndSymlink(t *testing.T) {
	for _, tc := range []struct {
		name    string
		paths   []string
		symlink bool
		valid   bool
	}{
		{"exact", []string{"src/a.py"}, false, true},
		{"traversal", []string{"../src/a.py"}, false, false},
		{"absolute", []string{"/src/a.py"}, false, false},
		{"duplicate", []string{"src/a.py", "src/a.py"}, false, false},
		{"file as directory", []string{"src/a.py", "src/a.py/child"}, false, false},
		{"file directory alias", []string{"src/a.py", "src/a.py/"}, false, false},
		{"symlink", []string{"src/a.py"}, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buffer bytes.Buffer
			writer := zip.NewWriter(&buffer)
			for _, path := range tc.paths {
				h := &zip.FileHeader{Name: path, Method: zip.Deflate}
				h.SetMode(0644)
				if tc.symlink {
					h.SetMode(os.ModeSymlink | 0777)
				}
				w, err := writer.CreateHeader(h)
				if err != nil {
					t.Fatal(err)
				}
				if !strings.HasSuffix(path, "/") {
					if _, err = w.Write([]byte("file")); err != nil {
						t.Fatal(err)
					}
				}
			}
			if err := writer.Close(); err != nil {
				t.Fatal(err)
			}
			expanded := uint64(0)
			index, err := indexArchive(t.Context(), buffer.Bytes(), &expanded)
			if err == nil {
				_, err = index.read(t.Context(), map[string]bool{"src/a.py": true})
			}
			if (err == nil) != tc.valid {
				t.Fatalf("archive validation: %v", err)
			}
		})
	}
}
func TestFlowConnectivityAndUntrustedToolClaims(t *testing.T) {
	source := []byte("source\nsink")
	doc := testEvidence(t, "source", "text/plain", source)
	value := testAnalysis()
	value.Sources = []Source{{ID: "source", Kind: "file", ArtifactRef: doc.Ref}}
	value.Locations = []Location{{ID: "source", SourceID: "source", Path: "a", FileDigest: doc.Digest, Region: &Region{StartLine: 1}}, {ID: "sink", SourceID: "source", Path: "a", FileDigest: doc.Digest, Region: &Region{StartLine: 2}}}
	value.Flows = []Flow{{ID: "flow", Kind: "data-flow", Description: "Producer assertion", Completeness: "complete", Steps: []Step{
		{ID: "a", Roles: []string{"source"}, Message: "Input", EvidenceRefs: []contracts.ArtifactRef{doc.Ref}, LocationID: "source"},
		{ID: "b", Roles: []string{"sink"}, Message: "Sink", EvidenceRefs: []contracts.ArtifactRef{doc.Ref}, LocationID: "sink"},
	}, Edges: []Edge{{From: "a", To: "b", Relation: "assignment", Basis: "producer-asserted", EvidenceRefs: []contracts.ArtifactRef{doc.Ref}}}, Gaps: []Gap{}, EvidenceRefs: []contracts.ArtifactRef{doc.Ref}}}
	if _, err := resolvedFixture(t, value, map[string][]byte{doc.ID: source}, doc); err != nil {
		t.Fatal(err)
	}
	value.Flows[0].Edges[0].Basis = "tool-reported"
	if _, err := resolvedFixture(t, value, map[string][]byte{doc.ID: source}, doc); err == nil {
		t.Fatal("copied evidence minted a trusted tool relation")
	}
	value.Flows[0].Edges = nil
	value.Flows[0].Gaps = []Gap{{AfterStepID: "a", BeforeStepID: "b", Reason: "Unresolved propagation"}}
	value.Flows[0].Edges = []Edge{}
	if _, err := resolvedFixture(t, value, map[string][]byte{doc.ID: source}, doc); err == nil {
		t.Fatal("complete path bridged a gap")
	}
	value.Flows[0].Completeness = "partial"
	if _, err := resolvedFixture(t, value, map[string][]byte{doc.ID: source}, doc); err != nil {
		t.Fatal(err)
	}
}
func TestHTTPProjectionUsesActualCaptureAndOmitsPrivateFields(t *testing.T) {
	payload := []byte(`{"schema":"contractor.http.exchange.v1","capture":{"source":"http","request_id":"1","request_tag":"tag","attempts":[{"request":{"method":"POST","url":"https://example.test/check?token=private#secret","message":{"kind":"structured","headers":[{"name":"Cookie","value":"private-cookie"}],"body_base64":"cHJpdmF0ZQ=="}},"response":{"status":201,"message":{"kind":"structured","headers":[{"name":"Content-Type","value":"application/json"}],"body_base64":"cHJpdmF0ZQ=="},"complete":true}}]},"finding":{"proposal_id":"proposal","receipt_id":"receipt"}}`)
	doc := testEvidence(t, "http", auditdomain.HTTPExchangeMediaType, payload)
	exchange := HTTPExchange{ID: "http", EvidenceRef: doc.Ref, Decoder: "contractor-http-exchange/v1"}
	value := testAnalysis()
	value.HTTPExchanges = []HTTPExchange{exchange}
	value.PrimaryHTTPExchangeID = "http"
	result, err := resolvedFixture(t, value, map[string][]byte{doc.ID: payload}, doc)
	if err != nil {
		t.Fatal(err)
	}
	projection := result.HTTP[0]
	if projection.URI != "https://example.test/check" || projection.Method != "POST" || projection.Status == nil || *projection.Status != 201 {
		t.Fatalf("projection: %+v", projection)
	}
	encoded, err := strictjson.Canonical(result.HTTP)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte("private")) || bytes.Contains(encoded, []byte("secret")) {
		t.Fatalf("leaked private HTTP values: %s", encoded)
	}
	// Reusing an earlier saved capture is valid; its original association is
	// provenance, and a new analysis need not invent another target request.
	trusted := doc
	trusted.Ref.Namespace = "finding-proposals"
	trusted.Ref.Name = "http.exchange.proposal"
	projection, err = resolveHTTP(exchange, trusted, payload)
	if err != nil || projection.Origin != "runtime-capture" {
		t.Fatalf("trusted saved capture: %+v %v", projection, err)
	}
	invalid := bytes.Replace(payload, []byte(`"receipt_id":"receipt"`), []byte(`"receipt_id":""`), 1)
	if _, err := resolveHTTP(exchange, doc, invalid); err == nil {
		t.Fatal("accepted invalid capture provenance")
	}
}
func TestMaximumLocationCardinality(t *testing.T) {
	value := testAnalysis()
	data := []byte("file")
	doc := testEvidence(t, "source", "text/plain", data)
	value.Sources = []Source{{ID: "source", Kind: "file", ArtifactRef: doc.Ref}}
	for i := range 256 {
		value.Locations = append(value.Locations, Location{ID: fmt.Sprintf("location-%d", i), SourceID: "source", Path: "a", FileDigest: doc.Digest})
	}
	if _, err := resolvedFixture(t, value, map[string][]byte{doc.ID: data}, doc); err != nil {
		t.Fatal(err)
	}
	value.Locations = append(value.Locations, Location{ID: "overflow", SourceID: "source", Path: "a", FileDigest: doc.Digest})
	if _, err := resolvedFixture(t, value, map[string][]byte{doc.ID: data}, doc); err == nil {
		t.Fatal("accepted location overflow")
	}
}

func TestResolverExpandsSharedArchiveOnceAcross256Receipts(t *testing.T) {
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	member, err := writer.Create("src/a.py")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = member.Write([]byte("source\nsink")); err != nil {
		t.Fatal(err)
	}
	if err = writer.Close(); err != nil {
		t.Fatal(err)
	}
	source := testEvidence(t, "source", "application/zip", buffer.Bytes())
	value := testAnalysis()
	value.Sources = []Source{{ID: "source", Kind: "archive", ArtifactRef: source.Ref}}
	value.Locations = []Location{{ID: "sink", SourceID: "source", Path: "src/a.py", FileDigest: contentdigest.Bytes([]byte("source\nsink")), Region: &Region{StartLine: 2}}}
	data, err := strictjson.Canonical(value)
	if err != nil {
		t.Fatal(err)
	}
	analysis := testEvidence(t, "analysis", MediaType, data)
	contents := map[string][]byte{source.ID: buffer.Bytes(), analysis.ID: data}
	var resolver Resolver
	for range auditdomain.MaximumCollectionEntries {
		if _, err := resolver.Resolve(t.Context(), "candidate", []auditdomain.FindingCollectionDocument{source, analysis}, contents, nil); err != nil {
			t.Fatal(err)
		}
	}
	if resolver.expanded != uint64(len("source\nsink")) || len(resolver.archives) != 1 || len(resolver.archives[source.ID].files) != 1 {
		t.Fatalf("repeated source expansion: %+v", resolver)
	}
}
