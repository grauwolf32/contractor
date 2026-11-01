package sarif

import (
	"archive/zip"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findinganalysis"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func exactDocument(t *testing.T, name, media string, data []byte) auditdomain.FindingCollectionDocument {
	t.Helper()
	revision := "r1"
	doc := auditdomain.FindingCollectionDocument{Scope: auditdomain.FindingCollectionSource{Kind: "run", ID: "run"}, Ref: contracts.ArtifactRef{Namespace: "evidence", Name: name, Revision: &revision}, Digest: contentdigest.Bytes(data), MediaType: media, SizeBytes: int64(len(data))}
	var err error
	doc.ID, err = auditdomain.FindingCollectionDocumentID(doc)
	if err != nil {
		t.Fatal(err)
	}
	return doc
}

func attachAnalysis(t *testing.T, s *findingexport.Snapshot, analysis findinganalysis.Document, docs []auditdomain.FindingCollectionDocument, contents map[string][]byte) {
	t.Helper()
	receipt := &s.Receipts[0]
	analysis.ClientKey = receipt.Proposal.ClientKey
	data, err := strictjson.Canonical(analysis)
	if err != nil {
		t.Fatal(err)
	}
	doc := exactDocument(t, "analysis", findinganalysis.MediaType, data)
	docs = append(docs, doc)
	contents[doc.ID] = data
	receipt.Evidence = []findingexport.Evidence{}
	receipt.Proposal.EvidenceIDs = []string{}
	for i, d := range docs {
		receipt.Evidence = append(receipt.Evidence, findingexport.Evidence{Original: d, Retained: d})
		receipt.Proposal.EvidenceIDs = append(receipt.Proposal.EvidenceIDs, "evidence-"+string(rune('a'+i)))
	}
	receipt.Analysis, err = new(findinganalysis.Resolver).Resolve(t.Context(), analysis.ClientKey, docs, contents, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range receipt.Analysis.HTTP {
		if receipt.Analysis.HTTP[i].ID == analysis.PrimaryHTTPExchangeID {
			receipt.HTTP = &receipt.Analysis.HTTP[i]
		}
	}
	// Reflect the selected proposal's exact bytes after adding evidence IDs.
	data, err = auditdomain.EncodeFindingProposal(receipt.Proposal)
	if err != nil {
		t.Fatal(err)
	}
	doc = receipt.ProposalArtifact.Original
	doc.Digest, doc.SizeBytes = contentdigest.Bytes(data), int64(len(data))
	doc.ID, err = auditdomain.FindingCollectionDocumentID(doc)
	if err != nil {
		t.Fatal(err)
	}
	receipt.ProposalArtifact = findingexport.Evidence{Original: doc, Retained: doc}
}

func codeSnapshot(t *testing.T, partial bool) findingexport.Snapshot {
	t.Helper()
	s := noAnalysisSnapshot(t, 1)
	source := []byte("\tα😀x\r\ncall(value)\nvalidate(value)\nsink(value)\n")
	path := "src/café #%.py"
	var zipped bytes.Buffer
	writer := zip.NewWriter(&zipped)
	header := &zip.FileHeader{Name: path, Method: zip.Store}
	header.SetMode(0644)
	member, err := writer.CreateHeader(header)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = member.Write(source); err != nil {
		t.Fatal(err)
	}
	if err = writer.Close(); err != nil {
		t.Fatal(err)
	}
	doc := exactDocument(t, "repository", "application/zip", zipped.Bytes())
	start, end := int64(2), int64(5)
	analysis := findinganalysis.Document{Schema: findinganalysis.Schema, Sources: []findinganalysis.Source{{ID: "repo", Kind: "archive", ArtifactRef: doc.Ref}}, Locations: []findinganalysis.Location{}, Flows: []findinganalysis.Flow{}, HTTPExchanges: []findinganalysis.HTTPExchange{}, Rule: &findinganalysis.Rule{ID: "python/unsafe-flow", Version: "1.0.0", Name: "Unsafe flow", Description: "Inspect explicit propagation", HelpURI: "https://example.test/rules/unsafe-flow"}, PrimaryLocationID: "d"}
	for i, id := range []string{"a", "b", "c", "d"} {
		location := findinganalysis.Location{ID: id, SourceID: "repo", Path: path, FileDigest: contentdigest.Bytes(source), Region: &findinganalysis.Region{StartLine: int64(i + 1)}}
		if i == 0 {
			location.Region.StartColumn, location.Region.EndColumn = &start, &end
		}
		if i == 3 {
			location.Symbol = &findinganalysis.Symbol{Name: "sink", QualifiedName: "app.sink", Language: "python", Kind: "function", Signature: "sink(value)"}
		}
		analysis.Locations = append(analysis.Locations, location)
	}
	flow := findinganalysis.Flow{ID: "flow", Kind: "data-flow", Description: "Input reaches sink", Completeness: "complete", Steps: []findinganalysis.Step{}, Edges: []findinganalysis.Edge{}, Gaps: []findinganalysis.Gap{}, EvidenceRefs: []contracts.ArtifactRef{doc.Ref}}
	for i, id := range []string{"a", "b", "c", "d"} {
		role := "propagation"
		if i == 0 {
			role = "source"
		}
		if i == 2 {
			role = "validation"
		}
		if i == 3 {
			role = "sink"
		}
		flow.Steps = append(flow.Steps, findinganalysis.Step{ID: id, LocationID: id, Roles: []string{role}, Message: "Step " + id, EvidenceRefs: []contracts.ArtifactRef{doc.Ref}, Value: "value", ValueState: "unknown"})
		if i > 0 {
			flow.Edges = append(flow.Edges, findinganalysis.Edge{From: flow.Steps[i-1].ID, To: id, Relation: "assignment", Basis: "producer-asserted", EvidenceRefs: []contracts.ArtifactRef{doc.Ref}})
		}
	}
	flow.Steps[2].ValueState = "validated"
	if partial {
		flow.Completeness = "partial"
		flow.Edges = append(flow.Edges[:1], flow.Edges[2:]...)
		flow.Gaps = []findinganalysis.Gap{{AfterStepID: "b", BeforeStepID: "c", Reason: "Unresolved propagation"}}
	}
	analysis.Flows = append(analysis.Flows, flow)
	s.Receipts[0].Proposal.StandardRefs = []auditdomain.StandardReference{{Scheme: "CWE", Version: "4.20", RequirementID: "CWE-89"}, {Scheme: "asvs", Version: "4.0.3", RequirementID: "5.3.4"}, {Scheme: "owasp-asvs", Version: "5.0.0", RequirementID: "1.2.3"}, {Scheme: "custom", Version: "1", RequirementID: "unmapped"}}
	s.Receipts[0].Proposal.SeveritySuggestion = "low"
	attachAnalysis(t, &s, analysis, []auditdomain.FindingCollectionDocument{doc}, map[string][]byte{doc.ID: zipped.Bytes()})
	s.Receipts[0].Evidence[0].Git = &findingexport.GitSource{RepositoryURI: "https://example.test/repo.git", Commit: strings.Repeat("a", 40)}
	s.Results[0].Review = findingexport.Review{State: findingStateForTest("confirmed"), Verdict: "true_positive", Severity: "high", Rationale: "Confirmed in this Audit"}
	algorithm, value := "contractor/producer-issue/v1", strings.Repeat("b", 64)
	group, err := auditdomain.FindingGroupKey("project", algorithm, value)
	if err != nil {
		t.Fatal(err)
	}
	s.Results[0].PrimaryKey = &findingexport.PrimaryKey{ProjectID: "project", Algorithm: algorithm, Value: value, GroupKey: group, Policy: auditdomain.FindingCorrelationPolicy}
	return s
}

func rendered(t *testing.T, s findingexport.Snapshot) ([]byte, object, findingexport.Diagnostics) {
	t.Helper()
	data, diagnostics, err := Render(s)
	if err != nil {
		t.Fatal(err)
	}
	doc, err := validateSchema(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateSemantics(doc, s); err != nil {
		t.Fatalf("output semantics: %v", err)
	}
	return data, doc, diagnostics
}
func runResult(document object) (object, object) {
	run := obj(array(document["runs"])[0])
	return run, obj(array(run["results"])[0])
}
func hasDiagnostic(d findingexport.Diagnostics, code string) bool {
	for _, item := range d.Items {
		if item.Code == code {
			return true
		}
	}
	return false
}
func golden(t *testing.T, name string, data []byte) {
	t.Helper()
	path := filepath.Join("testdata", name+".sarif")
	if os.Getenv("UPDATE_SARIF_GOLDENS") == "1" {
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0644); err != nil {
			t.Fatal(err)
		}
	}
	expected, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(expected, data) {
		t.Fatalf("golden differs: %s", path)
	}
}

func TestRenderCodeGoldensAndNativeClaims(t *testing.T) {
	for _, partial := range []bool{false, true} {
		name := "complete-code"
		count := 1
		if partial {
			name, count = "partial-code", 2
		}
		t.Run(name, func(t *testing.T) {
			snapshot := codeSnapshot(t, partial)
			data, document, diagnostics := rendered(t, snapshot)
			golden(t, name, data)
			run, result := runResult(document)
			if _, exists := obj(obj(run["tool"])["driver"])["supportedTaxonomies"]; exists {
				t.Fatal("result classifications advertised as shared rule taxonomy support")
			}
			if len(array(result["codeFlows"])) != count || len(array(run["artifacts"])) != 1 || len(array(run["versionControlProvenance"])) != 1 || len(array(result["taxa"])) != 3 || result["level"] != "error" {
				t.Fatal("lost code mapping")
			}
			location := obj(array(result["locations"])[0])
			if contractor(location)["native_location_id"] != "d" || len(array(location["logicalLocations"])) != 1 {
				t.Fatal("lost explicit primary/symbol")
			}
			uri := str(obj(obj(array(run["artifacts"])[0])["location"])["uri"])
			if uri != "src/caf%C3%A9%20%23%25.py" {
				t.Fatalf("URI: %s", uri)
			}
			if !hasDiagnostic(diagnostics, "classification_unmapped") || hasDiagnostic(diagnostics, "flow_partial") != partial {
				t.Fatalf("diagnostics: %+v", diagnostics)
			}
			if !bytes.Contains(data, []byte("Producer claim; this does not establish safety.")) {
				t.Fatal("unqualified validation claim")
			}
			again, _, err := Render(snapshot)
			if err != nil || !bytes.Equal(data, again) {
				t.Fatal("unstable rendering")
			}
		})
	}
}

func TestRenderNoInferredPrimaryAndCallPath(t *testing.T) {
	snapshot := codeSnapshot(t, false)
	snapshot.Receipts[0].Analysis.Analysis.PrimaryLocationID = ""
	snapshot.Receipts[0].Analysis.Analysis.Flows[0].Kind = "call-path"
	_, document, diagnostics := rendered(t, snapshot)
	_, result := runResult(document)
	if len(array(result["locations"])) != 0 || len(array(result["relatedLocations"])) != 4 || !hasDiagnostic(diagnostics, "primary_location_missing") || contractor(obj(array(result["codeFlows"])[0]))["kind"] != "call-path" {
		t.Fatal("inferred primary or lost call-path")
	}
}

func TestRenderEmptyFileAndIndependentRoots(t *testing.T) {
	s := noAnalysisSnapshot(t, 1)
	first, second := []byte{}, []byte("different\n")
	a, b := exactDocument(t, "a", "text/plain", first), exactDocument(t, "b", "text/plain", second)
	analysis := findinganalysis.Document{Schema: findinganalysis.Schema, Sources: []findinganalysis.Source{{ID: "a", Kind: "file", ArtifactRef: a.Ref}, {ID: "b", Kind: "file", ArtifactRef: b.Ref}}, Locations: []findinganalysis.Location{{ID: "a", SourceID: "a", Path: "same.py", FileDigest: a.Digest}, {ID: "b", SourceID: "b", Path: "same.py", FileDigest: b.Digest}}, Flows: []findinganalysis.Flow{}, HTTPExchanges: []findinganalysis.HTTPExchange{}, PrimaryLocationID: "a"}
	attachAnalysis(t, &s, analysis, []auditdomain.FindingCollectionDocument{a, b}, map[string][]byte{a.ID: first, b.ID: second})
	// Git metadata alone does not make this independently named file repo-relative.
	s.Receipts[0].Evidence[0].Git = &findingexport.GitSource{RepositoryURI: "https://example.test/repo.git", Commit: strings.Repeat("a", 40)}
	data, document, diagnostics := rendered(t, s)
	golden(t, "empty-and-multiple-roots", data)
	run, result := runResult(document)
	if len(array(run["artifacts"])) != 2 || len(obj(run["originalUriBaseIds"])) != 2 || len(array(run["versionControlProvenance"])) != 0 || !hasDiagnostic(diagnostics, "primary_region_missing") {
		t.Fatal("source identity collapsed or invented Git mapping")
	}
	if _, exists := obj(obj(array(result["locations"])[0])["physicalLocation"])["region"]; exists {
		t.Fatal("invented empty-file region")
	}
}

func TestRenderHTTPWithoutInventedCode(t *testing.T) {
	s := noAnalysisSnapshot(t, 1)
	payload := []byte(`{"schema":"contractor.http.exchange.v1","capture":{"source":"http","request_id":"1","request_tag":"tag","attempts":[{"request":{"method":"POST","url":"https://example.test/check?token=private-query#secret-fragment","message":{"kind":"structured","headers":[{"name":"Cookie","value":"private-cookie"}],"body_base64":"cHJpdmF0ZS1ib2R5"}},"response":{"status":201,"message":{"kind":"structured","headers":[{"name":"Content-Type","value":"application/json; charset=utf-8"}],"body_base64":"cHJpdmF0ZS1ib2R5"},"complete":true}}]},"finding":{"proposal_id":"proposal","receipt_id":"receipt"}}`)
	doc := exactDocument(t, "http", auditdomain.HTTPExchangeMediaType, payload)
	analysis := findinganalysis.Document{Schema: findinganalysis.Schema, Sources: []findinganalysis.Source{}, Locations: []findinganalysis.Location{}, Flows: []findinganalysis.Flow{}, HTTPExchanges: []findinganalysis.HTTPExchange{{ID: "http", EvidenceRef: doc.Ref, Decoder: "contractor-http-exchange/v1"}}, PrimaryHTTPExchangeID: "http"}
	attachAnalysis(t, &s, analysis, []auditdomain.FindingCollectionDocument{doc}, map[string][]byte{doc.ID: payload})
	data, document, diagnostics := rendered(t, s)
	golden(t, "http-only", data)
	run, result := runResult(document)
	if len(array(run["artifacts"])) != 0 || len(array(result["locations"])) != 0 || len(array(result["codeFlows"])) != 0 || !hasDiagnostic(diagnostics, "http_fields_omitted") || str(obj(result["webRequest"])["target"]) != "https://example.test/check" {
		t.Fatal("HTTP mapping")
	}
	for _, secret := range []string{"password", "private-query", "secret-fragment", "private-cookie", "private-body"} {
		if bytes.Contains(data, []byte(secret)) {
			t.Fatalf("HTTP leak: %s", secret)
		}
	}
}

func TestSemanticValidatorRejectsSchemaValidFabrication(t *testing.T) {
	snapshot := codeSnapshot(t, true)
	original, _, _ := rendered(t, snapshot)
	cases := map[string]func(object, object){
		"unsupported-driver-classification": func(run, result object) {
			obj(obj(run["tool"])["driver"])["supportedTaxonomies"] = []any{object{"name": "CWE@4.20", "index": 0}}
		},
		"unused-root": func(run, result object) {
			obj(run["originalUriBaseIds"])["HOST"] = object{"uri": "file:///private/source/"}
		},
		"invented-symbol": func(run, result object) {
			obj(array(obj(array(result["locations"])[0])["logicalLocations"])[0])["fullyQualifiedName"] = "other.sink"
		},
		"upgraded-flow-proof": func(run, result object) {
			code := obj(array(result["codeFlows"])[0])
			contractor(code)["partial_projection"] = false
		},
		"upgraded-edge-proof": func(run, result object) {
			code := obj(array(result["codeFlows"])[0])
			thread := obj(array(code["threadFlows"])[0])
			point := obj(array(thread["locations"])[1])
			obj(contractor(point)["edge_from_previous"])["basis"] = "tool-reported"
		},
		"dangling-rule": func(run, result object) { result["ruleIndex"] = 99 },
		"dangling-artifact": func(run, result object) {
			obj(obj(array(result["locations"])[0])["physicalLocation"])["artifactLocation"].(map[string]any)["index"] = 99
		},
		"wrong-file-hash": func(run, result object) {
			obj(obj(array(run["artifacts"])[0])["hashes"])["sha-256"] = strings.Repeat("0", 64)
		},
		"invented-region": func(run, result object) {
			obj(obj(obj(array(result["locations"])[0])["physicalLocation"])["region"])["startLine"] = 99
		},
		"dangling-taxonomy": func(run, result object) { obj(obj(array(result["taxa"])[0])["toolComponent"])["index"] = 99 },
		"borrowed-rating":   func(run, result object) { result["level"] = "note" },
		"bridge-native-gap": func(run, result object) {
			flows := array(result["codeFlows"])
			first := obj(array(obj(flows[0])["threadFlows"])[0])
			second := obj(array(obj(flows[1])["threadFlows"])[0])
			points := append(array(first["locations"]), array(second["locations"])...)
			for i, point := range points {
				obj(point)["executionOrder"] = i + 1
			}
			first["locations"] = points
			result["codeFlows"] = flows[:1]
			obj(contractor(result)["flow_segments"])["flow"] = 1
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			var document object
			if err := json.Unmarshal(original, &document); err != nil {
				t.Fatal(err)
			}
			run, result := runResult(document)
			mutate(run, result)
			data, err := strictjson.Canonical(document)
			if err != nil {
				t.Fatal(err)
			}
			parsed, err := validateSchema(data)
			if err != nil {
				t.Fatalf("mutation is not schema-valid: %v", err)
			}
			if validateSemantics(parsed, snapshot) == nil {
				t.Fatal("accepted fabricated projection")
			}
		})
	}
}

func TestRenderRejectsRuleConflictAndUntrustedEdge(t *testing.T) {
	s := codeSnapshot(t, false)
	s.Receipts[0].Analysis.Analysis.Flows[0].Edges[0].Basis = "tool-reported"
	if _, _, err := Render(s); err == nil {
		t.Fatal("accepted unsupported tool proof")
	}
	s = noAnalysisSnapshot(t, 2)
	first := codeSnapshot(t, false).Receipts[0]
	s.Receipts[0] = first
	other := codeSnapshot(t, false).Receipts[0]
	other.ReceiptID, other.ProposalID = s.Receipts[1].ReceiptID, s.Receipts[1].ProposalID
	other.Analysis.Analysis.Rule.Version = "2.0.0"
	s.Receipts[1] = other
	if _, _, err := Render(s); err == nil {
		t.Fatal("accepted conflicting rule definitions")
	}
}

func TestRenderPartialUnlocatedStepStaysNativeOnly(t *testing.T) {
	s := codeSnapshot(t, true)
	flow := &s.Receipts[0].Analysis.Analysis.Flows[0]
	// A missing location splits the first segment; the isolated located point
	// remains related evidence and cannot create a bridge to the later segment.
	flow.Steps[1].LocationID = ""
	flow.Edges = flow.Edges[1:]
	flow.Gaps = append(flow.Gaps, findinganalysis.Gap{AfterStepID: "a", BeforeStepID: "b", Reason: "Unlocated step"})
	_, document, diagnostics := rendered(t, s)
	_, result := runResult(document)
	if len(array(result["codeFlows"])) != 1 || !hasDiagnostic(diagnostics, "flow_coordinates_missing") || !hasDiagnostic(diagnostics, "flow_partial") {
		t.Fatal("unlocated step did not split flow")
	}
	native := obj(obj(contractor(result)["presentation"])["analysis"])
	if len(array(obj(native["analysis"])["flows"])) != 1 {
		t.Fatal("native flow disappeared")
	}
	flow.Completeness = "complete"
	if _, _, err := Render(s); err == nil {
		t.Fatal("accepted a complete data-flow with missing coordinates")
	}
}

func TestRenderCaidoAndExactOpenAPIBinding(t *testing.T) {
	s := noAnalysisSnapshot(t, 1)
	rawRequest := "POST /check?token=private HTTP/1.1\r\nHost: example.test\r\nCookie: private-cookie\r\n\r\nprivate-body"
	rawResponse := "HTTP/1.1 201 Created\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: 12\r\n\r\nprivate-body"
	requestBytes, responseBytes := base64.StdEncoding.EncodeToString([]byte(rawRequest)), base64.StdEncoding.EncodeToString([]byte(rawResponse))
	capture := auditdomain.CapturedHTTPExchange{Source: "caido", RequestID: "request-native", Attempts: []auditdomain.CapturedHTTPAttempt{{Request: &auditdomain.CapturedHTTPRequest{Method: "POST", URL: "https://example.test/check?token=private#secret", Message: auditdomain.HTTPMessage{Kind: "raw", DataBase64: &requestBytes}}, Response: &auditdomain.CapturedHTTPResponse{Status: 201, Complete: true, Message: auditdomain.HTTPMessage{Kind: "raw", DataBase64: &responseBytes}}}}}
	payload, err := auditdomain.EncodeHTTPExchangeArtifact(capture, "proposal", "receipt")
	if err != nil {
		t.Fatal(err)
	}
	doc := exactDocument(t, "caido", auditdomain.HTTPExchangeMediaType, payload)
	specBytes := []byte(`{"openapi":"3.1.0","paths":{"/check":{"post":{"responses":{"201":{"description":"Created"}}}}}}`)
	spec := exactDocument(t, "openapi", "application/json", specBytes)
	analysis := findinganalysis.Document{Schema: findinganalysis.Schema, Sources: []findinganalysis.Source{}, Locations: []findinganalysis.Location{}, Flows: []findinganalysis.Flow{}, HTTPExchanges: []findinganalysis.HTTPExchange{{ID: "http", EvidenceRef: doc.Ref, Decoder: "contractor-http-exchange/v1", Operation: &findinganalysis.Operation{DocumentRef: spec.Ref, Pointer: "/paths/~1check/post", Method: "POST", PathTemplate: "/check"}}}, PrimaryHTTPExchangeID: "http"}
	attachAnalysis(t, &s, analysis, []auditdomain.FindingCollectionDocument{doc, spec}, map[string][]byte{doc.ID: payload, spec.ID: specBytes})
	data, document, _ := rendered(t, s)
	golden(t, "caido-openapi", data)
	_, result := runResult(document)
	if !same(result["webResponse"], object{"statusCode": 201, "headers": object{"Content-Type": "application/json"}}) {
		t.Fatal("raw response was not projected")
	}
	for _, secret := range []string{"private-cookie", "private-body", "token=private", "#secret"} {
		if bytes.Contains(data, []byte(secret)) {
			t.Fatalf("raw HTTP leak: %s", secret)
		}
	}
}

func TestRenderProposalLocationDoesNotInventResolvedSource(t *testing.T) {
	s := noAnalysisSnapshot(t, 1)
	line := int64(17)
	s.Receipts[0].Proposal.Locations = []auditdomain.FindingLocation{{File: "unverified.py", Line: &line}}
	_, document, diagnostics := rendered(t, s)
	run, result := runResult(document)
	if len(array(run["artifacts"])) != 0 || len(array(result["locations"])) != 0 || len(array(result["codeFlows"])) != 0 || !hasDiagnostic(diagnostics, "analysis_missing") {
		t.Fatal("inferred a source from the producer's unverified path")
	}
}
