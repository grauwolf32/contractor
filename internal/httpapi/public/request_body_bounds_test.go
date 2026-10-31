package public

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// requestBodyBound is the bound a handler enforces on an operation's request
// body and the error a larger body gets.
type requestBodyBound struct {
	bytes  int64
	status int
	code   string
}

// publicRequestBodyBounds is the request body bound of every public operation
// that takes a body, as its handler enforces it. Endpoint-specific bounds
// stay distinct: login (LIM-SEC-8), the Git key and import bodies (LIM-GIT-21,
// LIM-GIT-5), the Audit start and resume time limit (LIM-AUD-28), RuntimeConfig
// publication, Artifact payloads (LIM-ART-4) and Eval documents (API-R6);
// every other JSON body has the shared 1 MiB bound (LIM-API-1).
func publicRequestBodyBounds() map[string]requestBodyBound {
	tooLarge := func(byteCount int64) requestBodyBound {
		return requestBodyBound{bytes: byteCount, status: http.StatusRequestEntityTooLarge, code: "request_too_large"}
	}
	bounds := map[string]requestBodyBound{
		"publishAuditMaterial":     tooLarge(maxJSONRequestSize),
		"mutateProjectSitemap":     tooLarge(projectsitemap.MaxBatchBytes),
		"login":                    tooLarge(maximumLoginBodyBytes),
		"replaceGitKey":            tooLarge(maximumGitKeyBodyBytes),
		"importGitArtifact":        tooLarge(maximumGitImportBodyBytes),
		"importProjectGitArtifact": tooLarge(maximumGitImportBodyBytes),
		"startAudit":               tooLarge(maximumAuditTimeLimitBodyBytes),
		"resumeAudit":              tooLarge(maximumAuditTimeLimitBodyBytes),
		"publishRuntimeConfig":     tooLarge(maximumRuntimeConfigBodyBytes),
	}
	for _, operation := range []string{"putArtifact", "putProjectArtifact"} {
		bounds[operation] = requestBodyBound{
			bytes: artifacts.MaxPayloadSize, status: http.StatusRequestEntityTooLarge, code: "artifact_too_large",
		}
	}
	for _, operation := range []string{
		"importEvalDataset", "createEvalExperiment", "updateEvalDraft", "deleteEvalExperiment",
		"commandEvalExperiment", "submitEvalMember", "ingestEvalResult", "assessEvalMember", "selectEvalRecords",
	} {
		bounds[operation] = requestBodyBound{
			bytes: evaldomain.MaxDocumentBytes, status: http.StatusUnprocessableEntity, code: "eval_limit_exceeded",
		}
	}
	for _, operation := range []string{
		"publishFindingCollection", "publishAuditFindingExport", "createProject", "updateProject", "createProjectRun", "createAudit",
		"importAuditFindingProposal", "createAuditFindingReview", "decideAuditReview", "putOwnerQueueControl",
		"createRun", "retryRunGateway", "resumeRun", "cancelRun", "publishConfiguration", "putRuntimeLabel",
		"createRuntimeCredential", "replaceRuntimeAgentPrincipalLabels", "createCredential", "putModelSelection",
		"putSchedulerSettings",
	} {
		bounds[operation] = tooLarge(maxJSONRequestSize)
	}
	return bounds
}

// boundProbePathValues are valid path parameter values for the body probes.
var boundProbePathValues = map[string]string{
	"auditId": "audit-one", "credentialId": "credential-one", "findingId": "finding-one",
	"id": "experiment-one", "kind": "agent-templates", "label": "label-one", "memberId": strings.Repeat("a", 64),
	"name": "source", "namespace": "sources", "projectId": "project-one", "requestId": "request-one",
	"runId": "run-one", "runtimeAgentId": strings.Repeat("b", 64),
}

type boundProbe struct {
	method, path string
	operation    *openapi3.Operation
}

// TestPublicRequestBodyBoundsMatchHandlers holds each operation's declared
// x-max-request-body-bytes to the bound its handler enforces. Every
// operation with a request body declares one, and only those do. Through the
// real handler, a body one byte over the bound fails with the operation's
// documented class (413 request_too_large or artifact_too_large, or the
// Managed Eval 422 eval_limit_exceeded), and a body exactly at the bound is
// read, so it fails, if at all, for another reason.
func TestPublicRequestBodyBoundsMatchHandlers(t *testing.T) {
	document := loadPublicOpenAPI(t)
	bounds := publicRequestBodyBounds()
	probes := map[string]boundProbe{}
	for path, item := range document.Paths.Map() {
		for method, operation := range item.Operations() {
			declared, hasDeclaration := operation.Extensions["x-max-request-body-bytes"]
			if operation.RequestBody == nil {
				if hasDeclaration {
					t.Errorf("%s declares x-max-request-body-bytes without a request body", operation.OperationID)
				}
				continue
			}
			bound, known := bounds[operation.OperationID]
			if !known {
				t.Errorf("%s takes a request body that publicRequestBodyBounds does not list", operation.OperationID)
				continue
			}
			var value int64
			raw, err := json.Marshal(declared)
			if !hasDeclaration || err != nil || json.Unmarshal(raw, &value) != nil || value != bound.bytes {
				t.Errorf("%s declares x-max-request-body-bytes %v, want the handler bound %d", operation.OperationID, declared, bound.bytes)
			}
			probes[operation.OperationID] = boundProbe{method: method, path: path, operation: operation}
		}
	}
	for operation := range bounds {
		if _, found := probes[operation]; !found {
			t.Errorf("publicRequestBodyBounds lists %s, which takes no request body", operation)
		}
	}
	if t.Failed() {
		t.FailNow()
	}

	fixture := newHandlerFixture(t)
	now := time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC)
	fixture.projects.projects[boundProbePathValues["projectId"]] = projectstore.Project{
		ProjectID: boundProbePathValues["projectId"], OwnerID: "user-1", Kind: projectstore.KindProject,
		Lifecycle: projectstore.LifecycleActive, Name: "Bounds", Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	names := make([]string, 0, len(probes))
	for name := range probes {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		probe, bound := probes[name], bounds[name]
		t.Run(name, func(t *testing.T) {
			over, failure := sendBoundProbe(t, fixture.handler, probe, bound.bytes+1)
			if over.Code != bound.status || failure.Code != bound.code {
				t.Fatalf("%d-byte body = %d %s, want %d %s", bound.bytes+1, over.Code, over.Body.String(), bound.status, bound.code)
			}
			if bound.code == "artifact_too_large" {
				return // An at-bound probe would store a 64 MiB Artifact.
			}
			at, failure := sendBoundProbe(t, fixture.handler, probe, bound.bytes)
			if at.Code == bound.status && failure.Code == bound.code {
				t.Fatalf("%d-byte body = %d %s, want it read", bound.bytes, at.Code, at.Body.String())
			}
		})
	}
}

// takesJSON reports whether the probe's operation takes a JSON request body;
// the other bodies are Artifact payloads of any media type.
func (p boundProbe) takesJSON() bool {
	if p.operation.RequestBody == nil {
		return false
	}
	_, isJSON := p.operation.RequestBody.Value.Content["application/json"]
	return isJSON
}

// sendBoundProbe sends size bytes of JSON whitespace after {} (or of Artifact
// content) with the headers the operation requires, so the handler reaches
// its body read.
func sendBoundProbe(t *testing.T, handler http.Handler, probe boundProbe, size int64) (*httptest.ResponseRecorder, errorResponse) {
	t.Helper()
	body := bytes.Repeat([]byte(" "), int(size))
	if probe.takesJSON() {
		copy(body, "{}")
	}
	return serveProbe(handler, boundProbeRequest(t, probe, bytes.NewReader(body), size))
}

// boundProbeRequest returns a request for the probe's operation that carries
// the headers the operation requires and body, which declares contentLength
// bytes.
func boundProbeRequest(t *testing.T, probe boundProbe, body io.Reader, contentLength int64) *http.Request {
	t.Helper()
	path := probe.path
	for name, value := range boundProbePathValues {
		path = strings.ReplaceAll(path, "{"+name+"}", value)
	}
	if strings.Contains(path, "{") {
		t.Fatalf("no probe value for a parameter of %s", probe.path)
	}
	request := newPublicContractRequest(probe.method, path, nil)
	request.Body, request.ContentLength = io.NopCloser(body), contentLength
	contentType := "text/plain"
	if probe.takesJSON() {
		contentType = "application/json"
	}
	request.Header.Set("Content-Type", contentType)
	for _, parameter := range probe.operation.Parameters {
		if parameter.Value == nil || parameter.Value.In != openapi3.ParameterInHeader {
			continue
		}
		switch parameter.Value.Name {
		case idempotencyKeyHeader:
			request.Header.Set(idempotencyKeyHeader, "body-bound")
		case "If-Match":
			switch {
			case strings.HasSuffix(parameter.Ref, "/ArtifactIfMatch"):
				request.Header.Set("If-None-Match", "*")
			case strings.HasSuffix(parameter.Ref, "/RequiredModelSelectionIfMatch"):
				request.Header.Set("If-Match", `"1:`+strings.Repeat("c", 64)+`"`)
			default:
				request.Header.Set("If-Match", `"1"`)
			}
		}
	}
	if probe.operation.OperationID == "login" {
		request.Header.Del("Authorization")
		request.Header.Set("Origin", testBrowserOrigin)
	}
	return request
}

func serveProbe(handler http.Handler, request *http.Request) (*httptest.ResponseRecorder, errorResponse) {
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	var failure errorResponse
	_ = json.Unmarshal(response.Body.Bytes(), &failure)
	return response, failure
}
