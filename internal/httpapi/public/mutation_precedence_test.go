package public

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// trackedBody is a request body that counts the reads a handler makes. With
// abort set it ends with io.ErrUnexpectedEOF instead of io.EOF, like a client
// that disconnects before it has sent the length it declared.
type trackedBody struct {
	data  []byte
	reads int
	abort bool
}

func (b *trackedBody) Read(p []byte) (int, error) {
	b.reads++
	if len(b.data) == 0 {
		if b.abort {
			return 0, io.ErrUnexpectedEOF
		}
		return 0, io.EOF
	}
	n := copy(p, b.data)
	b.data = b.data[n:]
	return n, nil
}

// countingProjects counts the Project reads that reach the Project store.
type countingProjects struct {
	ProjectManagement
	gets *atomic.Int64
}

func (p countingProjects) Get(ctx context.Context, ownerID, projectID string) (projectstore.Project, error) {
	p.gets.Add(1)
	return p.ProjectManagement.Get(ctx, ownerID, projectID)
}

// countingAudits counts the Audit lifecycle calls that reach the Audit
// service, so a test can tell that a request got past its header checks even
// when the fake's empty result cannot be rendered.
type countingAudits struct {
	AuditManagement
	calls *atomic.Int64
}

func (a countingAudits) Start(ctx context.Context, params auditservice.StartParams) (auditservice.StartedAudit, error) {
	a.calls.Add(1)
	return a.AuditManagement.Start(ctx, params)
}

func (a countingAudits) Pause(ctx context.Context, params auditservice.MutationParams) (auditservice.MutationResult, error) {
	a.calls.Add(1)
	return a.AuditManagement.Pause(ctx, params)
}

func (a countingAudits) Resume(ctx context.Context, params auditservice.MutationParams) (auditservice.MutationResult, error) {
	a.calls.Add(1)
	return a.AuditManagement.Resume(ctx, params)
}

func (a countingAudits) Cancel(ctx context.Context, params auditservice.MutationParams) (auditservice.MutationResult, error) {
	a.calls.Add(1)
	return a.AuditManagement.Cancel(ctx, params)
}

func (a countingAudits) Delete(ctx context.Context, params auditservice.MutationParams) (auditservice.MutationResult, error) {
	a.calls.Add(1)
	return a.AuditManagement.Delete(ctx, params)
}

// precedenceFixture serves the real handlers over fakes, with an active
// Project project-one and a deleting Project project-deleting of user-1, a
// Project foreign of another owner, and counts of Project-store reads and of
// Audit lifecycle calls.
type precedenceFixture struct {
	handlerFixture
	projectReads *atomic.Int64
	auditCalls   *atomic.Int64
}

func newPrecedenceFixture(t *testing.T) precedenceFixture {
	t.Helper()
	reads, auditCalls := new(atomic.Int64), new(atomic.Int64)
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(services *testServices) {
			services.Projects = countingProjects{ProjectManagement: services.Projects, gets: reads}
			services.Audits = countingAudits{AuditManagement: services.Audits, calls: auditCalls}
		})
	now := time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC)
	for _, project := range []projectstore.Project{
		{ProjectID: "project-one", OwnerID: "user-1", Lifecycle: projectstore.LifecycleActive},
		{ProjectID: "project-deleting", OwnerID: "user-1", Lifecycle: projectstore.LifecycleDeleting},
		{ProjectID: "foreign", OwnerID: "user-2", Lifecycle: projectstore.LifecycleActive},
	} {
		project.Kind, project.Name, project.Revision = projectstore.KindProject, project.ProjectID, 1
		project.CreatedAt, project.UpdatedAt = now, now
		fixture.projects.projects[project.ProjectID] = project
	}
	return precedenceFixture{handlerFixture: fixture, projectReads: reads, auditCalls: auditCalls}
}

// mutationProbes returns a probe for every public mutation, by operationId.
func mutationProbes(document *openapi3.T) map[string]boundProbe {
	probes := map[string]boundProbe{}
	for path, item := range document.Paths.Map() {
		for method, operation := range item.Operations() {
			if method != http.MethodGet {
				probes[operation.OperationID] = boundProbe{method: method, path: path, operation: operation}
			}
		}
	}
	return probes
}

// sortedProbeNames returns the operationIds of probes in order.
func sortedProbeNames(probes map[string]boundProbe) []string {
	names := make([]string, 0, len(probes))
	for name := range probes {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// managedEvalOperation reports an operation whose request errors use the
// Managed Eval codes (API-R69, API-R70).
func managedEvalOperation(operationID string) bool {
	return publicRequestBodyBounds()[operationID].code == "eval_limit_exceeded"
}

// headerCase is one invalid mandatory header of a mutation and the error the
// operation documents for it.
type headerCase struct {
	name   string
	apply  func(http.Header)
	status int
	code   string
}

// mandatoryHeaderCases returns the invalid Idempotency-Key and If-Match
// variants of the operation: a required header that is missing, every key
// outside the shared grammar and every repeated key, and malformed If-Match
// values. Managed Eval routes answer them with their own codes.
func mandatoryHeaderCases(t *testing.T, probe boundProbe) []headerCase {
	t.Helper()
	eval := managedEvalOperation(probe.operation.OperationID)
	invalid := func(name string, apply func(http.Header)) headerCase {
		if eval {
			return headerCase{name, apply, http.StatusUnprocessableEntity, "eval_invalid"}
		}
		return headerCase{name, apply, http.StatusBadRequest, "invalid_request"}
	}
	keys := contractstest.ReadIdempotencyKeyCases(t)
	var cases []headerCase
	for _, parameter := range probe.operation.Parameters {
		value := parameter.Value
		if value == nil || value.In != openapi3.ParameterInHeader {
			continue
		}
		switch {
		case value.Name == idempotencyKeyHeader && value.Required:
			cases = append(cases, invalid("missing Idempotency-Key", func(h http.Header) { h.Del(idempotencyKeyHeader) }))
			rejected := append([][]string(nil), keys.DuplicateHeaders...)
			for _, key := range keys.Invalid {
				rejected = append(rejected, []string{key})
			}
			for _, values := range rejected {
				cases = append(cases, invalid("Idempotency-Key "+strconv.Quote(strings.Join(values, ",")),
					func(h http.Header) { h[idempotencyKeyHeader] = values }))
			}
		case value.Name == "If-Match":
			if value.Required {
				missing := invalid("missing If-Match", func(h http.Header) { h.Del("If-Match") })
				if eval {
					missing.status, missing.code = http.StatusPreconditionRequired, "eval_precondition_required"
				}
				cases = append(cases, missing)
			}
			for _, malformed := range []string{`1`, `W/"1"`, `"1", "2"`, `"1`} {
				cases = append(cases, invalid("If-Match "+malformed, func(h http.Header) {
					h.Del("If-None-Match")
					h.Set("If-Match", malformed)
				}))
			}
		}
	}
	return cases
}

// sendHeaderProbe sends the probe's operation with the headers it requires,
// changed by apply when apply is not nil. An operation with a request body
// gets the JSON body {} (or two bytes of content); one without gets no body,
// so a body-less operation's answer never comes from an unexpected body.
func sendHeaderProbe(t *testing.T, fixture precedenceFixture, probe boundProbe, apply func(http.Header)) (*httptest.ResponseRecorder, errorResponse, *trackedBody) {
	t.Helper()
	body, length := &trackedBody{}, int64(0)
	if probe.operation.RequestBody != nil {
		body.data, length = []byte("{}"), 2
	}
	request := boundProbeRequest(t, probe, body, length)
	if apply != nil {
		apply(request.Header)
	}
	fixture.projectReads.Store(0)
	fixture.auditCalls.Store(0)
	response, failure := serveProbe(fixture.handler, request)
	return response, failure, body
}

// requestErrorOf returns the status and code with which the operation
// answers a malformed request header (API-R69, API-R70).
func requestErrorOf(probe boundProbe) (int, string) {
	if managedEvalOperation(probe.operation.OperationID) {
		return http.StatusUnprocessableEntity, "eval_invalid"
	}
	return http.StatusBadRequest, "invalid_request"
}

// checkPassesHeaderChecks fails unless a request with valid headers got past
// the header checks: an operation with a body read it, and a body-less
// operation reached the Audit service or did not answer with its header
// error.
func checkPassesHeaderChecks(t *testing.T, fixture precedenceFixture, probe boundProbe, label string, response *httptest.ResponseRecorder, failure errorResponse, body *trackedBody) {
	t.Helper()
	status, code := requestErrorOf(probe)
	switch {
	case probe.operation.RequestBody != nil && body.reads == 0:
		t.Errorf("%s = %d %s without reading the body", label, response.Code, response.Body.String())
	case probe.operation.RequestBody == nil && fixture.auditCalls.Load() == 0 &&
		response.Code == status && failure.Code == code:
		t.Errorf("%s = %d %s, rejected like a malformed header", label, response.Code, response.Body.String())
	}
}

// TestSharedMutationHeadersAreRejectedBeforeTheBodyAndStores sends every public
// mutation that declares an Idempotency-Key or If-Match header with each
// invalid value of it and otherwise valid metadata. Each request fails with
// the operation's documented header error before the handler reads a byte of
// its body or reads a Project, while the same request with valid headers
// gets past the header checks: it reads its body, or for a body-less
// operation, which is sent no body, it is not rejected as malformed. So a
// body-less operation such as deleteProject cannot pass only because it
// rejects a body. Operations without these headers have no such case.
func TestSharedMutationHeadersAreRejectedBeforeTheBodyAndStores(t *testing.T) {
	probes := mutationProbes(loadPublicOpenAPI(t))
	checked := 0
	for _, name := range sortedProbeNames(probes) {
		probe := probes[name]
		cases := mandatoryHeaderCases(t, probe)
		if len(cases) == 0 {
			continue
		}
		checked++
		t.Run(name, func(t *testing.T) {
			// A fresh fixture: a valid deleteProject really starts deletion.
			fixture := newPrecedenceFixture(t)
			response, failure, body := sendHeaderProbe(t, fixture, probe, nil)
			checkPassesHeaderChecks(t, fixture, probe, "valid headers", response, failure, body)
			for _, header := range cases {
				response, failure, body := sendHeaderProbe(t, fixture, probe, header.apply)
				if response.Code != header.status || failure.Code != header.code || failure.Retryable {
					t.Errorf("%s = %d %s, want %d %s", header.name, response.Code, response.Body.String(), header.status, header.code)
				}
				if body.reads != 0 || fixture.projectReads.Load() != 0 {
					t.Errorf("%s read the body %d times and %d Projects", header.name, body.reads, fixture.projectReads.Load())
				}
			}
		})
	}
	if checked == 0 {
		t.Fatal("no mutation declares an Idempotency-Key or If-Match header")
	}
}

// counterPrecondition reports an operation whose If-Match is a revision
// counter (the RequiredIfMatch and IfMatch parameters) and returns the
// parameter's OpenAPI pattern.
func counterPrecondition(probe boundProbe) (string, bool) {
	for _, parameter := range probe.operation.Parameters {
		if parameter.Value == nil || parameter.Value.Name != "If-Match" {
			continue
		}
		if strings.HasSuffix(parameter.Ref, "/RequiredIfMatch") || strings.HasSuffix(parameter.Ref, "/IfMatch") {
			return parameter.Value.Schema.Value.Pattern, true
		}
	}
	return "", false
}

// TestSharedRevisionIfMatchCasesGateEveryCounterPrecondition sends every
// public mutation with a counter If-Match each shared case. The OpenAPI
// pattern accepts the valid and out-of-range cases and no invalid one. Every
// rejected value and every repeated If-Match is the operation's request error
// (400 invalid_request, or 422 eval_invalid on Managed Eval routes), never a
// revision conflict, before the body or a Project is read; every valid value
// gets past the header checks. Projects, Audits and their reviews, Runtime
// labels and Agent labels, Scheduler settings and Managed Eval mutations
// therefore read one grammar ("01" is not revision 1).
func TestSharedRevisionIfMatchCasesGateEveryCounterPrecondition(t *testing.T) {
	cases := contractstest.ReadRevisionETagCases(t)
	probes := mutationProbes(loadPublicOpenAPI(t))
	checked := 0
	for _, name := range sortedProbeNames(probes) {
		probe := probes[name]
		pattern, ok := counterPrecondition(probe)
		if !ok {
			continue
		}
		checked++
		t.Run(name, func(t *testing.T) {
			fixture := newPrecedenceFixture(t)
			schema := regexp.MustCompile(pattern)
			for _, valid := range cases.Valid {
				if !schema.MatchString(valid.Header) {
					t.Errorf("OpenAPI pattern %s rejects valid %q", pattern, valid.Header)
				}
				response, failure, body := sendHeaderProbe(t, fixture, probe, func(h http.Header) {
					h.Del("If-None-Match")
					h.Set("If-Match", valid.Header)
				})
				checkPassesHeaderChecks(t, fixture, probe, "If-Match "+valid.Header, response, failure, body)
			}
			for _, raw := range cases.OutOfRange {
				if !schema.MatchString(raw) {
					t.Errorf("OpenAPI pattern %s rejects out-of-range %q", pattern, raw)
				}
			}
			for _, raw := range cases.Invalid {
				if schema.MatchString(raw) {
					t.Errorf("OpenAPI pattern %s accepts invalid %q", pattern, raw)
				}
			}
			status, code := requestErrorOf(probe)
			rejected := [][]string{}
			for _, raw := range cases.Rejected() {
				if raw == "" && managedEvalOperation(name) {
					continue // an empty If-Match is an absent one there: 428, checked by the header test
				}
				rejected = append(rejected, []string{raw})
			}
			for _, values := range append(rejected, cases.RepeatedHeaders...) {
				response, failure, body := sendHeaderProbe(t, fixture, probe, func(h http.Header) {
					h.Del("If-None-Match")
					h["If-Match"] = values
				})
				if response.Code != status || failure.Code != code || failure.Retryable {
					t.Errorf("If-Match %q = %d %s, want %d %s", values, response.Code, response.Body.String(), status, code)
				}
				if body.reads != 0 || fixture.projectReads.Load() != 0 {
					t.Errorf("If-Match %q read the body %d times and %d Projects", values, body.reads, fixture.projectReads.Load())
				}
			}
		})
	}
	if checked == 0 {
		t.Fatal("no mutation declares a counter If-Match")
	}
}

// mediaTypeErrorOf returns the operation's answer to a JSON body with
// another media type: the declared 415, the Managed Eval 422 eval_invalid or
// 400 invalid_request (API-R112).
func mediaTypeErrorOf(probe boundProbe) (int, string) {
	if probe.operation.Responses.Value("415") != nil {
		return http.StatusUnsupportedMediaType, "unsupported_media_type"
	}
	return requestErrorOf(probe)
}

// TestJSONMutationsCheckTheMediaTypeBeforeHeadersAndBody sends every public
// mutation whose request body is JSON with a missing, repeated, parameterized
// or other Content-Type and otherwise valid headers. Each answers its
// documented media-type error before it reads a byte of its body or a
// Project, and the OpenAPI document declares that status. An operation that
// declares 415 answers 415 even when its required If-Match is missing too,
// so the media type comes first. With application/json the same request
// reads its body. An operation whose JSON body is optional (Audit start and
// resume) needs no Content-Type when it announces no body.
func TestJSONMutationsCheckTheMediaTypeBeforeHeadersAndBody(t *testing.T) {
	probes := mutationProbes(loadPublicOpenAPI(t))
	variants := map[string]func(http.Header){
		"missing Content-Type":    func(h http.Header) { h.Del("Content-Type") },
		"text/plain":              func(h http.Header) { h.Set("Content-Type", "text/plain") },
		"charset parameter":       func(h http.Header) { h.Set("Content-Type", "application/json; charset=utf-8") },
		"two Content-Type values": func(h http.Header) { h["Content-Type"] = []string{"application/json", "application/json"} },
	}
	checked := 0
	for _, name := range sortedProbeNames(probes) {
		probe := probes[name]
		if !probe.takesJSON() {
			continue
		}
		checked++
		t.Run(name, func(t *testing.T) {
			fixture := newPrecedenceFixture(t)
			response, failure, body := sendHeaderProbe(t, fixture, probe, nil)
			checkPassesHeaderChecks(t, fixture, probe, "application/json", response, failure, body)
			status, code := mediaTypeErrorOf(probe)
			if probe.operation.Responses.Value(strconv.Itoa(status)) == nil {
				t.Errorf("the OpenAPI document does not declare the media-type status %d", status)
			}
			for variant, apply := range variants {
				response, failure, body := sendHeaderProbe(t, fixture, probe, apply)
				if response.Code != status || failure.Code != code || failure.Retryable {
					t.Errorf("%s = %d %s, want %d %s", variant, response.Code, response.Body.String(), status, code)
				}
				if body.reads != 0 || fixture.projectReads.Load() != 0 {
					t.Errorf("%s read the body %d times and %d Projects", variant, body.reads, fixture.projectReads.Load())
				}
			}
			if status == http.StatusUnsupportedMediaType {
				response, failure, _ := sendHeaderProbe(t, fixture, probe, func(h http.Header) {
					h.Set("Content-Type", "text/plain")
					h.Del("If-Match")
				})
				if response.Code != status || failure.Code != code {
					t.Errorf("text/plain without If-Match = %d %s, want %d %s", response.Code, response.Body.String(), status, code)
				}
			}
			if !probe.operation.RequestBody.Value.Required {
				request := boundProbeRequest(t, probe, &trackedBody{}, 0)
				request.Header.Del("Content-Type")
				fixture.auditCalls.Store(0)
				if response, _ := serveProbe(fixture.handler, request); fixture.auditCalls.Load() != 1 {
					t.Errorf("no body and no Content-Type = %d %s without reaching the Audit service", response.Code, response.Body.String())
				}
			}
		})
	}
	if checked == 0 {
		t.Fatal("no operation takes a JSON body")
	}
}

// TestJSONRequestBodyReadFailuresAreClientErrors sends every public operation
// that takes a JSON body a body that ends before its declared length. The
// read failure is the client's request error (400 invalid_request, the Git
// key's 400 git_key_invalid or the Managed Eval 422 eval_invalid), never a
// retryable internal error. Artifact payloads, whose transfers have their own
// socket deadlines (FAIL-BLB-1), take no JSON body and are not covered here.
func TestJSONRequestBodyReadFailuresAreClientErrors(t *testing.T) {
	fixture := newPrecedenceFixture(t)
	probes := mutationProbes(loadPublicOpenAPI(t))
	checked := 0
	for _, name := range sortedProbeNames(probes) {
		probe := probes[name]
		if !probe.takesJSON() {
			continue
		}
		checked++
		status, code := http.StatusBadRequest, "invalid_request"
		switch {
		case managedEvalOperation(name):
			status, code = http.StatusUnprocessableEntity, "eval_invalid"
		case name == "replaceGitKey":
			code = "git_key_invalid"
		}
		body := &trackedBody{data: []byte(`{"`), abort: true}
		response, failure := serveProbe(fixture.handler, boundProbeRequest(t, probe, body, 64))
		if body.reads == 0 {
			t.Errorf("%s = %d %s without reading the body", name, response.Code, response.Body.String())
			continue
		}
		if response.Code != status || failure.Code != code || failure.Retryable {
			t.Errorf("%s with an aborted body = %d %s, want %d %s", name, response.Code, response.Body.String(), status, code)
		}
	}
	if checked == 0 {
		t.Fatal("no operation takes a JSON body")
	}
}

// TestRuntimeConfigPublishBodyReadFailurePublishesNothing holds RuntimeConfig
// publication to the shared body read: a body that fails mid-read is 400
// invalid_request and publishes no version, an empty body is 400, and a body
// over the 128 KiB bound stays 413 request_too_large, unread when its length
// is declared and stopped at the bound when it is not.
func TestRuntimeConfigPublishBodyReadFailurePublishesNothing(t *testing.T) {
	fixture := newHandlerFixture(t)
	document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"aborted","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces"}}}}`)
	publish := func(body *trackedBody, declared int64) (int, errorResponse) {
		t.Helper()
		request := newPublicContractRequest(http.MethodPost, "/v1/operations/runtime-configs", nil)
		request.Body, request.ContentLength = io.NopCloser(body), declared
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set(idempotencyKeyHeader, "publish-aborted")
		response, failure := serveProbe(fixture.handler, request)
		return response.Code, failure
	}
	versions := func() int {
		fixture.runtimeConfigs.mu.Lock()
		defer fixture.runtimeConfigs.mu.Unlock()
		return len(fixture.runtimeConfigs.versions)
	}
	before := versions()

	aborted := &trackedBody{data: document[:len(document)/2], abort: true}
	if status, failure := publish(aborted, int64(len(document))); status != http.StatusBadRequest ||
		failure.Code != "invalid_request" || failure.Retryable || aborted.reads == 0 {
		t.Fatalf("aborted body = %d %+v after %d reads, want 400 invalid_request", status, failure, aborted.reads)
	}
	if status, failure := publish(&trackedBody{}, 0); status != http.StatusBadRequest || failure.Code != "invalid_request" {
		t.Fatalf("empty body = %d %+v, want 400 invalid_request", status, failure)
	}
	oversized := bytes.Repeat([]byte(" "), int(maximumRuntimeConfigBodyBytes)+1)
	declared := &trackedBody{data: oversized}
	if status, failure := publish(declared, int64(len(oversized))); status != http.StatusRequestEntityTooLarge ||
		failure.Code != "request_too_large" || declared.reads != 0 {
		t.Fatalf("declared oversized body = %d %+v after %d reads, want 413 request_too_large unread", status, failure, declared.reads)
	}
	if status, failure := publish(&trackedBody{data: oversized}, -1); status != http.StatusRequestEntityTooLarge ||
		failure.Code != "request_too_large" {
		t.Fatalf("undeclared oversized body = %d %+v, want 413 request_too_large", status, failure)
	}
	if after := versions(); after != before {
		t.Fatalf("rejected bodies published %d RuntimeConfig versions", after-before)
	}
	if status, failure := publish(&trackedBody{data: document}, int64(len(document))); status != http.StatusCreated {
		t.Fatalf("complete body = %d %+v, want 201", status, failure)
	}
}

// precedenceCase is one request of a mutation family and its documented
// outcome.
type precedenceCase struct {
	name, method, path string
	headers            map[string]string
	body               string
	// declared overrides the declared body length when it is not zero.
	declared     int64
	status       int
	code         string
	readsBody    bool
	projectReads int64
}

func (c precedenceCase) run(t *testing.T, fixture precedenceFixture) {
	t.Helper()
	body := &trackedBody{data: []byte(c.body)}
	request := newPublicContractRequest(c.method, c.path, nil)
	request.Body, request.ContentLength = io.NopCloser(body), int64(len(c.body))
	if c.declared != 0 {
		request.ContentLength = c.declared
	}
	for name, value := range c.headers {
		request.Header.Set(name, value)
	}
	fixture.projectReads.Store(0)
	response, failure := serveProbe(fixture.handler, request)
	if response.Code != c.status || failure.Code != c.code {
		t.Errorf("%s = %d %s, want %d %s", c.name, response.Code, response.Body.String(), c.status, c.code)
	}
	if (body.reads != 0) != c.readsBody {
		t.Errorf("%s read its body %d times, want read %t", c.name, body.reads, c.readsBody)
	}
	if reads := fixture.projectReads.Load(); reads != c.projectReads {
		t.Errorf("%s read %d Projects, want %d", c.name, reads, c.projectReads)
	}
}

// TestProjectArtifactPutValidatesMetadataBeforeTheProject holds the order of
// PUT /v1/projects/{projectId}/artifacts/{namespace}/{name}: query, names,
// media type and precondition are checked before the Project is read, so a
// malformed request reads no Project whichever Project it names; then
// Project ownership (the non-disclosing 404) and the deleting state (409),
// then the Audit-managed namespace (the same 404), and only then the body.
func TestProjectArtifactPutValidatesMetadataBeforeTheProject(t *testing.T) {
	fixture := newPrecedenceFixture(t)
	put := func(name, path string, headers map[string]string, status int, code string, readsBody bool, projectReads int64) precedenceCase {
		merged := map[string]string{"Content-Type": "text/plain", "If-None-Match": "*"}
		for key, value := range headers {
			if value == "" {
				delete(merged, key)
			} else {
				merged[key] = value
			}
		}
		return precedenceCase{
			name: name, method: http.MethodPut, path: path, headers: merged, body: "report",
			status: status, code: code, readsBody: readsBody, projectReads: projectReads,
		}
	}
	const owned, missing = "/v1/projects/project-one/artifacts/", "/v1/projects/project-missing/artifacts/"
	cases := []precedenceCase{
		put("valid", owned+"sources/report", nil, http.StatusCreated, "", true, 1),
		put("unknown query", owned+"sources/report?mode=x", nil, http.StatusBadRequest, "invalid_request", false, 0),
		put("unknown query, missing Project", missing+"sources/report?mode=x", nil, http.StatusBadRequest, "invalid_request", false, 0),
		put("invalid namespace", missing+"_sources/report", nil, http.StatusBadRequest, "invalid_request", false, 0),
		put("invalid name", "/v1/projects/project-deleting/artifacts/sources/r%20eport", nil, http.StatusBadRequest, "invalid_request", false, 0),
		put("missing Content-Type", missing+"sources/report", map[string]string{"Content-Type": ""}, http.StatusBadRequest, "invalid_request", false, 0),
		put("Content-Type with parameters", missing+"sources/report", map[string]string{"Content-Type": "text/plain; charset=utf-8"}, http.StatusBadRequest, "invalid_request", false, 0),
		put("If-Match with If-None-Match", missing+"sources/report", map[string]string{"If-Match": `"rev-1"`}, http.StatusBadRequest, "invalid_request", false, 0),
		put("weak If-Match", missing+"sources/report", map[string]string{"If-None-Match": "", "If-Match": `W/"rev-1"`}, http.StatusBadRequest, "invalid_request", false, 0),
		put("If-None-Match other than *", missing+"sources/report", map[string]string{"If-None-Match": `"rev-1"`}, http.StatusBadRequest, "invalid_request", false, 0),
		put("Audit-managed namespace, malformed", owned+"audit-run/report?x=1", nil, http.StatusBadRequest, "invalid_request", false, 0),
		put("missing Project", missing+"sources/report", nil, http.StatusNotFound, "not_found", false, 1),
		put("other owner's Project", "/v1/projects/foreign/artifacts/sources/report", nil, http.StatusNotFound, "not_found", false, 1),
		put("deleting Project", "/v1/projects/project-deleting/artifacts/sources/report", nil, http.StatusConflict, "project_deleting", false, 1),
		put("Audit-managed namespace", owned+"audit-run/report", nil, http.StatusNotFound, "not_found", false, 1),
		put("Audit-managed namespace, missing Project", missing+"audit-run/report", nil, http.StatusNotFound, "not_found", false, 1),
	}
	for _, c := range cases {
		c.run(t, fixture)
	}
}

// TestMutationPrecedenceTable documents, through the real handlers, the order
// in which the affected mutation families check a request: query and media
// type with each endpoint's own status, then Idempotency-Key and If-Match,
// then Project ownership where the route names a Project, then the body
// (its size, then strict JSON). A request that fails a header check reads
// neither its body nor a Project, and a valid request still succeeds.
func TestMutationPrecedenceTable(t *testing.T) {
	fixture := newPrecedenceFixture(t)
	jsonBody := map[string]string{"Content-Type": "application/json"}
	with := func(base map[string]string, pairs ...string) map[string]string {
		merged := map[string]string{}
		for key, value := range base {
			merged[key] = value
		}
		for index := 0; index < len(pairs); index += 2 {
			merged[pairs[index]] = pairs[index+1]
		}
		return merged
	}
	keyed := with(jsonBody, idempotencyKeyHeader, "precedence-key")
	project := `{"kind":"project","name":"Ordered"}`
	runtimeConfig := `{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"ordered","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces"}}}}`
	oversized := int64(maxJSONRequestSize) + 1
	cases := []precedenceCase{
		// Projects.
		{name: "createProject valid", method: http.MethodPost, path: "/v1/projects", headers: keyed, body: project,
			status: http.StatusCreated, readsBody: true},
		{name: "createProject text/plain", method: http.MethodPost, path: "/v1/projects",
			headers: with(keyed, "Content-Type", "text/plain"), body: project,
			status: http.StatusBadRequest, code: "invalid_request"},
		{name: "createProject without key", method: http.MethodPost, path: "/v1/projects", headers: jsonBody, body: project,
			status: http.StatusBadRequest, code: "invalid_request"},
		{name: "createProject without key, oversized", method: http.MethodPost, path: "/v1/projects", headers: jsonBody,
			body: project, declared: oversized, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "createProject oversized", method: http.MethodPost, path: "/v1/projects", headers: keyed,
			body: project, declared: oversized, status: http.StatusRequestEntityTooLarge, code: "request_too_large"},
		{name: "createProject duplicate member", method: http.MethodPost, path: "/v1/projects", headers: keyed,
			body: `{"kind":"project","name":"A","name":"B"}`, status: http.StatusBadRequest, code: "invalid_request", readsBody: true},
		// Runs.
		{name: "createRun without key", method: http.MethodPost, path: "/v1/runs", headers: jsonBody,
			body: `{"workflow":"artifact-copy@1"}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "createRun unknown member", method: http.MethodPost, path: "/v1/runs", headers: keyed,
			body: `{"Workflow":"artifact-copy@1"}`, status: http.StatusBadRequest, code: "invalid_request", readsBody: true},
		{name: "createProjectRun without key", method: http.MethodPost, path: "/v1/projects/project-one/runs", headers: jsonBody,
			body: `{"workflow":"artifact-copy@1"}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "createProjectRun without key, missing Project", method: http.MethodPost, path: "/v1/projects/project-missing/runs",
			headers: jsonBody, body: `{"workflow":"artifact-copy@1"}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "createProjectRun text/plain, missing Project", method: http.MethodPost, path: "/v1/projects/project-missing/runs",
			headers: with(keyed, "Content-Type", "text/plain"), body: `{"workflow":"artifact-copy@1"}`,
			status: http.StatusBadRequest, code: "invalid_request"},
		{name: "createProjectRun without Content-Type", method: http.MethodPost, path: "/v1/projects/project-one/runs",
			headers: map[string]string{idempotencyKeyHeader: "precedence-key"}, body: `{"workflow":"artifact-copy@1"}`,
			status: http.StatusBadRequest, code: "invalid_request"},
		{name: "cancelRun text/plain", method: http.MethodPost, path: "/v1/runs/run-one/cancel",
			headers: map[string]string{"Content-Type": "text/plain"}, body: `{}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "createProjectRun missing Project", method: http.MethodPost, path: "/v1/projects/project-missing/runs",
			headers: keyed, body: `{"workflow":"artifact-copy@1"}`, status: http.StatusNotFound, code: "not_found", projectReads: 1},
		{name: "createProjectRun other owner's Project", method: http.MethodPost, path: "/v1/projects/foreign/runs",
			headers: keyed, body: `{"workflow":"artifact-copy@1"}`, status: http.StatusNotFound, code: "not_found", projectReads: 1},
		{name: "createProjectRun trailing data", method: http.MethodPost, path: "/v1/projects/project-one/runs",
			headers: keyed, body: `{"workflow":"artifact-copy@1"} {}`, status: http.StatusBadRequest, code: "invalid_request",
			readsBody: true, projectReads: 1},
		// Audits.
		{name: "createAudit without key", method: http.MethodPost, path: "/v1/projects/project-one/audits", headers: jsonBody,
			body: `{}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "startAudit without If-Match", method: http.MethodPost, path: "/v1/audits/audit-one/start",
			headers: keyed, body: `{"deadlineSeconds":60}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "startAudit text/plain body", method: http.MethodPost, path: "/v1/audits/audit-one/start",
			headers: with(keyed, "Content-Type", "text/plain", "If-Match", `"1"`), body: `{"deadlineSeconds":60}`,
			status: http.StatusBadRequest, code: "invalid_request"},
		{name: "decideAuditReview text/plain", method: http.MethodPost, path: "/v1/audits/audit-one/reviews/request-one/decisions",
			headers: with(keyed, "Content-Type", "text/plain", "If-Match", `"1"`), body: `{}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "startAudit oversized", method: http.MethodPost, path: "/v1/audits/audit-one/start",
			headers: with(keyed, "If-Match", `"1"`), body: "{}" + strings.Repeat(" ", int(maximumAuditTimeLimitBodyBytes)-1),
			status: http.StatusRequestEntityTooLarge, code: "request_too_large", readsBody: true},
		{name: "pauseAudit without key", method: http.MethodPost, path: "/v1/audits/audit-one/pause",
			headers: map[string]string{"If-Match": `"1"`}, body: "x", status: http.StatusBadRequest, code: "invalid_request"},
		{name: "deleteAudit malformed If-Match", method: http.MethodDelete, path: "/v1/audits/audit-one",
			headers: map[string]string{idempotencyKeyHeader: "precedence-key", "If-Match": "1"}, body: "x",
			status: http.StatusBadRequest, code: "invalid_request"},
		{name: "decideAuditReview without key", method: http.MethodPost, path: "/v1/audits/audit-one/reviews/request-one/decisions",
			headers: with(jsonBody, "If-Match", `"1"`), body: `{}`, status: http.StatusBadRequest, code: "invalid_request"},
		// Configuration and credentials.
		{name: "publishConfiguration text/plain", method: http.MethodPost, path: "/v1/configurations/agent-templates",
			headers: with(keyed, "Content-Type", "text/plain"), body: `{}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "publishConfiguration without key", method: http.MethodPost, path: "/v1/configurations/agent-templates",
			headers: jsonBody, body: `{}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "createCredential without key", method: http.MethodPost, path: "/v1/operations/credentials",
			headers: jsonBody, body: `{}`, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "createRuntimeCredential valid", method: http.MethodPost, path: "/v1/operations/runtime-credentials", headers: keyed,
			body:   `{"credentialId":"ordered","kind":"otlp-headers@1","material":{"headers":{"authorization":"x"}}}`,
			status: http.StatusCreated, readsBody: true},
		{name: "publishRuntimeConfig without key", method: http.MethodPost, path: "/v1/operations/runtime-configs",
			headers: jsonBody, body: runtimeConfig, status: http.StatusBadRequest, code: "invalid_request"},
		{name: "publishRuntimeConfig duplicate member", method: http.MethodPost, path: "/v1/operations/runtime-configs",
			headers: with(keyed, idempotencyKeyHeader, "publish-duplicate"),
			body:    strings.Replace(runtimeConfig, `"kind":"RuntimeConfig"`, `"kind":"RuntimeConfig","kind":"RuntimeConfig"`, 1),
			status:  http.StatusBadRequest, code: "runtime_config_invalid", readsBody: true},
		{name: "publishRuntimeConfig valid", method: http.MethodPost, path: "/v1/operations/runtime-configs", headers: keyed,
			body: runtimeConfig, status: http.StatusCreated, readsBody: true},
		{name: "putModelSelection text/plain", method: http.MethodPut, path: "/v1/operations/settings/models",
			headers: map[string]string{"Content-Type": "text/plain"}, body: `{}`,
			status: http.StatusUnsupportedMediaType, code: "unsupported_media_type"},
		{name: "putSchedulerSettings text/plain", method: http.MethodPut, path: "/v1/operations/settings/scheduler",
			headers: map[string]string{"Content-Type": "text/plain"}, body: `{}`,
			status: http.StatusUnsupportedMediaType, code: "unsupported_media_type"},
		{name: "mutateProjectSitemap without key", method: http.MethodPost, path: "/v1/projects/project-one/sitemap/mutations", headers: jsonBody, body: `{}`, status: http.StatusBadRequest, code: "invalid_request"},
	}
	for _, c := range cases {
		c.run(t, fixture)
	}
}

// TestArtifactPayloadInterruptionIsRetryableAndStoresNothing pins the outcome
// of an Artifact payload whose upload stops before its declared length, the
// same failure a missed socket read deadline produces (BLB-R28, FAIL-BLB-1):
// 500 internal_error with retryable true on the user and the Project upload
// route, and no binding. The complete payload is then accepted. A JSON body,
// the Git import's included, is the client's 400 instead
// (TestJSONRequestBodyReadFailuresAreClientErrors).
func TestArtifactPayloadInterruptionIsRetryableAndStoresNothing(t *testing.T) {
	fixture := newPrecedenceFixture(t)
	for _, path := range []string{"/v1/artifacts/sources/interrupted", "/v1/projects/project-one/artifacts/sources/interrupted"} {
		upload := func(body *trackedBody, declared int64) (*httptest.ResponseRecorder, errorResponse) {
			request := newPublicContractRequest(http.MethodPut, path, nil)
			request.Body, request.ContentLength = io.NopCloser(body), declared
			request.Header.Set("Content-Type", "text/plain")
			request.Header.Set("If-None-Match", "*")
			return serveProbe(fixture.handler, request)
		}
		interrupted := &trackedBody{data: []byte("part"), abort: true}
		response, failure := upload(interrupted, 64)
		if response.Code != http.StatusInternalServerError || failure.Code != "internal_error" || !failure.Retryable || interrupted.reads == 0 {
			t.Errorf("PUT %s interrupted = %d %s after %d reads, want retryable 500 internal_error", path, response.Code, response.Body.String(), interrupted.reads)
		}
		read, _ := serveProbe(fixture.handler, newPublicContractRequest(http.MethodGet, path, nil))
		if read.Code != http.StatusNotFound {
			t.Errorf("GET %s after an interrupted upload = %d %s, want 404", path, read.Code, read.Body.String())
		}
		if response, _ := upload(&trackedBody{data: []byte("complete")}, 8); response.Code != http.StatusCreated {
			t.Errorf("PUT %s complete = %d %s, want 201", path, response.Code, response.Body.String())
		}
	}
}
