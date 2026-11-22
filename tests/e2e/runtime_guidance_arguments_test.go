//go:build e2e

package e2e

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func runtimeArgumentContractsScenario(t *testing.T) {
	secure := newEngagementTargetFixture(t)
	target := newEngagementHTTPObserver(t, engagementFixtureIP(t, secure))
	const structured = `{"amount":"8100","enabled":true,"nested":[null,1.5,"{}"]}`
	const exact = " \r\n{\"amount\":8100,\"amount\":0, broken 雪\\n} \t"
	var mu sync.Mutex
	bodies := map[string][]byte{}
	target.setHandler(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(io.LimitReader(r.Body, 1025))
		if err != nil || len(body) > 1024 || r.Method != "POST" || r.Header.Get("Content-Type") != "application/json" {
			t.Error("invalid request at local receiver")
		}
		mu.Lock()
		if _, exists := bodies[r.URL.Path]; exists {
			t.Error("unexpected semantic resend")
		}
		bodies[r.URL.Path] = body
		mu.Unlock()
		if r.URL.Path == "/api/structured" && (r.URL.RawQuery != "amount=01" || r.Header.Get("X-Note") != "kept") {
			t.Error("container values changed")
		}
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("intentional fixture 400"))
	})
	h := startEngagementProcessHarness(t, secure.caFile, "pass")
	pool := engagementPool(t, h)
	route := engagementRoute(target.server.URL, "/api")
	route.Methods = []string{"POST"}
	audit := startEngagementMatrixAudit(t, h, []string{target.server.URL}, nil, []engagementcontract.Route{route}, nil, 1024, func() {
		stage := engagementRequestStage(nil)
		steps := stage.steps[:2]
		steps = append(steps, toolGatewayStep("http_request", fixedArguments(map[string]any{
			"url": target.server.URL + "/api/structured", "method": "POST", "headers": `{"secret-canary":`,
		})))
		corrected := toolGatewayStep("http_request", fixedArguments(map[string]any{
			"url": target.server.URL + "/api/structured", "method": "POST", "headers": `{"X-Note":"kept"}`,
			"query": `{"amount":"01"}`, "body_type": "json", "body": structured,
		}))
		corrected.validate = requireArgumentDiagnostic("http_request", "http_request_invalid", "headers", "object")
		steps = append(steps, corrected, toolGatewayStep("http_request", fixedArguments(map[string]any{
			"url": target.server.URL + "/api/exact", "method": "POST", "headers": map[string]any{"Content-Type": "application/json"},
			"body_type": "text", "body": exact,
		})))
		batch := func(bad bool) func(map[string]any) (map[string]any, error) {
			return func(request map[string]any) (map[string]any, error) {
				value, err := auditResultArguments(map[string]any{
					"rationale":         "Observed two explicit HTTP responses, without inferring a vulnerability.",
					"tested_conditions": `["explicit HTTP probes"]`, "limitations": `["No vulnerability verdict"]`,
					"evidence": `[]`, "http_request_ids": `[1,2]`,
				})(request)
				if err != nil {
					return nil, err
				}
				if bad {
					value["limitations"] = `["secret-canary"`
				}
				encoded, err := json.Marshal([]map[string]any{value})
				return map[string]any{"items": string(encoded)}, err
			}
		}
		steps = append(steps, toolGatewayStep("submit_check_results", batch(true)))
		fixed := toolGatewayStep("submit_check_results", batch(false))
		fixed.validate = requireArgumentDiagnostic("submit_check_results", "audit_result_invalid", "items[0].limitations", "array")
		final := finalGatewayStep("Recorded canonical results", nil)
		final.validate = func(request map[string]any) error {
			if err := validateExecutionFacts(request, "remainingToolCalls"); err != nil {
				return err
			}
			results := findingsToolResponses(request, "submit_check_results")
			if len(results) != 2 || results[1]["complete"] != true || results[1]["normalization"] == nil {
				return fmt.Errorf("batch correction did not complete once: %v", results)
			}
			return nil
		}
		steps = append(steps, fixed, final)
		stage.steps = steps
		h.gateway.mu.Lock()
		h.gateway.stages = []domainGatewayStage{stage}
		h.gateway.mu.Unlock()
	})
	waitEngagementAuditClosed(t, h, pool, audit)
	tasks := commonPreparedTaskPages(t, h.findingsHarness, audit)
	if len(tasks) != 1 || tasks[0].Result == nil || tasks[0].GapCode != nil {
		t.Fatalf("canonical result absent: %+v; gateway %v", tasks, h.gateway.Failures())
	}
	var document struct {
		Schema string `json:"schema"`
		Items  []struct {
			TestedConditions []string                       `json:"tested_conditions"`
			Evidence         []contracts.MaterialDescriptor `json:"evidence"`
		} `json:"items"`
	}
	if err := json.Unmarshal(basicTaskArtifact(t, h.findingsHarness, audit, tasks[0], "result"), &document); err != nil {
		t.Fatal(err)
	}
	if document.Schema != "contractor.audit.result.v1" || len(document.Items) != 1 || len(document.Items[0].Evidence) != 2 || len(document.Items[0].TestedConditions) != 1 {
		t.Fatalf("noncanonical result: %+v", document)
	}
	var runID string
	for _, material := range document.Items[0].Evidence {
		if material.ScopeKind != "audit" || material.ScopeID != audit || material.MediaType != auditdomain.HTTPExchangeMediaType {
			t.Fatalf("foreign evidence: %+v", material)
		}
		descriptor, err := strictjson.Canonical(material)
		if err != nil {
			t.Fatal(err)
		}
		key := "evidence/" + strings.TrimPrefix(contentdigest.Bytes(descriptor), "sha256:")
		path := "/v1/audits/" + audit + "/artifacts?" + url.Values{"key": {key}, "revision": {*material.Ref.Revision}}.Encode()
		request, err := http.NewRequestWithContext(h.ctx, http.MethodGet, h.baseURL+path, nil)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		response := do(t, h.client, request, http.StatusOK)
		payload, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		response.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		capture, err := auditdomain.DecodeHTTPExchangeArtifact(payload)
		if err != nil || len(capture.Capture.Attempts) != 1 {
			t.Fatalf("invalid capture: %+v %v", capture, err)
		}
		attempt := capture.Capture.Attempts[0]
		if attempt.Response.Status != 400 || attempt.Request.Message.BodyBase64 == nil {
			t.Fatal("capture lost the observed response/body")
		}
		body, err := base64.StdEncoding.Strict().DecodeString(*attempt.Request.Message.BodyBase64)
		if err != nil {
			t.Fatal(err)
		}
		parsed, err := url.Parse(attempt.Request.URL)
		if err != nil {
			t.Fatal(err)
		}
		mu.Lock()
		same := bytes.Equal(body, bodies[parsed.Path])
		mu.Unlock()
		if !same || contentdigest.Bytes(payload) != material.Digest {
			t.Fatal("retained capture differs from actual wire bytes")
		}
		runID = capture.Origin.RunID
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 2 || !bytes.Equal(bodies["/api/structured"], []byte(structured)) || !bytes.Equal(bodies["/api/exact"], []byte(exact)) {
		t.Fatalf("transmitted body mismatch: %v", bodies)
	}
	status := waitForFindingsRun(t, h.findingsHarness, runID, "succeeded")
	if len(status.Attempts) != 1 {
		t.Fatal("unexpected retry")
	}
	metrics := status.Attempts[0].Metrics
	if metrics == nil || !metrics.ReportsComplete || metrics.ToolCalls != 8 || metrics.ToolFailures != 2 {
		t.Fatalf("argument accounting: %+v, want 8 calls including passthrough / 2 failures", metrics)
	}
	var captures int
	if err := pool.QueryRow(h.ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind='run' AND scope_id=$1 AND namespace='http-exchanges'`, runID).Scan(&captures); err != nil {
		t.Fatal(err)
	}
	if captures != 2 || len(h.gateway.Failures()) != 0 || h.gateway.CompletedStages() != 1 {
		t.Fatal("duplicate publication or incomplete Gateway script", h.gateway.Failures())
	}
}

func requireArgumentDiagnostic(tool, code, field, expected string) func(map[string]any) error {
	return func(request map[string]any) error {
		responses := findingsToolResponses(request, tool)
		if len(responses) != 1 {
			return fmt.Errorf("missing first rejected %s call", tool)
		}
		detail, _ := responses[0]["error"].(map[string]any)
		encoded, _ := json.Marshal(detail)
		if detail["code"] != code || detail["field"] != field || detail["expected_type"] != expected || detail["actual_type"] != "string" || detail["retryable"] != false || bytes.Contains(encoded, []byte("secret-canary")) {
			return fmt.Errorf("unsafe/inexact argument diagnostic: %s", encoded)
		}
		return nil
	}
}
